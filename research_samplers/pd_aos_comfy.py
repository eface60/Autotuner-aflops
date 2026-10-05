# -*- coding: utf-8 -*-
"""PD-AOS as a first-class ComfyUI sampler name.

WHY THIS FILE EXISTS.  PD-AOS is normally used through its own Sampler node, which returns a SAMPLER object.  This
module makes the same sampler selectable in **`KSamplerSelect`** (and therefore in the plain `KSampler`, and in
anything else that lists sampler names), by registering the name and patching the function ComfyUI resolves it to.

READ FROM COMFYUI'S OWN SOURCE, because the three ends are not the same object and one of them is easy to miss:
  * `KSamplerSelect`'s combo is `comfy.samplers.SAMPLER_NAMES`;
  * `SAMPLER_NAMES = KSAMPLER_NAMES + [...]` is a NEW list, so `KSAMPLER_NAMES` and `KSampler.SAMPLERS` need their
    own update;
  * `comfy.samplers.sampler_object(name)` -> `ksampler(name)` -> `getattr(comfy.k_diffusion.sampling, "sample_" + name)`,
    so the FUNCTION has to exist under that exact attribute name or the dropdown entry raises when selected.

THE ALIAS DELEGATES, IT DOES NOT REIMPLEMENT.  Our node already returns the sampler object this needs, so this
module builds that object once and forwards to its `sampler_function`.  A second implementation of the engine here
would be a copy that drifts -- which is the failure this project has paid for repeatedly.

Nothing is written outside this repository: the registration happens in memory, at import time, by appending to
ComfyUI's own lists and setting one attribute on `comfy.k_diffusion.sampling`.
"""

import inspect
import logging

SAMPLER_NAME = "pd_aos"          # the dropdown entry, identifier-safe like the `a_flops`/`a_euler` precedent
PRODUCT_NAME = "PD-AOS"
_KDIFF_FUNC = "sample_" + SAMPLER_NAME

_sampler_fn = None
_build_error = None


def _default_sampler_function():
    """Build our node's sampler object once, using ITS OWN defaults (which are the engine's decided values).

    The node's `get_sampler` takes widget values; every one of them has a default, so it can be called with none.
    If that ever stops being true the error is RECORDED rather than swallowed, and the rig prints it -- a dropdown
    entry that silently falls back to something else is the defect this project keeps re-learning.
    """
    global _sampler_fn, _build_error
    if _sampler_fn is not None or _build_error is not None:
        return _sampler_fn
    try:
        # `..nodes`: nodes.py sits at the PACKAGE ROOT, not inside `research_samplers/`.  The first version
        # imported `.nodes`, so the registration passed every static check while every DYNAMIC check failed with
        # ModuleNotFoundError -- a half-wired state worth remembering: the wiring looked complete from outside.
        from ..nodes import ResearchAFlopsSampler
        node = ResearchAFlopsSampler()
        fn = getattr(node, ResearchAFlopsSampler.FUNCTION)
        # THE NODE'S OWN DECLARED DEFAULTS, read out of INPUT_TYPES -- not `None`.  The first version passed None
        # for every required parameter and the node raised `KeyError: 'log_errors'`: "use its own defaults" has to
        # mean reading them, and a required widget with a default is still required to the CALLER.
        declared = {}
        try:
            for k, spec in (ResearchAFlopsSampler.INPUT_TYPES().get("required") or {}).items():
                if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
                    if "default" in spec[1]:
                        declared[k] = spec[1]["default"]
        except Exception:                                        # noqa: BLE001 - absence is reported below
            pass
        kwargs, unresolved = dict(declared), []
        for k in (ResearchAFlopsSampler.INPUT_TYPES().get("required") or {}):
            if k not in kwargs:
                unresolved.append(k)          # a required widget with NO declared default: disclosed, not guessed
        # `get_sampler(self, **kw)` indexes `kw["log_errors"]` DIRECTLY, so its keyword-only inputs never appear
        # in `inspect.signature` as parameters -- calling it with nothing raised `KeyError: 'log_errors'`.
        # The node's OWN declared defaults are therefore what has to be passed, which is what this module's
        # docstring claims and what the first two versions only appeared to do.
        sampler = fn(**kwargs) if kwargs else fn()
        # A NODE'S FUNCTION RETURNS A TUPLE OF ITS OUTPUTS: `get_sampler` returns `(sampler,)`.  The previous
        # version did `getattr(sampler, "sampler_function", sampler)` on the TUPLE, so it handed back the tuple and
        # the dynamic call died with "'tuple' object is not callable" while every static check passed.
        if isinstance(sampler, (tuple, list)) and sampler:
            sampler = sampler[0]
        if unresolved:
            logging.info("[PD-AOS] parameters with no declared default, passed as None: %s", unresolved)
        # a ComfyUI SAMPLER is either a KSAMPLER (with .sampler_function) or a bare callable
        _sampler_fn = getattr(sampler, "sampler_function", sampler)
    except Exception as exc:                                     # noqa: BLE001 - recorded, never silent
        _build_error = "%s: %s" % (type(exc).__name__, exc)
        _sampler_fn = None
    return _sampler_fn


def sample_pd_aos(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    """The name ComfyUI resolves for `pd_aos`.  Signature and return follow `KSAMPLER.sample`'s contract."""
    fn = _default_sampler_function()
    if fn is None:
        raise RuntimeError(
            "PD-AOS could not build its sampler from the node defaults (%s). Use the PD-AOS/PD-AOS Sampler node "
            "instead, or report this -- the dropdown entry must not silently do something else." % _build_error)
    return fn(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **kwargs)


def register(verbose=True):
    """Idempotently add PD-AOS to every container the UI and the resolver read."""
    added = []
    try:
        import comfy.k_diffusion.sampling as kds
        if getattr(kds, _KDIFF_FUNC, None) is not sample_pd_aos:
            setattr(kds, _KDIFF_FUNC, sample_pd_aos)
            added.append("comfy.k_diffusion.sampling." + _KDIFF_FUNC)
        import comfy.samplers as cs
        containers = []
        for holder in (cs, getattr(cs, "KSampler", None)):
            if holder is None:
                continue
            for attr in ("SAMPLER_NAMES", "KSAMPLER_NAMES", "SAMPLERS"):
                lst = getattr(holder, attr, None)
                if isinstance(lst, list):
                    containers.append((holder.__name__ if hasattr(holder, "__name__") else str(holder), attr, lst))
        for holder, attr, lst in containers:
            if SAMPLER_NAME not in lst:
                lst.append(SAMPLER_NAME)
                added.append("%s.%s" % (holder, attr))
    except Exception as exc:                                     # noqa: BLE001
        logging.warning("[PD-AOS] registration failed: %s: %s", type(exc).__name__, exc)
        return []
    if verbose and added:
        logging.info("[PD-AOS] registered %s as `%s` (%s)", PRODUCT_NAME, SAMPLER_NAME, ", ".join(added))
    return added


def status():
    """For the rig and for support: what the three ends look like right now."""
    try:
        import comfy.k_diffusion.sampling as kds
        import comfy.samplers as cs
    except Exception as exc:                                     # noqa: BLE001
        return {"error": str(exc)}
    return {
        "function_patched": getattr(kds, _KDIFF_FUNC, None) is sample_pd_aos,
        "SAMPLER_NAMES": SAMPLER_NAME in getattr(cs, "SAMPLER_NAMES", []),
        "KSAMPLER_NAMES": SAMPLER_NAME in getattr(cs, "KSAMPLER_NAMES", []),
        "KSampler.SAMPLERS": SAMPLER_NAME in getattr(getattr(cs, "KSampler", None), "SAMPLERS", []),
        "sampler_builds": _default_sampler_function() is not None,
        "build_error": _build_error,
    }

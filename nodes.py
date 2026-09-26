import json
import comfy.samplers
from .research_samplers import aflops as aflops_mod
from .research_samplers.aflops import aflops_engine, ENGINE_DEFAULTS

OPT_TYPE = "RS_SAMPLER_OPT"


def _f(v):
    return float(v[0]) if isinstance(v, (list, tuple)) else float(v)


def _b(v):
    v = v[0] if isinstance(v, (list, tuple)) else v
    return bool(v)


def _s(v):
    return v[0] if isinstance(v, (list, tuple)) else v


def _opts(v):
    """Normalize the options+ input: None, a single dict, or a list of dicts."""
    if v is None:
        return []
    if isinstance(v, dict):
        return [v]
    return [o for o in v if o]


def _sig_tensor_list(v):
    """SIGMAS input (tensor or list) -> descending float list or None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and v and not isinstance(v[0], (int, float)):
        v = v[0]
    try:
        lst = [float(x) for x in v]
    except Exception:
        return None
    if len(lst) < 4 or lst[0] <= lst[1]:
        return None
    return lst




class ResearchAFlopsSampler:
    DESCRIPTION = ("A-FloPS (Adaptive Flow Path Sampler). Implements the core method "
                   "from Jin et al. (AAAI 2026, arXiv:2509.00036): the velocity field "
                   "is decomposed into a linear drift and a slowly-varying residual via "
                   "an adaptive coefficient, then integrated with an exponential "
                   "integrator. Pure single-solver engine: every step runs the "
                   "A-FloPS formula with ZERO extra model calls, and the per-pixel "
                   "adaptive order ladder feeds the residual integrator. Without extra "
                   "inputs it runs fully automatic: stochasticity (a minimal-noise, "
                   "escape-based schedule that fades late) controls how much random "
                   "noise is added each step, and everything else (noise color, anomaly "
                   "detection, the local per-region schedule field with its spatial "
                   "fragility prior, flow/VE detection) is self-tuned from the probes. "
                   "Connect a MODEL for auto-detection. "
                   "Connect Options nodes to override any group of settings.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stochasticity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much random noise to add during sampling. 0 = fully deterministic, 1 = maximum noise. Adds variation and can help with detail, but too much can blur. Fades to zero toward the end of the run. Overridden if a Noise Options node is connected."}),
                "log_errors": ("BOOLEAN", {"default": False,
                    "tooltip": "Print per-step decisions to the console (solver chosen, errors, guard trips, etc.). Also feeds the Error Report node. Useful for debugging; off for normal use."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Connect the MODEL for auto-tuning. A-FloPS will detect the model type (flow vs VE), read sigma_data for correct SNR math, detect video latents (auto-enables temporal noise shaping), and detect distilled models (turbo/schnell/lightning). All auto-tuning can be overridden by Options nodes. Without this, A-FloPS uses a sigma-magnitude heuristic."}),
                "options+": (OPT_TYPE, {"tooltip": "Connect any number of A-FloPS Options nodes here. Each connected group overrides the automatic settings for its fields."}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"

    def get_sampler(self, **kw):
        log = _b(kw["log_errors"])
        merged = dict(ENGINE_DEFAULTS)
        for o in _opts(kw.get("options+", kw.get("options"))):
            for k, v in o.items():
                merged[k] = v
        # Extract model info for auto-tuning (optional MODEL input).
        model = kw.get("model")
        if isinstance(model, (list, tuple)):
            model = model[0] if model else None
        if model is not None:
            mi = aflops_mod._extract_model_info(model)
            if mi is not None:
                merged["_model_info"] = mi
                if log:
                    import logging as _lg
                    _lg.info("[A-FloPS] model info: is_flow=%s sigma_data=%s "
                             "is_video=%s is_distilled=%s image_model=%s",
                             mi.get("is_flow"), mi.get("sigma_data"),
                             mi.get("is_video"), mi.get("is_distilled"),
                             mi.get("image_model"))
        if merged["eta"] is None:
            merged["eta"] = _f(kw["stochasticity"])
        return (comfy.samplers.KSAMPLER(aflops_engine,
                                        extra_options={"cfg": merged, "log_errors": log}),)


# ---------------------------------------------------------------------------
# options nodes
# ---------------------------------------------------------------------------


class ResearchAFlopsAutotuner:
    DESCRIPTION = ("A-FloPS Autotuner: probes + schedule fine-tune in one node. "
                   "Connect the guider the run will use and ANY scheduler's sigmas "
                   "(e.g. BasicScheduler \"simple\"); the node measures the model and "
                   "this prompt's trajectory at the schedule's noise levels (tiny "
                   "probe latents, cached per model+prompt+schedule), then re-allocates "
                   "the steps of YOUR sigma list where the measured curvature needs "
                   "them (error-equidistributed; flat curvature reproduces your list "
                   "exactly). Outputs the fine-tuned sigmas for the sampler and the "
                   "probe options for the A-FloPS Sampler node (connect its options "
                   "output to the Sampler's options+ input). Elementary knobs only: "
                   "probe steps/resolution and the shift bias (1.0 = pure error-optimal "
                   "placement; >1 tilts steps toward high sigma / structure, <1 toward "
                   "low sigma / detail).")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER", {"tooltip": "The guider the run will use (CFGGuider / BasicGuider / DualCFGGuider). The autotuner borrows its ModelPatcher, conditioning and live cfg scale to measure the model AND this prompt, so the fine-tune matches the run's real sampling configuration. Runs right here, before the KSampler."}),
                "sigmas": ("SIGMAS", {"tooltip": "The schedule to fine-tune — connect any scheduler's output (BasicScheduler \"simple\", Karras, ...). Your list IS the schedule: the probe re-allocates its steps by measured curvature, and with no curvature evidence it passes through untouched."}),
                "probe_steps": ("INT", {"default": 10, "min": 5, "max": 16, "step": 1,
                    "tooltip": "Probe trajectory steps (both probes share this). Minimum 5: the curvature curve has K-2 points and the schedule fine-tune needs at least 3, so below 5 the per-prompt measurement cannot drive the schedule. More = finer per-prompt measurement, slightly longer precompute on the first run."}),
                "probe_resolution": ("INT", {"default": 48, "min": 16, "max": 128, "step": 8,
                    "tooltip": "Spatial size of the probe latent (both probes). Smaller = faster probe; 48 is the established sweet spot."}),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 10.0, "step": 0.05,
                    "tooltip": "Density bias on the fine-tune. 1.0 = pure error-optimal re-allocation of your schedule (flat measured curvature then reproduces your list exactly). >1 tilts steps toward high sigma (structure emphasis), <1 toward low sigma (detail emphasis)."}),
            },
        }

    RETURN_TYPES = ("SIGMAS", OPT_TYPE)
    RETURN_NAMES = ("sigmas", "options")
    FUNCTION = "autotune"

    def autotune(self, guider, sigmas, probe_steps, probe_resolution, shift):
        import torch as _t
        import logging as _lg
        if isinstance(guider, (list, tuple)):
            guider = guider[0] if guider else None
        ext = _sig_tensor_list(sigmas)
        steps = int(probe_steps[0] if isinstance(probe_steps, (list, tuple))
                    else probe_steps)
        res = int(probe_resolution[0] if isinstance(probe_resolution,
                                                    (list, tuple))
                  else probe_resolution)
        steps = max(5, min(16, steps))
        sh = _f(shift)
        opts = {
            "probe_enabled": True,
            "probe_steps": steps,
            "probe_resolution": res,
            "cond_probe_enabled": True,
            "cond_probe_steps": steps,
            "cond_probe_resolution": res,
            "cond_probe_weight": 0.5,
            "_user_shift": float(sh),
        }
        if guider is None:
            # Nothing to measure: pass the schedule through untouched.
            _lg.warning("[A-FloPS-autotuner] no guider connected; sigmas "
                        "passed through unmodified")
            opts["_shift_schedule_node"] = {"mode": "off", "adopted": False,
                                            "base": "external"}
            return (sigmas, opts)
        if ext is None:
            _lg.warning("[A-FloPS-autotuner] unusable sigmas input; passed "
                        "through unmodified")
            opts["_shift_schedule_node"] = {"mode": "off", "adopted": False,
                                            "base": "external"}
            return (sigmas, opts)

        # 1) measure: model probe then per-prompt cond probe, both on the
        #    incoming schedule (schedule-keyed caches, no double probe).
        profile_m = None
        profile_c = None
        try:
            profile_m, _ = aflops_mod._run_node_side_model_probe(
                guider, opts, sigmas=ext)
        except Exception as e:
            _lg.warning("[A-FloPS-autotuner] model probe failed: %s", e)
        try:
            probe_cfg = aflops_mod._guider_probe_cfg(guider)
            if probe_cfg is None:
                probe_cfg = 7.0
            opts["cond_probe_sig"] = aflops_mod._guider_cond_signature(
                guider, probe_cfg)
            profile_c, _ = aflops_mod._run_node_side_cond_probe_guider(
                guider, opts, sigmas=ext)
        except Exception as e:
            _lg.warning("[A-FloPS-autotuner] cond probe failed: %s", e)

        # 2) fine-tune: the per-prompt profile wins when it can drive the
        #    schedule (>= 3 curvature points; guaranteed at probe_steps >= 5),
        #    else the model profile, else untouched passthrough.
        _p = None
        for _c in (profile_c, profile_m):
            if (isinstance(_c, dict) and _c.get("space") == "flow"
                    and len(_c.get("curv_curve") or []) >= 3):
                _p = _c
                break
        sched_info = {"mode": "off", "adopted": False, "base": "external"}
        grid = None
        if _p is not None:
            grid = aflops_mod._build_curvature_schedule(
                _p, len(ext) - 1, 1.0, shift=float(sh), base_sigmas=ext)
        if grid is not None and grid.numel() == len(ext):
            sched_info = {"mode": "curvature", "adopted": True,
                          "base": "external", "shift": round(float(sh), 4)}
            _lg.info("[A-FloPS-autotuner] schedule fine-tuned from probe "
                     "curvature (shift bias=%.3f, %d steps)",
                     float(sh), len(ext) - 1)
            out_grid = grid
        else:
            _lg.info("[A-FloPS-autotuner] no usable curvature evidence; "
                     "schedule passed through")
            out_grid = _t.tensor(ext, dtype=_t.float32)
        opts["_shift_schedule_node"] = sched_info
        return (out_grid, opts)


class ResearchAFlopsErrorReport:
    DESCRIPTION = ("Outputs the per-step internal decisions from the last run as JSON: "
                   "solver chosen, errors, guard trips, anomaly verdicts, call counts, "
                   "etc. Connect the sampled latent to 'trigger' to force this node to "
                   "run after sampling. Only updates when the graph re-runs.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"trigger": ("LATENT", {"tooltip": "Connect the sampled latent here to force this node to run after sampling completes."})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("errors_json",)
    FUNCTION = "report"
    OUTPUT_NODE = True

    def report(self, trigger=None):
        # Include the probe profile cache summary so you can confirm a profile
        # was built and inspect its calibrations.
        probe_summary = {}
        try:
            for key, prof in aflops_mod._PROBE_PROFILES.items():
                # NOTE: med_local_err / guard_sig_* / log_gap_* are part of
                # the local field's auto-tune evidence; they used to be
                # dropped from this summary, which made the derived gains
                # unauditable from the report JSON.
                probe_summary[str(key)] = {
                    "version": prof.get("version"),
                    "space": prof.get("space"),
                    "n_probe_steps": prof.get("n_probe_steps"),
                    "is_distilled_eff": prof.get("is_distilled_eff"),
                    "safe_fresh_frac": prof.get("safe_fresh_frac"),
                    "med_jump": prof.get("med_jump"),
                    "med_curv": prof.get("med_curv"),
                    # the trajectory evidence curves the shift/schedule
                    # recommendation consumes -- without them the "how fast can
                    # we drop sigma" question cannot be answered from the log.
                    "sigma_curve": prof.get("sigma_curve"),
                    "curv_curve": prof.get("curv_curve"),
                    "jump_curve": prof.get("jump_curve"),
                    # per-interval error-model evidence (err vs curv vs gap):
                    # the schedule's density exponent a/p is fitted from
                    # these; without them the exponent cannot be measured
                    # from the report at all.
                    "local_err_curve": prof.get("local_err_curve"),
                    "gap_curve": prof.get("gap_curve"),
                    "real_gap_curve": prof.get("real_gap_curve"),
                    "slope_curve": prof.get("slope_curve"),
                    "whitepoint": prof.get("whitepoint"),
                    "med_local_err": prof.get("med_local_err"),
                    "guard_sig_med": prof.get("guard_sig_med"),
                    "guard_sig_p90": prof.get("guard_sig_p90"),
                    "log_gap_probe": prof.get("log_gap_probe"),
                    "log_gap_real": prof.get("log_gap_real"),
                    # v7: the escape-threshold derivation inputs (stalled =
                    # 1/4 of the measured normal jump, flat = 1/2 of the
                    # measured normal detail) and the late-fade anchor, so
                    # the adaptive-eta math is fully auditable from the
                    # report instead of only from the in-memory profile.
                    "med_jump_rel": prof.get("med_jump_rel"),
                    "med_detail_abs": prof.get("med_detail_abs"),
                    "conv_sigma": prof.get("conv_sigma"),
                    "osc_sigma": prof.get("osc_sigma"),
                    "early_frac": prof.get("early_frac"),
                    "conv_step": prof.get("conv_step"),
                    "fragility": prof.get("fragility"),
                    "calib": prof.get("calib"),
                    # v6: the local-field evidence the model probe
                    # contributes -- trajectory state, the run-level
                    # direction feedback and the cfg scale the probe ran
                    # at.  Without these the derived gains were only half
                    # auditable from this report.
                    "trajectory_state": prof.get("trajectory_state"),
                    "dir_real": prof.get("dir_real"),
                    "probe_cfg": prof.get("probe_cfg"),
                }
        except Exception:
            pass
        # Per-prompt (conditioned) probe profiles: refreshed every generation
        # the Cond Probe node runs, keyed by (model, prompt signature).
        cond_probe_summary = {}
        try:
            # Defensive: _COND_PROBE_PROFILES is not defined until the
            # cond-probe engine support lands.  getattr avoids an
            # AttributeError that would silently empty this section.
            _cond_store = getattr(aflops_mod, '_COND_PROBE_PROFILES', {})
            for key, prof in _cond_store.items():
                cond_probe_summary["%s | prompt %s" % (key[0], key[1])] = {
                    "cond_sig": prof.get("cond_sig"),
                    "space": prof.get("space"),
                    "n_probe_steps": prof.get("n_probe_steps"),
                    "med_jump": prof.get("med_jump"),
                    "med_curv": prof.get("med_curv"),
                    "sigma_curve": prof.get("sigma_curve"),
                    "curv_curve": prof.get("curv_curve"),
                    "jump_curve": prof.get("jump_curve"),
                    "local_err_curve": prof.get("local_err_curve"),
                    "gap_curve": prof.get("gap_curve"),
                    "real_gap_curve": prof.get("real_gap_curve"),
                    "slope_curve": prof.get("slope_curve"),
                    "safe_fresh_frac": prof.get("safe_fresh_frac"),
                    "is_distilled_eff": prof.get("is_distilled_eff"),
                    "med_local_err": prof.get("med_local_err"),
                    "guard_sig_med": prof.get("guard_sig_med"),
                    "guard_sig_p90": prof.get("guard_sig_p90"),
                    "log_gap_probe": prof.get("log_gap_probe"),
                    "log_gap_real": prof.get("log_gap_real"),
                    "med_jump_rel": prof.get("med_jump_rel"),
                    "med_detail_abs": prof.get("med_detail_abs"),
                    "conv_sigma": prof.get("conv_sigma"),
                    "osc_sigma": prof.get("osc_sigma"),
                    "early_frac": prof.get("early_frac"),
                    "conv_step": prof.get("conv_step"),
                    "fragility": prof.get("fragility"),
                    "gap": prof.get("gap"),
                    "probe_cfg": prof.get("probe_cfg"),
                    "dir_real": prof.get("dir_real"),
                    "trajectory_state": prof.get("trajectory_state"),
                    "calib": prof.get("calib"),
                    "focused": prof.get("focused"),
                }
        except Exception:
            pass
        return (json.dumps({"aflops_cfg": aflops_mod._LAST_CFG,
                            "probe_profiles": probe_summary,
                            "cond_probe_profiles": cond_probe_summary,
                            "aflops": aflops_mod._LAST_ERRORS}, indent=2),)


NODE_CLASS_MAPPINGS = {
    "ResearchAFlopsSampler": ResearchAFlopsSampler,
    "ResearchAFlopsAutotuner": ResearchAFlopsAutotuner,
    "ResearchAFlopsErrorReport": ResearchAFlopsErrorReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResearchAFlopsSampler": "A-FloPS Sampler",
    "ResearchAFlopsAutotuner": "A-FloPS Autotuner (probe + schedule)",
    "ResearchAFlopsErrorReport": "A-FloPS Error Report",
}

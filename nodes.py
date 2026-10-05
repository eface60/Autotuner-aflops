import json
import comfy.samplers
from .research_samplers import aflops as aflops_mod
from .research_samplers import paper_aeuler as paper_aeuler_mod
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
                   "inputs it runs fully automatic: noise colour, anomaly "
                   "detection, the local per-region schedule field with its spatial "
                   "fragility prior, and flow/VE detection are all self-tuned from the probes. "
                   "Connect a MODEL for auto-detection. "
                   "Connect Options nodes to override any group of settings.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "log_errors": ("BOOLEAN", {"default": False,
                    "tooltip": "Writes what the sampler decides at every step to the console, and fills the A-FloPS Error Report node. Leave it on while you are testing a prompt; off for normal use."}),
                # THE BOUND A/B WAS HERE AND IS GONE (2026-10-04).  `order_bound_fix` was a Sampler
                # widget for one turn; the A/B ran on three real pairs, and the corrected remainder
                # scale that it toggled is now the ONLY path in `_pixel_order_bound`.  Operator:
                # *"with removal we both mean also implement the new code, right?  Old code goes too."*
                # **A SIDE BENEFIT: removing it restores the shift-safety the five-switch move had
                # won.** The widget order is `[log_errors]` again, which IS a prefix of the published
                # order, so a published-era graph's surplus values are discarded instead of landing on
                # a live switch.  See NOTES for what the A/B measured -- it was INERT on both
                # configurations tried, which says the per-pixel criterion is not the binding
                # constraint, not that the correction is wrong.
                # THE FIVE FINISHED SWITCHES MOVED OFF THIS NODE (2026-10-04): `lf_ab`,
                # `order_by_bound`, `order_floor`, `an_enable` and `wmax_eg_dir` are no longer
                # widgets here.  Each had reached a decision -- by measurement plus the
                # operator's eye -- so each is fixed at its ENGINE DEFAULT and driven, when
                # anyone wants to drive it, from the **A-FloPS Debug Options** node, whose
                # `options` output plugs into this node's `options+` input.  The tooltips moved
                # with the switches; nothing was dropped.
                #
                # WHY THEY ALL MOVED AT ONCE, and this is not tidiness.  ComfyUI stores a UI
                # graph's widget values as a POSITIONAL list (`widgets_values`), so removing a
                # widget from the MIDDLE of this list re-maps every later value onto the wrong
                # widget when an existing saved workflow is loaded -- silently, and onto
                # plausible values.  `wmax_eg_dir` is last in the old order, so keeping it (or
                # any other non-prefix subset) would have handed old graphs `wmax_eg_dir = 1.0`
                # from what used to be `lf_ab = True`: the most aggressive extrapolation, on the
                # knob that changes image STRUCTURE.  `log_errors` is at index 0, so keeping
                # ONLY it means an old list's first value still lands correctly and the surplus
                # is discarded -- existing workflows keep working and pick up the engine
                # defaults, which is the intended outcome of the move.
                #
                # The `if "x" in kw:` blocks in `get_sampler` are kept as BACKWARD-COMPATIBLE
                # OVERRIDES: an API-format graph that still passes the old names, or an Options
                # node that sets the keys, is honoured rather than silently ignored.
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
        # The corrector A/B arm.  NO LONGER A WIDGET (release build): it is fixed at its
        # engine default.  This branch stays so a saved graph, or an Options node that sets
        # the key, is honoured rather than silently ignored -- the normal case now is that
        # it never fires.  Set BEFORE the Options nodes are applied so a connected Options
        # node can still override it.
        if "corrector_ab" in kw:
            merged["corrector_ab"] = _b(kw["corrector_ab"])
        # The order/extrapolation A/B arm (same convention, same reason it is not a widget).
        if "order_ab" in kw:
            merged["order_ab"] = _b(kw["order_ab"])
        # The forced per-pixel order floor.  WAS a Sampler widget; now on the Debug Options node
        # (see the INPUT_TYPES note for why all five moved together).  Kept here as a
        # backward-compatible override, so an API-format graph or an Options node still drives
        # it -- one switch, one place.
        if "order_floor" in kw:
            merged["order_floor"] = int(kw["order_floor"])
        # Which per-pixel ORDER CRITERION runs.  Same convention: Debug Options node now, this
        # override kept for saved graphs and Options nodes.
        if "order_by_bound" in kw:
            merged["order_by_bound"] = _b(kw["order_by_bound"])
        # The SIGN of the endgame-error response.  Moved to the Debug Options node with the other
        # four.  Clamped here as well as in the engine, so a saved graph from before the clamp
        # cannot inject an out-of-range value.
        if "wmax_eg_dir" in kw:
            merged["wmax_eg_dir"] = max(-1.0, min(1.0, _f(kw["wmax_eg_dir"])))
        # The two DERIVED-CONSTANT arms.  Independent, so each can be flipped alone; the
        # derivation logs BOTH the shipped and the derived value regardless, so one run
        # shows the delta even with the switch off.
        # NOTE: pct_derived / wmax_bound are NOT WIDGETS ON EITHER NODE in the release build
        # (the Autotuner used to own them).  They are fixed at their engine defaults and are
        # kept here only so Options nodes and saved graphs can still drive them.  The
        # Autotuner's `options` output is applied BELOW, i.e. AFTER these assignments, so an
        # Autotuner copy would silently override a Sampler copy anyway -- two switches with
        # one name, where one of them quietly does nothing.  One switch, one place.
        # DERIVED CONSTANTS and the CUT VARIABLE are NO LONGER WIDGETS -- both A/Bs
        # reached their verdicts, so they are fixed settings in ENGINE_DEFAULTS
        # (`derived_ab = True`, `cut_var = "midpoint"`).  The two blocks below are
        # kept deliberately as BACKWARD-COMPATIBLE OVERRIDES: a saved graph that
        # still carries the old widget value, or an Options node that sets the key,
        # is honoured instead of being silently ignored.  If neither supplies them
        # (the normal case now) these branches never fire and the defaults stand.
        # Both keys remain stamped into `_LAST_CFG`, so every log still states which
        # behaviour produced it.
        if "derived_ab" in kw:
            merged["derived_ab"] = _b(kw["derived_ab"])
        if "cut_ab" in kw:
            merged["cut_var"] = "midpoint" if _b(kw["cut_ab"]) else "corrected"
        # self-describing about which behaviour produced it.
        # LOCAL-FIELD arm.  Moved to the Debug Options node with the other four; kept as an
        # override so a saved graph or an Options node still drives it.
        if "lf_ab" in kw:
            merged["lf_enable"] = _b(kw["lf_ab"])
        # THE ANOMALY DETECTOR'S ARM.  Moved to the Debug Options node with the other four and
        # DEFAULTING TO ON, on the operator's own A/B: *"i think my preference was on, there was
        # a prompt where it actually didn't fire, but for when it does, it does generally fix
        # things in a way regular sampling doesn't seem to be able to recover in time."*
        # It gates a path that can damp a small region for several steps, so testing it by eye
        # needs the same prompt and seed with the switch flipped -- which the Debug node still
        # provides.  The name matches the engine key exactly, so there is no mapping to get wrong.
        if "an_enable" in kw:
            merged["an_enable"] = _b(kw["an_enable"])
        # THE BOUND A/B IS GONE (2026-10-04): its widget and its mapping were removed in the same
        # change that made the corrected remainder scale the only path in `_pixel_order_bound`.  No
        # `if "order_bound_fix" in kw:` override is kept, because there is no longer a second
        # behaviour for it to select -- a leftover key would silently mean nothing.

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
        if merged["eta"] is None and kw.get("stochasticity") is not None:
            # The `stochasticity` WIDGET IS NO LONGER EXPOSED in the release build.  It was
            # measured dead before hiding it: across 12 consecutive runs whose cfg.eta reads
            # 0.1 -- and one that reads 1.0 -- EVERY step reports eta_eff 0.0, so the slider
            # never changed the output.  The key is still honoured here for saved graphs and
            # for Options nodes that set it programmatically; with neither, eta keeps its
            # engine default.
            merged["eta"] = _f(kw["stochasticity"])
        return (comfy.samplers.KSAMPLER(aflops_engine,
                                        extra_options={"cfg": merged, "log_errors": log}),)


class ResearchAFlopsDebugOptions:
    DESCRIPTION = ("A-FloPS Debug Options: the FINISHED switches, moved off the Sampler. "
                   "Every one of these reached a decision -- by measurement plus the operator's "
                   "eye -- so none of them belongs on the node you use every day. Connect this "
                   "node's `options` output to the A-FloPS Sampler's `options+` input to drive "
                   "them again: for an A/B on the same prompt and seed, to re-test a verdict on "
                   "a new model, or to reproduce an old run. With this node disconnected the "
                   "Sampler uses the engine defaults, which ARE the decided values. "
                   "The defaults here are READ FROM THE ENGINE, not repeated, so the two cannot "
                   "drift apart.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        # DEFAULTS COME FROM THE ENGINE (rule: one value, one place).  A literal here would be a
        # second copy of a shipped default, and two copies of one value is how this project has
        # shipped a no-op before.  `lf_ab` is the UI name for the engine's `lf_enable`.
        return {
            "required": {
                "lf_ab": ("BOOLEAN", {"default": bool(ENGINE_DEFAULTS.get("lf_enable", True)),
                    "tooltip": "Turns the local field on or off. ON: the sampler also makes small local corrections, in the areas where its own step is least reliable. OFF: no local corrections; the step placement, the corrector and the order are untouched. DECIDED ON: preferred for krea2 by eye on the LF x ORDER 2x2 (2026-10-04)."}),
                "order_by_bound": ("BOOLEAN", {"default": bool(ENGINE_DEFAULTS.get("order_by_bound", True)),
                    "tooltip": "Chooses how the sampler picks the per-pixel ORDER. Order = how many previous steps the sampler looks at when it works out where the image is heading. OFF: it takes the number of previous steps that predicts the next one best. ON: it takes the highest number of previous steps that is still safe for that pixel, so smooth areas get a high order and busy or detailed areas keep a low one. DECIDED ON (release ruling, 2026-10-03)."}),
                "order_floor": ("INT", {"default": int(ENGINE_DEFAULTS.get("order_floor", 6)),
                    "min": 1, "max": 6, "step": 1,
                    "tooltip": "Sets the lowest order the sampler may use, per pixel. 1 = the sampler decides by itself, and this setting does nothing. Above 1, every pixel is pushed up to at least this value, which makes the sampler follow the direction of the trajectory harder. Two safeguards still apply: an order is refused when the step would come out too large, and a pixel the sampler has just reset stays at 1. DECIDED 6, and measured: the demand is largely REFUSED by the safety path, which is why the order actually used tops out below it -- see NOTES, 'the floor is a DEMAND'."}),
                "an_enable": ("BOOLEAN", {"default": bool(ENGINE_DEFAULTS.get("an_enable", True)),
                    "tooltip": "Turns the ANOMALY DETECTOR on or off. ON: at each step the sampler looks for a few pixels whose local change is far outside its own neighbourhood -- an abrupt, spatially concentrated event -- and damps the local field's correction on that small region for the next several steps. OFF: that detection never runs and nothing is damped. DECIDED ON by the operator's own A/B: there are prompts where it does not fire, but where it does it fixes things that regular sampling does not recover from in time."}),
                "wmax_eg_dir": ("FLOAT", {"default": float(ENGINE_DEFAULTS.get("wmax_eg_dir", 0.0)),
                    "min": -1.0, "max": 1.0, "step": 0.25,
                    "tooltip": "Which way the sampler reacts when its own end-of-run error measurement comes out HIGH. 0 (shipped default) = no reaction; the error-based limit acts alone. +1 = it allows the most extrapolation then. -1 = it does the OPPOSITE and allows the least extrapolation then. This setting decides WHICH PIXELS get the higher extrapolation order, so it changes the STRUCTURE of the image more than its overall detail or sharpness. Measured: -1 is worse than +1 against an exact ODE solution on three analytic flows (by up to 47%); that test scores distance to an exact solution and not how an image looks, and by eye the operator's preference across many runs and two model families has been the positive side. The response saturates, so +0.5 already behaves much like +1. KEPT REACHABLE rather than hidden: the operator's own taste has run +1 while the shipped default is 0."}),
                # APPENDED LAST ON PURPOSE (rule 35): ComfyUI stores a UI graph's widget values
                # POSITIONALLY, so a widget inserted anywhere but the end re-maps every later value of
                # an already-saved workflow.  The five above keep their slots exactly.
                "cond_rtol_loosen": ("FLOAT", {"default": float(ENGINE_DEFAULTS.get("cond_rtol_loosen", 1.5)),
                    "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "EXPERIMENTAL, for A/B-ing one behaviour by eye. Scales the tolerance the sampler derives from the PROMPT probe (not the model probe). 1.5 reproduces the shipped behaviour; 1.0 switches that loosening OFF, which is the arm to compare against; below 1.0 tightens instead -- a direction the shipped code cannot express at all. 0 is the tightest the downstream blend allows (about half the model-probe tolerance), not zero. The top of the range is rarely useful: the sampler mixes this value with the model-probe one and clamps the result, so on a typical run increases stop changing anything above roughly 2.5. Range derived from the run corpus, not chosen: above 8.0 nothing responds at all. NOTE: this one is measured INERT in practice -- across 31 runs on two models the tolerance the runs consumed never moved (see NOTES row 23, closed). See also the Order EMA widget, which is the one that does move the record."}),
                # APPENDED AFTER `cond_rtol_loosen`, so the five original widgets keep their positional
                # slots and every earlier graph still loads correctly (rule 35).
                "ord_ema": ("FLOAT", {"default": float(ENGINE_DEFAULTS.get("ord_ema", 0.37)),
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How fast the sampler forgets its own recent innovation estimate. It blends the previous estimate with the newest sample: 0.0 ignores the history entirely and follows the newest sample, 1.0 ignores the newest sample and never moves; 0.37, the shipped value, leans on the newest sample. Higher values make the estimate steadier and slower to react, lower values make the order decisions twitchier. RANGE IS NOT ARBITRARY: this is a blend weight between two things, so 0 to 1 is its whole meaningful span. GROUNDED, AND THE SHIPPED VALUE IS THE MEASURED OPTIMUM: reading the sampler's own logged innovation series (n=682), its step-to-step correlation is 0.630, which puts the best-fitting value at 0.370. It shipped at 0.5 -- 0.13 stiffer than that measurement wants -- until 2026-10-04, when a controlled A/B settled it: one session, one prompt, one seed at 0.5 / 0.37 / 0.5 gave a BIT-IDENTICAL replicate pair and a 33% change on the 0.37 arm, and 0.37 looked better. This widget is what that comparison was run with."}),
            },
            # CHAINING (2026-10-04).  Until this input existed only the Sampler could RECEIVE
            # options, so two options-producing nodes could not both drive a run -- the operator's
            # own report: *"i check the debug node, but it doesn't have options+ input -> i can't
            # have multiple chained options (scheduler+debugoptions) this way."*  It is OPTIONAL:
            # with nothing connected this node behaves exactly as before.
            "optional": {
                "options+": (OPT_TYPE, {"tooltip": "Connect another A-FloPS Options node here (the Autotuner's `options` output, or another Debug Options node) and this node passes its settings through, adding or overriding its own on top. Chain them in the order you want them applied: the node CLOSEST to the Sampler wins on any field two of them both set."}),
            },
        }

    RETURN_TYPES = (OPT_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "options"

    def options(self, lf_ab=True, order_by_bound=True, order_floor=6,
                an_enable=True, wmax_eg_dir=0.0,
                cond_rtol_loosen=float(ENGINE_DEFAULTS.get("cond_rtol_loosen", 1.5)),
                ord_ema=float(ENGINE_DEFAULTS.get("ord_ema", 0.37)), **kw):
        # The Sampler merges this dict into its own cfg BEFORE the Autotuner's options, so a
        # Debug node wins over a probe-side default and can be flipped without touching either
        # node's other settings.  `lf_ab` -> `lf_enable` is the ONLY rename here.
        out = {}
        # UPSTREAM FIRST, THEN OURS (see the `options+` tooltip): a chained node passes its fields
        # through untouched, and this node's own deliberate settings win where the two overlap --
        # which today is nowhere, because no other options node sets any of these six keys.
        for o in _opts(kw.get("options+", kw.get("options"))):
            if isinstance(o, dict):
                out.update(o)
        out.update({
            "lf_enable": _b(lf_ab),
            "order_by_bound": _b(order_by_bound),
            "order_floor": int(_f(order_floor)),
            "an_enable": _b(an_enable),
            "wmax_eg_dir": max(-1.0, min(1.0, _f(wmax_eg_dir))),
            # The engine clamps this one too (`_apply_ab_modes`), and its lower guard is a CORRECTNESS
            # guard rather than taste: the multiply has no lower clamp and the blend's clamp is skipped
            # when the blend is not positive, so a negative value would reach cfg as a negative tolerance.
            "cond_rtol_loosen": max(0.0, min(8.0, _f(cond_rtol_loosen))),
            # `ord_ema` WAS MISSING FROM THIS DICT, and that is the whole of the 2026-10-04 failure: the
            # widget existed in INPUT_TYPES (so the UI showed it and the API graph SENT it), the engine
            # read `cfg["ord_ema"]` -- and this method neither accepted the parameter nor emitted the key,
            # so the value landed in `**kw` and the engine fell back to its default.  The operator set
            # 0.37 correctly, the graph carried 0.37, and the run used 0.5.
            # A widget is wired only when BOTH ends exist: the engine reading the key AND the node
            # EMITTING it.  `scratch_knob_wiring_rig.py` now checks both directions (W1 engine-reads,
            # W1c node-emits), because checking one of them is what let this through.
            "ord_ema": max(0.0, min(1.0, _f(ord_ema))),
        })
        return (out,)


class ResearchPaperAEuler:
    DESCRIPTION = ("A-FloPS A-Euler - the paper's Algorithm 2 (Jin et al., AAAI 2026, "
                   "arXiv:2509.00036) applied DIRECTLY to a native flow model, with no "
                   "extras. This is the faithful reference: it factorizes the FM velocity "
                   "v = lambda*x + h with the adaptive lambda = <dv,dx>/||dx||^2 clamped "
                   "to [-1,1], solves the linear drift exactly, and extrapolates the "
                   "residual from the previous step (exponential integrator, first step "
                   "Euler). One model call per step. The method is grid-agnostic: it runs "
                   "on WHATEVER sigma list you feed it (each step's local flow-time span "
                   "is read from the model's sigma<->time map), so connect it to a "
                   "SamplerCustomAdvanced and drive it with any scheduler. For the paper's "
                   "exact uniform-flow-time placement, use the companion 'A-FloPS A-Euler "
                   "Schedule (paper)' node as the sigmas source.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(paper_aeuler_mod.sample_paper_aeuler),)


class ResearchPaperAEulerSchedule:
    DESCRIPTION = ("The paper's own schedule for A-FloPS A-Euler: sigmas placed at "
                   "UNIFORM flow time t = n/N (noise -> clean), mapped through the model's "
                   "sigma<->time map (flux_time_shift for Flux, t/(1-t) for Cosmos/RFlow). "
                   "Connect this node's SIGMAS output to SamplerCustomAdvanced (or any "
                   "sampler's sigmas input). This is the schedule Algorithm 2 assumes; the "
                   "sampler itself also accepts any other scheduler's sigmas. Note: on "
                   "cosmos-style sigma=t/(1-t) schedules the noise end is compressed, so "
                   "the first step is very large - you may need more steps than 'simple' "
                   "to match its quality.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model the run will use. Its own sigma<->time map defines the flow-time grid."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise=1.0):
        import torch as _t
        steps = int(steps[0] if isinstance(steps, (list, tuple)) else steps)
        denoise = float(denoise[0] if isinstance(denoise, (list, tuple)) else denoise)
        if denoise <= 0.0:
            return (_t.FloatTensor([]),)
        total_steps = max(int(steps / denoise), steps)
        ms = model.get_model_object("model_sampling")
        sigmas = paper_aeuler_mod._uniform_flow_time_sigmas(ms, total_steps)
        sigmas = sigmas[-(steps + 1):]
        return (sigmas,)


# ---------------------------------------------------------------------------
# options nodes
# ---------------------------------------------------------------------------


class ResearchAFlopsAutotuner:
    DESCRIPTION = ("A-FloPS Autotuner: probes + schedule fine-tune in one node. "
                   "Connect the guider the run will use and ANY scheduler's sigmas "
                   "(e.g. BasicScheduler \"simple\"); the node measures the model and "
                   "this prompt's trajectory at the schedule's noise levels (tiny "
                   "probe latents), then re-allocates the steps of YOUR sigma list "
                   "where the measured curvature needs them (error-equidistributed; "
                   "flat curvature reproduces your list exactly). Outputs the "
                   "fine-tuned sigmas for the sampler and the probe options for the "
                   "A-FloPS Sampler node (connect its options output to the Sampler's "
                   "options+ input). Two probes, two sizes: the MODEL probe is the "
                   "bigger schedule-driving one and is cached per model+params+"
                   "schedule (across prompts); the COND probe is the small per-prompt "
                   "one, so its cost is what you feel each generation. Elementary "
                   "knobs only: those two probe sizes, plus bias and warp, both "
                   "1.0 = neutral (the probe's own error-optimal placement of your "
                   "schedule). Both knobs use the factor 4**(w-1), so +0.5 and -0.5 "
                   "are exact reciprocals of each other: equal power in either "
                   "direction.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER", {"tooltip": "Connect the same guider your run uses, so the Autotuner measures the model and this prompt with the settings the run will actually use. It does its work here, before the sampler starts."}),
                "sigmas": ("SIGMAS", {"tooltip": "Connect any scheduler's output (BasicScheduler \"simple\", Karras, and so on). This is the schedule the Autotuner fine-tunes. When it has no usable measurements the list is passed through unchanged, so your scheduler is always the fallback."}),
                "probe_steps": ("INT", {"default": 10, "min": 5, "max": 16, "step": 1,
                    "tooltip": "How many steps the model probe uses, 5 at the minimum. This probe measures the model and decides the fine-tune of the schedule. Its result is kept and reused for every prompt with the same model and settings, so it costs you on the first run only."}),
                "probe_resolution": ("INT", {"default": 48, "min": 16, "max": 128, "step": 8,
                    "tooltip": "The picture size the model probe runs at. 48 is the established value. Larger sizes are slower and measure more finely."}),
                "cond_probe_steps": ("INT", {"default": 4, "min": 3, "max": 8, "step": 1,
                    "tooltip": "How many steps the prompt probe uses, 3 at the minimum. This probe runs again for every new prompt, so this is the setting you feel on each generation."}),
                "cond_probe_resolution": ("INT", {"default": 16, "min": 8, "max": 32, "step": 8,
                    "tooltip": "The picture size the prompt probe runs at. 16 is the default; 8 is the fastest and 32 the finest."}),
                "bias": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Where the fine-tune puts most of the steps. 1.00 keeps the placement the measurements chose. Above 1.00 puts more steps in the early, noise-heavy part of the run, which builds up structure. Below 1.00 puts more steps in the late part, which builds up detail. The sigma values themselves are never changed."}),
                "warp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Moves the sigma values themselves, leaving the first and last ones where they are. 1.00 keeps your schedule exactly as it is. Below 1.00 reaches low noise sooner and spends more of the run there. Above 1.00 does the opposite. Does nothing while auto_warp is on, and large values take the run outside the range of noise levels the probe measured."}),
                "auto_warp": ("BOOLEAN", {"default": False,
                    "tooltip": "Chooses the warp for you. ON: the warp setting is ignored and the Autotuner uses the strongest value that keeps every step inside the range of noise levels it measured. OFF: your warp value is used exactly as you set it."}),
                # EXPERIMENTAL SWITCHES, HIDDEN IN THE RELEASE BUILD.  `pct_derived`,
                # `wmax_bound` and `wmax_budget` are no longer WIDGETS: they are fixed at
                # their engine defaults (pct_derived = True, wmax_bound = False,
                # wmax_budget = 1.0), which is the shipped behaviour.  They are still
                # accepted as parameters below and still listed in `opts`, so a saved graph
                # or an Options node can drive them and the research rigs can keep testing
                # them -- they are simply not offered to users as knobs.  Reason: each was
                # measured and none of them moves the image on the models we tested
                # (`pct_derived` is a bug fix and is ON, the other two were measured inert),
                # so exposing them would be untested clutter.
            },
        }

    RETURN_TYPES = ("SIGMAS", OPT_TYPE)
    RETURN_NAMES = ("sigmas", "options")
    FUNCTION = "autotune"

    def autotune(self, guider, sigmas, probe_steps, probe_resolution,
                 cond_probe_steps, cond_probe_resolution,
                 bias, warp, auto_warp=False,
                 pct_derived=True, wmax_bound=False, wmax_budget=1.0):
        import torch as _t
        import logging as _lg
        if isinstance(guider, (list, tuple)):
            guider = guider[0] if guider else None
        ext = _sig_tensor_list(sigmas)

        def _iv(v, default):
            if v is None:
                return default
            return int(v[0] if isinstance(v, (list, tuple)) else v)

        # Two probes, two sizes (restored): the MODEL probe is the bigger,
        # schedule-driving one and is cached across prompts; the COND probe is
        # the small per-prompt one, so its cost is what the user feels on each
        # generation.
        steps = max(5, min(16, _iv(probe_steps, 10)))
        res = max(16, min(128, _iv(probe_resolution, 48)))
        csteps = max(3, min(8, _iv(cond_probe_steps, 4)))
        cres = max(8, min(32, _iv(cond_probe_resolution, 16)))
        # The lambda clamp is always probe-driven now.  The A/B mode widget was
        # removed once 'auto' proved to be the right setting, so nothing writes a
        # fixed aflops_lam_min/max any more; the engine's own automatic path
        # picks the per-step trust region from the probe's measured x0 jump.
        bi = _f(bias)
        wa = _f(warp)
        # Keep the WIDGET value separate: auto_warp overwrites `wa` with the
        # adopted value, and without both we cannot tell "auto picked 1.0" from
        # "the user asked for 1.0".  The report records neither today, which
        # makes any warp/bias A/B unverifiable from the logs.
        wa_widget = float(wa)

        def _sched_stamp(info):
            info = dict(info or {})
            info["widgets"] = {"bias": round(float(bi), 4),
                               "warp": round(wa_widget, 4),
                               "auto_warp": bool(auto_warp),
                               "warp_used": round(float(wa), 4)}
            return info

        opts = {
            "probe_enabled": True,
            "probe_steps": steps,
            "probe_resolution": res,
            "cond_probe_enabled": True,
            "cond_probe_steps": csteps,
            "cond_probe_resolution": cres,
            "cond_probe_weight": 0.5,
            # THE DERIVED-CONSTANT ARMS MUST BE LISTED HERE OR THEY DO NOTHING.
            # This dict is hand-written, so a key that is not named here is absent from
            # the node-side probes' cfg -- which is precisely how the Sampler's switch came
            # to read ON in the log while every node/cond probe ran with it OFF.
            "pct_derived": bool(pct_derived),
            "wmax_bound": bool(wmax_bound),
            # THE NUDGE MUST BE LISTED HERE TOO, for the same reason as the two arms above:
            # a key absent from this dict is absent from the engine cfg, and then the gate
            # silently reads its neutral default 1.0 whatever the widget says.
            "wmax_budget": float(wmax_budget),
        }
        # Record the model's ACTUAL sampling + the user's input schedule so
        # the report is self-describing (no more guessing Flux/Cosmos/shift).
        try:
            _p = guider
            if hasattr(guider, "model_patcher"):
                _p = guider.model_patcher
            elif hasattr(guider, "inner_model"):
                _in = guider.inner_model
                if hasattr(_in, "model_patcher"):
                    _p = _in.model_patcher
            _ms = _p.get_model_object("model_sampling")

            def _fval(x):
                try:
                    import torch as _t
                    if _t.is_tensor(x):
                        return float(x.detach().item())
                    return float(x)
                except Exception:
                    return None

            _sigs = getattr(_ms, "sigmas", None)
            _ms_info = {
                "class": type(_ms).__name__,
                "shift": _fval(getattr(_ms, "shift", None)),
                "sigma_min": (_fval(_ms.sigma_min)
                              if hasattr(_ms, "sigma_min") else None),
                "sigma_max": (_fval(_ms.sigma_max)
                              if hasattr(_ms, "sigma_max") else None),
                "multiplier": _fval(getattr(_ms, "multiplier", None)),
                "n_sigmas": (len(_sigs) if _sigs is not None else None),
            }
        except Exception as _e:
            _ms_info = {"error": str(_e)}
        _ms_info["sigmas_in"] = ([round(float(v), 5) for v in ext]
                                 if ext is not None else None)
        opts["_ms_info"] = _ms_info
        if guider is None:
            # Nothing to measure: pass the schedule through untouched.
            _lg.warning("[A-FloPS-autotuner] no guider connected; sigmas "
                        "passed through unmodified")
            opts["_shift_schedule_node"] = _sched_stamp(
                {"mode": "off", "adopted": False, "base": "external"})
            return (sigmas, opts)
        if ext is None:
            _lg.warning("[A-FloPS-autotuner] unusable sigmas input; passed "
                        "through unmodified")
            opts["_shift_schedule_node"] = _sched_stamp(
                {"mode": "off", "adopted": False, "base": "external"})
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
        wa_info = None
        if _p is not None and bool(auto_warp):
            # auto-warp: the probe picks the most aggressive safe warp; the
            # manual widget is disabled while this is on.
            wa, wa_info = aflops_mod._warp_frontier(
                _p, ext, bias=float(bi))
            _lg.info("[A-FloPS-autotuner] auto-warp: adopted %.2f "
                     "(safe range %.2f-1.00, %s; worst-step ratio %.3f, "
                     "steps below measured floor %s)"
                     % (wa, wa_info.get("safe_range", [wa, 1.0])[0],
                        wa_info.get("why", "?"),
                        wa_info.get("worst_ratio", 1.0),
                        wa_info.get("below", "?")))
        # THE `schedule_source` A/B WAS CLOSED OUT (see NOTES.md).  Measured
        # outcome: the re-allocation is INERT on both models -- Krea2 produced a
        # BYTE-IDENTICAL grid (the engine arm found no usable curvature evidence
        # and passed through), and Anima's adopted grid differed from its input
        # by ~8e-8 relative, i.e. float32 rounding level.  The cross-arm image
        # difference on Anima was SMALLER than that model's own same-seed
        # repeat difference (0.0528 vs 0.0958), so it was never attributable to
        # this widget.  The switch is gone; the re-allocation path below is left
        # in place, inert, so behaviour is unchanged.
        if _p is not None:
            grid = aflops_mod._build_curvature_schedule(
                _p, len(ext) - 1, 1.0, bias=float(bi), warp=float(wa),
                base_sigmas=ext)
        if grid is not None and grid.numel() == len(ext):
            sched_info = {"mode": "curvature", "adopted": True,
                          "base": "external",
                          "bias": round(float(bi), 4),
                          "warp": round(float(wa), 4)}
            if wa_info is not None:
                sched_info["auto_warp"] = {
                    "safe_range": wa_info.get("safe_range"),
                    "why": wa_info.get("why"),
                    "worst_ratio": wa_info.get("worst_ratio"),
                    "below": wa_info.get("below")}
            _lg.info("[A-FloPS-autotuner] schedule fine-tuned from probe "
                     "curvature (bias=%.2f warp=%.2f%s, %d steps)",
                     float(bi), float(wa),
                     " auto" if wa_info is not None else "", len(ext) - 1)
            out_grid = grid
        else:
            # "adopted" above does NOT mean the schedule CHANGED -- a grid
            # numerically identical to the input lands here too, which is how
            # the inert re-allocation stayed invisible for so long.  A future
            # A/B should stamp the max relative change so "adopted" and
            # "actually moved" can be told apart.
            sched_info = {"mode": "off", "adopted": False, "base": "external"}
            _lg.info("[A-FloPS-autotuner] no usable curvature evidence; "
                     "schedule passed through")
            out_grid = _t.tensor(ext, dtype=_t.float32)
        opts["_shift_schedule_node"] = _sched_stamp(sched_info)
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
        # Include only the CURRENT run's own profiles, not the whole cache:
        # every saved/cached profile for every model+prompt+schedule was being
        # dumped into each log, so the JSON grew with the session instead of
        # describing this run (context rot).
        _cfg = aflops_mod._LAST_CFG or {}
        _mk = None
        for _k in ("_probe_key", "probe_key"):
            if _cfg.get(_k):
                _mk = str(_cfg[_k]).split("|s:")[0]
                break
        _psig = None
        for _k in ("_cond_probe_key", "cond_probe_key"):
            if _cfg.get(_k):
                _parts = str(_cfg[_k]).split("|")
                if len(_parts) >= 2:
                    # THE LABEL MUST BE STRIPPED OR THIS FILTER MATCHES NOTHING.
                    # `_cond_key_str` spells the middle field as " prompt <sig>" (the store's canonical
                    # spelling, kept because 443 logs use it), so splitting on "|" alone leaves
                    # "` prompt e80822dbd9e05270`" in `_psig`, and `_psig in str(key[1])` -- which is
                    # compared against the BARE signature -- can never be true.  Measured on the
                    # operator's A/B: `_cond_dump_diag` = {store: 4, same_model: 2, strict_ok: 0,
                    # dumped: 0}, i.e. EVERY cond profile was dropped by this node's own filter, which is
                    # why an empty `cond_probe_profiles` was indistinguishable from "none exists".
                    _psig = _parts[1].strip().split()[-1] or None
                break

        def _own_model(key):
            if _mk is None:
                return True
            return key == _mk or key.startswith(_mk + "|s:")

        def _own_cond(key):
            if _mk is None:
                return True
            if key[0] != _mk:
                return False
            if _psig is None:
                return True
            return _psig in str(key[1])

        probe_summary = {}
        try:
            for key, prof in aflops_mod._PROBE_PROFILES.items():
                if not _own_model(str(key)):
                    continue
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
                    # THE PROBE-SIDE RESCUE COVERAGE CURVE.  This report is a
                    # WHITELIST, not a dump: any profile key not named here never
                    # reaches the log, whatever the engine computes.  That is how
                    # `probe_diag` and `med_local_err_real` have been produced by
                    # the profile builder for a long time yet are absent from
                    # every log, and it is why the first cov_curve build appeared
                    # to have no effect.  Since the rescue gate's coverage CLIFF
                    # is the pitfall these runs exist to derive a rule for, the
                    # curve and its provenance must be here or the campaign
                    # produces data that cannot answer the question.
                    "cov_curve": prof.get("cov_curve"),

                    "cov_curve_terms": prof.get("cov_curve_terms"),
                    "cov_curve_meta": prof.get("cov_curve_meta"),
                    # THE DERIVED VARIANT + its measured rate.  The synthetic
                    # fields could not decide between the shipped per-step decay
                    # and ref_drop = k*du (scratch_ref_decay_rig.py refuted both
                    # of its fields), so both curves must reach the log and the
                    # real trajectories decide.
                    "cov_curve_kdu": prof.get("cov_curve_kdu"),
                    "cov_curve_k": prof.get("cov_curve_k"),
                    "probe_call": prof.get("probe_call"),
                    # WHICH PATCHER THE PROBE RAN ON.  The clone-path freeze is the
                    # open defect (NOTES.md, "THE FREEZE IS AN INTERACTION") and the
                    # clone was removed to test it -- but that change could NOT be
                    # confirmed from the corpus, because the WHITELIST documented
                    # above is precisely the thing that silently drops a new profile
                    # key.  Stamped by _engine_cond_probe_run; ABSENT on every build
                    # that used a clone, so its presence in a log is the proof that
                    # the running code carries the removal.
                    "probe_patcher": prof.get("probe_patcher"),
                    "probe_mopt_swapped": prof.get("probe_mopt_swapped"),
                    # THE RANK LADDER'S DECISION.  The node-side probes fall
                    # back to rank 4, and a 4-D latent through a 5-D model is a
                    # SILENT pass-through (rank 4 -> 9 frozen / 0 healthy on the
                    # corpus, rank 5 -> 0 frozen / 16 healthy).  These record
                    # which rank was used, what was tried, and the ladder order.
                    "probe_rank": prof.get("probe_rank"),
                    "probe_rank_tried": prof.get("probe_rank_tried"),
                    "probe_rank_ladder": prof.get("probe_rank_ladder"),
                    # present in the profile builder, absent from every log so far:
                    "probe_diag": prof.get("probe_diag"),
                    "med_local_err_real": prof.get("med_local_err_real"),
                    # THE INPUTS THE DERIVED THRESHOLDS ARE COMPUTED FROM.  A log that
                    # carries only the OUTPUT (calib) cannot say WHY a value came out
                    # clamped: reading calib alone, guard_floor 0.075 looks like a floor
                    # when it is really 0.15*sqrt(r/4) at r = 1, and wmax_base 2.5 looks
                    # like a ceiling when it is the map saturating below med_curv 1.4775.
                    # `jump_q` and `err_q` are named by _derive_anomaly_calibrations as the
                    # evidence for an_spatial / an_novelty / an_value, and n_jump feeds
                    # conv_step; without them those thresholds are unauditable from a log.
                    # Found by scratch_log_coverage_audit.py, which diffs the profile keys
                    # the engine assigns against this list.
                    "jump_q": prof.get("jump_q"),
                    "err_q": prof.get("err_q"),
                    "n_jump": prof.get("n_jump"),
                    "n_err": prof.get("n_err"),
                    "n_guard": prof.get("n_guard"),
                    "conv_step": prof.get("conv_step"),
                    "endgame": prof.get("endgame"),
                    "fragile_ranges": prof.get("fragile_ranges"),
                    "dir_cons": prof.get("dir_cons"),
                    "dir_cons_valid": prof.get("dir_cons_valid"),
                    "ojf_curve": prof.get("ojf_curve"),
                    "band_curve": prof.get("band_curve"),
                    # THE PER-ORDER DERIVATIVE SCALE the `wmax_bound` arm consumes, so the
                    # log says which orders this grid could certify and which it could not.
                    "dk_scale": prof.get("dk_scale"),
                    # WHICH A/B FLAGS THIS PROFILE WAS BUILT UNDER.  Two different nodes can
                    # build probes (the Autotuner's opts dict and the Sampler's cfg), so
                    # without this a disagreement is invisible and the log's cfg block
                    # describes only the Sampler's settings -- which is exactly how a
                    # switch read ON while the probes ran OFF.
                    "calib_flags": prof.get("calib_flags"),
                    # THE TWO ESTIMATORS, SIDE BY SIDE.  So one run shows the shipped-vs-
                    # derived delta without needing a toggle flipped, and `n_guard` says
                    # whether the p90/med collision is in play for this profile.
                    "guard_sig_med_shipped": prof.get("guard_sig_med_shipped"),
                    "guard_sig_p90_shipped": prof.get("guard_sig_p90_shipped"),
                    "guard_sig_med_derived": prof.get("guard_sig_med_derived"),
                    "guard_sig_p90_derived": prof.get("guard_sig_p90_derived"),
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
            # ONE FORMATTER, SHARED WITH THE ENGINE -- called directly, with NO inline
            # fallback: a fallback that re-spells the key is how the two spellings diverged
            # in the first place (the log's store keys and the stamped `cfg["_cond_probe_key"]`
            # matched in 0 of 443 logs).  If the helper is ever missing, that is an error worth
            # seeing, not one worth papering over.
            for key, prof in _cond_store.items():
                if not _own_cond(key):
                    continue
                cond_probe_summary[aflops_mod._cond_key_str(key)] = {
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
                    # probe-side rescue coverage curve -- see the model-profile
                    # whitelist above for why this must be named explicitly.
                    "cov_curve": prof.get("cov_curve"),

                    "cov_curve_terms": prof.get("cov_curve_terms"),
                    "cov_curve_meta": prof.get("cov_curve_meta"),
                    # THE DERIVED VARIANT + its measured rate.  The synthetic
                    # fields could not decide between the shipped per-step decay
                    # and ref_drop = k*du (scratch_ref_decay_rig.py refuted both
                    # of its fields), so both curves must reach the log and the
                    # real trajectories decide.
                    "cov_curve_kdu": prof.get("cov_curve_kdu"),
                    "cov_curve_k": prof.get("cov_curve_k"),
                    "probe_call": prof.get("probe_call"),
                    # which patcher the cond probe ran on (see the model-probe
                    # whitelist for why this must be named explicitly)
                    "probe_patcher": prof.get("probe_patcher"),
                    "probe_mopt_swapped": prof.get("probe_mopt_swapped"),
                    # the rank ladder's decision (see the model-probe whitelist)
                    "probe_rank": prof.get("probe_rank"),
                    "probe_rank_tried": prof.get("probe_rank_tried"),
                    "probe_rank_ladder": prof.get("probe_rank_ladder"),
                    "med_local_err_real": prof.get("med_local_err_real"),
                    # THE INPUTS THE COND THRESHOLDS ARE COMPUTED FROM -- see the
                    # model-probe whitelist for why: a log carrying only `calib` cannot say
                    # why a value landed on a clamp.  Adding a key the cond profile does
                    # not carry is harmless (prof.get returns None).
                    "jump_q": prof.get("jump_q"),
                    "err_q": prof.get("err_q"),
                    "n_jump": prof.get("n_jump"),
                    "n_err": prof.get("n_err"),
                    "n_guard": prof.get("n_guard"),
                    "conv_step": prof.get("conv_step"),
                    "endgame": prof.get("endgame"),
                    "fragile_ranges": prof.get("fragile_ranges"),
                    "dir_cons": prof.get("dir_cons"),
                    "dir_cons_valid": prof.get("dir_cons_valid"),
                    "ojf_curve": prof.get("ojf_curve"),
                    "band_curve": prof.get("band_curve"),
                    # THE PER-ORDER DERIVATIVE SCALE the `wmax_bound` arm consumes, so the
                    # log says which orders this grid could certify and which it could not.
                    "dk_scale": prof.get("dk_scale"),
                    # WHICH A/B FLAGS THIS PROFILE WAS BUILT UNDER.  Two different nodes can
                    # build probes (the Autotuner's opts dict and the Sampler's cfg), so
                    # without this a disagreement is invisible and the log's cfg block
                    # describes only the Sampler's settings -- which is exactly how a
                    # switch read ON while the probes ran OFF.
                    "calib_flags": prof.get("calib_flags"),
                    # THE TWO ESTIMATORS, SIDE BY SIDE.  So one run shows the shipped-vs-
                    # derived delta without needing a toggle flipped, and `n_guard` says
                    # whether the p90/med collision is in play for this profile.
                    "guard_sig_med_shipped": prof.get("guard_sig_med_shipped"),
                    "guard_sig_p90_shipped": prof.get("guard_sig_p90_shipped"),
                    "guard_sig_med_derived": prof.get("guard_sig_med_derived"),
                    "guard_sig_p90_derived": prof.get("guard_sig_p90_derived"),
                }
        except Exception:
            pass
        # WHY IS THE COND DUMP EMPTY? (2026-10-04)  On the operator's first `cond_rtol_loosen` A/B the
        # dump was EMPTY in all 12 runs (`cond_probe_profiles: 0`) even though the cond probe had RUN
        # (`node_cond_probe_state: ran`) -- and an empty dump is indistinguishable from "no cond profile
        # exists", which is what left that A/B's null unexplained.  This records WHICH filter rejected
        # them, in the log, instead of leaving it to be inferred: how many cond profiles exist, how many
        # carry this run's model key, how many also pass the prompt-signature test, and how many were
        # actually dumped.  `_LAST_CFG` is what the log serialises, so a diagnostic put here lands in the
        # corpus.  It changes no behaviour.
        if _cfg is not None:
            try:
                _cond_all = getattr(aflops_mod, "_COND_PROBE_PROFILES", {}) or {}
                _same_model = sum(1 for _k in _cond_all
                                  if _mk is None or (isinstance(_k, (tuple, list)) and _k
                                                     and _k[0] == _mk))
                _strict = sum(1 for _k in _cond_all
                              if isinstance(_k, (tuple, list)) and len(_k) >= 2
                              and (_mk is None or _k[0] == _mk)
                              and (_psig is None or _psig in str(_k[1])))
                _cfg["_cond_dump_diag"] = {"store": len(_cond_all), "same_model": _same_model,
                                           "strict_ok": _strict, "dumped": len(cond_probe_summary),
                                           "mk": _mk, "psig": (str(_psig)[:28] if _psig else None)}
            except Exception:
                pass
        return (json.dumps({"aflops_cfg": aflops_mod._LAST_CFG,
                            "probe_profiles": probe_summary,
                            "cond_probe_profiles": cond_probe_summary,
                            "aflops": aflops_mod._LAST_ERRORS}, indent=2),)


NODE_CLASS_MAPPINGS = {
    "ResearchAFlopsSampler": ResearchAFlopsSampler,
    "ResearchAFlopsAutotuner": ResearchAFlopsAutotuner,
    "ResearchAFlopsDebugOptions": ResearchAFlopsDebugOptions,
    "ResearchAFlopsErrorReport": ResearchAFlopsErrorReport,
    "ResearchPaperAEuler": ResearchPaperAEuler,
    "ResearchPaperAEulerSchedule": ResearchPaperAEulerSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResearchAFlopsSampler": "A-FloPS Sampler",
    "ResearchAFlopsAutotuner": "A-FloPS Autotuner (probe + schedule)",
    "ResearchAFlopsDebugOptions": "A-FloPS Debug Options (finished switches)",
    "ResearchAFlopsErrorReport": "A-FloPS Error Report",
    "ResearchPaperAEuler": "A-FloPS A-Euler (paper)",
    "ResearchPaperAEulerSchedule": "A-FloPS A-Euler Schedule (paper)",
}

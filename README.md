# PD-AOS Sampler for ComfyUI

An adaptive sampler. Before it samples anything, it measures the model and your
prompt with two small probe passes, and then uses those measurements to:

1. **place the steps of whatever schedule you plug in** where the measurements say
   they are needed, and
2. **decide, for each part of the picture separately, how far ahead it looks** when
   it works out where the image is heading.

> **Tested with Krea 2 and Anima models.** Other flow models are expected to work,
> but have not been verified.

## Provenance — what this is, and what it is not

**This project began as an implementation of A-FloPS** — Jin, Xiao & Gu, *A-FloPS: Accelerating Diffusion Models via
Adaptive Flow Path Sampler*, AAAI 2026 ([arXiv:2509.00036](https://arxiv.org/abs/2509.00036)) — **and became
something only very loosely inspired by it.** The name was changed to PD-AOS because the old one over-claimed, so
here is the split, stated plainly:

- **From the paper:** the idea of working in a reparameterised flow time, and — still included, as two reference
  nodes — the paper's own **Algorithm 2 ("A-Euler")**, a faithful reproduction of the authors' reference
  implementation, kept so the original method can be compared against this sampler directly.
- **This sampler, which is a different thing:** before sampling it measures *your* model and prompt with two small
  probe passes, then uses those measurements to place the steps of the schedule you plug in and to decide, per region
  of the image, how many previous steps to look at. The probes, the per-pixel order control, the Autotuner and the
  local-field machinery are this project's own and have no counterpart in the paper.
- **They are not the same algorithm.** On an exact analytic ODE test the paper's step parameterisation sits closer to
  the true solution than this sampler's own step does. That is a statement about *discretisation accuracy*, not about
  how the images look. If you want A-FloPS itself, use the paper's nodes below or the authors' implementation; if you
  want this, use PD-AOS.
- **The attribution stays**, and so does the comparison: the reference nodes are there so you can check the above for
  yourself rather than take it on trust.

## What it does

- **Measures first, samples second.** The probes run on tiny latents before the
  real sampler starts. Nothing is guessed and nothing is fitted to a formula: the
  schedule and the per-pixel decisions come from what was measured on your model
  and your prompt.
- **Keeps your scheduler.** The Autotuner fine-tunes the sigma list you give it. If
  it has no usable measurements, it passes your list through untouched.
- **Works per pixel.** Some areas of a picture follow a clear direction and can be
  pushed harder; others are busy or detailed and cannot. The sampler decides this
  separately for each area.
- **Refuses what is not safe.** A higher setting is rejected when it would make the
  step come out too large, and a corrected step is dropped when it departs too far
  from the plain one. Raising a setting can therefore never force a bad step through.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

## Nodes

### PD-AOS Autotuner (probe + schedule)

- Inputs: `guider` (the guider your run uses), `sigmas` (any scheduler's output).
- Outputs: `sigmas` (fine-tuned) and `options` (connect this to the Sampler's
  `options+` input).

| setting | what it does |
|---|---|
| `probe_steps` | How many steps the model probe uses. This probe decides the schedule, and its result is kept for every prompt with the same model and settings. |
| `probe_resolution` | The picture size the model probe runs at. 48 is the established value. |
| `cond_probe_steps` | How many steps the prompt probe uses. This probe runs again for every new prompt, so this is the setting you feel on each generation. |
| `cond_probe_resolution` | The picture size the prompt probe runs at. 16 is the default. |
| `bias` | Where the fine-tune puts most of the steps. 1.00 keeps the placement the measurements chose. Above 1.00 puts more steps early, in the noise-heavy part, which builds structure. Below 1.00 puts more steps late, which builds detail. |
| `warp` | Moves the sigma values themselves, leaving the first and last where they are. 1.00 keeps your schedule as it is. Below 1.00 reaches low noise sooner. |
| `auto_warp` | ON: the `warp` setting is ignored and the strongest safe value is chosen for you. OFF: your `warp` value is used as set. |

### PD-AOS Sampler

| setting | what it does |
|---|---|
| `log_errors` | Writes what the sampler decides at every step to the console, and fills the Error Report node. For testing. |

**That is the only setting on this node.** The switches that used to sit here — `lf_ab`, `order_by_bound`,
`order_floor`, `an_enable` and `wmax_eg_dir` — have each reached a decision, so they are no longer exposed
anywhere: **the engine defaults ARE the decided values**, and the evidence behind each one is in
`CONSTANTS_AUDIT.md` and `DEAD_FEATURES.md`.

### PD-AOS Debug Options (the automatic order demand)

**This node carries one system: the automatic order demand** — how hard the sampler asks itself to follow the
trajectory. The switches it used to carry are settled and gone from the UI; their values and evidence are the
"Settled settings" table below, so nothing is lost by removing them.

This node also has an **`options+` input of its own**, so it can be **chained** with another options
node — for example the Autotuner's `options` output into this node, and this node's `options` output
into the Sampler. Fields from the upstream node pass through untouched; where two nodes set the same
field, the one **closest to the Sampler** wins.

**Disconnected, the Sampler uses the engine defaults, and those ARE the decided values** — the
defaults on this node are read from the engine rather than repeated, so the two cannot drift apart.

| setting | shipped | what it does |
|---|---|---|
| `order_floor` | **0** (AUTOMATIC) | `0` = ask for the high order only while the run's **own measured error** is high, then let go. It is the shipped default because it **won 61 % of head-to-heads** against a fixed demand across 26 judged prompts and was never the worst pick. `1` = the sampler always decides by itself. `2`–`6` = ask for at least that many previous steps per pixel — a **request the safety path may refuse**. **Measured: the order actually used caps at 5** (8,362 step records, every arm and every build), so a demand of `6` cannot raise the maximum and the mean rises only a little above `5`. |
| `order_floor_adaptive_rtol` | **0.05** | The **flip point**: the automatic demand acts while the step's error over its tolerance is above this, and asks for nothing below it. Grounded on 768 step records — that ratio's p75 is 0.0505, so the shipped value demands in roughly the worst quarter of the steps. |
| `order_floor_adaptive_graded` | **ON** | ON: the order asked for **scales with the error** (about 4 at the bottom of the gate's range, 5 near the top). OFF: the previous behaviour exactly — a flat demand of 6 whenever the gate is open — which is what makes the two comparable in one test. |
| `order_floor_adaptive_lo` | **4** | The order the automatic mode asks for at the **bottom** of the gate's error range. **Derived, not chosen:** above the gate's trigger the order the engine actually uses is p25 4.17 / p50 4.53 / p75 4.88. Setting `lo` equal to `hi` gives a flat demand at that level. |
| `order_floor_adaptive_hi` | **5** | The order it asks for at the **top** of the error range. **`hi = 5` is the realised ceiling of the whole engine** — measured over 8,362 step records, across every arm and every build, the order actually used never exceeds 5 — so raising `hi` above 5 cannot raise the maximum, and the mean order only rises a little (going 4 → 5 buys +0.79 of mean order, 5 → 6 buys +0.21). |
| `order_floor_adaptive_r_hi` | **0.12** | How fast the ramp reaches `hi` — the error ratio at which the demand stops rising. The previously shipped `0.3031` reached full strength on only **~10 %** of its demanding steps against **~78 %** for `0.12`, and the `0.12` cell is the one the maintainer accepted by eye. |
| `order_demand_q` | **0** (off) | A per-pixel bound-derived request. **Measured not to do what it claims:** on the harness it produced a binary request rather than a graded one, asking for 6 on steps whose own ceiling was 3–5. `0` is the only verified setting. |

### Settled settings (no widget; frozen in the engine defaults)

Each was decided by measurement plus the maintainer's eye, and each now lives only as an engine default. They are
listed so the public record survives the removal of their widgets.

| setting | value | why it is settled |
|---|---|---|
| `lf_ab` (`lf_enable`) | ON | preferred by eye on the LF × ORDER 2×2 |
| `order_by_bound` | ON | release ruling; picks each pixel's order by the highest count that is still safe |
| `an_enable` | ON | his own A/B: where the anomaly detector fires, it fixes what ordinary sampling does not recover from in time |
| `wmax_eg_dir` | 0 | the error-direction reaction; the negative side measured worse against an exact solution |
| `cond_rtol_loosen` | 1.5 | **measured inert** across 31 runs on two models — the consumed tolerance never moved |
| `ord_ema` | 0.37 | the **measured optimum**: alpha* = 1 − rho_1 = 0.370 on its own logged innovation series |

### PD-AOS Error Report

Optional. Writes the sampler's per-step decisions to a JSON file.

### A-FloPS A-Euler (paper) / A-FloPS A-Euler Schedule (paper)

The reference implementation of Algorithm 2 from Jin et al., AAAI 2026
(arXiv:2509.00036), included so the adaptive sampler can be compared against it.
These are plain samplers: they do not use the probes and they make one model call
per step. They are not needed for normal use.

## Wiring

![Wiring example: BasicScheduler → PD-AOS Autotuner → the PD-AOS Sampler → SamplerCustomAdvanced](docs/wiring-example.png)

1. Build a normal `SamplerCustomAdvanced` graph (model, noise, guider, sampler,
   sigmas).
2. Add **PD-AOS Autotuner**: connect your guider to `guider`, and your scheduler's
   output (for example BasicScheduler "simple") to `sigmas`.
3. Connect the Autotuner's `sigmas` output to `SamplerCustomAdvanced.sigmas`.
4. Add **PD-AOS Sampler**: connect the Autotuner's `options` output to its
   `options+` input, and its `sampler` output to `SamplerCustomAdvanced.sampler`.
   The Sampler's own `model` input is optional: connecting the model there lets it
   read the model's type and settings directly. The Autotuner needs no such input,
   because it receives the model through the guider.
5. Run. The first generation probes the model and the prompt and stores the
   measurements; later generations with the same model and settings reuse them.

The screenshot above is that graph for a 16-step run, with `auto_warp` on and the Debug Options
node left disconnected — i.e. the engine defaults, with `order_floor` at 0 (AUTOMATIC), which is
what the engine defaults give.

## Notes

- **Higher is not automatically better.** Looking further ahead follows the
  trajectory's direction harder, and on some models and prompts that gives a
  cleaner, more finished image, while on others it does not. The sampler measures
  per pixel and refuses what is not safe; the settings above let you push further,
  and the result should be judged by eye.
- **The first run costs more than later ones**, because of the model probe. It is
  cached per model, settings and schedule, so it is paid once per model
  configuration rather than once per generation.
- **A few experimental switches exist in the code but are not shown in the node
  interface.** They are fixed at their shipped values. They are not described here
  because they are not proven.
- **The noise control was removed.** It was measured to have no effect on the
  output on the models we tested, so the setting is gone rather than left in place
  doing nothing.

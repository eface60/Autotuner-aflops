# A-FloPS Sampler for ComfyUI

An adaptive sampler. Before it samples anything, it measures the model and your
prompt with two small probe passes, and then uses those measurements to:

1. **place the steps of whatever schedule you plug in** where the measurements say
   they are needed, and
2. **decide, for each part of the picture separately, how far ahead it looks** when
   it works out where the image is heading.

> **Tested with Krea 2 and Anima models.** Other flow models are expected to work,
> but have not been verified.

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

### A-FloPS Autotuner (probe + schedule)

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

### A-FloPS Sampler

| setting | what it does |
|---|---|
| `log_errors` | Writes what the sampler decides at every step to the console, and fills the Error Report node. For testing. |

Inputs: `model` is optional - connecting it lets the sampler read the model's type
and settings directly. `options+` takes the Autotuner's `options` output, or an
**A-FloPS Debug Options** node (below), and the sampler obeys whatever it is told.

The sampler's own behaviour switches are grouped onto the Debug Options node so that
one node can drive every run of a graph. A workflow saved with the older settings on
the sampler itself keeps working: those inputs are still honoured.

### A-FloPS Debug Options (the switches)

Connect its `options` output to the Sampler's `options+`. It also accepts `options+`
itself, so several options nodes can be chained - the one closest to the sampler wins
wherever two of them set the same thing.

| setting | what it does |
|---|---|
| `lf_ab` | Turns the local field on or off. **ON is the shipped default.** ON: the sampler also makes small local corrections where its own step is least reliable. OFF: no local corrections. |
| `order_by_bound` | Chooses how the per-pixel order is picked. OFF: the number of previous steps that predicts the next one best. ON: the highest number of previous steps that is still safe for that pixel, so smooth areas look further ahead and busy areas do not. **ON is the shipped default.** It pairs with `order_floor`: the setting below is what the sampler is allowed to raise the order *to*. |
| `order_floor` | The lowest order the sampler may use, per pixel. 1 means the sampler decides by itself. Numbers above 1 push every pixel up to at least that value, which makes the sampler follow the direction of the trajectory harder. **6 is the shipped default**, paired with `order_by_bound` on. **Note what that combination costs:** measured on one prompt, one seed and one build, raising this from 1 to 6 nearly doubled the sampler's own final-step error (0.060-0.063 to 0.113-0.128, with no overlap between the two groups) and lowered the final image's gradient. That measurement scores distance to an exact solution, not how an image looks, and the shipped default follows the maintainer's eye across many runs rather than that number. Set it to 1 to let the sampler decide entirely on its own, or to `1` with `order_by_bound` off for the most conservative behaviour. |
| `an_enable` | Turns the anomaly detector on or off. **ON is the shipped default.** ON: at each step the sampler looks for a few pixels whose local change is far outside their own neighbourhood - an abrupt, spatially concentrated event - and damps the local field's correction on that small region for the next several steps. OFF: that detection never runs and nothing is damped. Decided ON by the maintainer's own A/B: there are prompts where it does not fire, but where it does, it fixes things regular sampling does not recover from in time. |
| `wmax_eg_dir` | Which way the sampler reacts when its own end-of-run error measurement comes out **high**. `0` (shipped default) = no reaction; the error-based limit acts alone. `+1` = it allows the most extrapolation then. `-1` = it does the **opposite** and allows the least extrapolation then. This setting decides **which pixels** get the higher extrapolation order, so it changes the **structure** of the image more than its overall detail or sharpness. Measured: `-1` is worse than `+1` against an exact ODE solution on three analytic flows (by up to 47%). **The live range is `-1` to `0`**: `+0.5` and `+1` behave the same as each other because the error-feedback loop that used to separate them has been removed, and the response saturates above an effective limit of about 3.8. |
| `ord_ema` | How fast the sampler forgets its own recent innovation estimate. It blends the previous estimate with the newest sample: `0.0` ignores the history and follows the newest sample, `1.0` never moves, and the **shipped `0.37` leans on the newest sample.** Higher is steadier and slower to react; lower is twitchier. **The range is not arbitrary** - it is a blend weight between two things. **Grounded, and the shipped value IS the measured optimum:** the sampler's own logged innovation series (n=682) has step-to-step correlation 0.630, which puts the best-fitting value at 0.370. It shipped at 0.5 - 0.13 stiffer than the measurement wants - until a controlled A/B settled it: one session, one prompt, one seed at 0.5 / 0.37 / 0.5 gave a **bit-identical** replicate pair and a 33 % change on the `0.37` arm, and `0.37` looked better. |
| `cond_rtol_loosen` | **Experimental, and honest about it.** Scales the tolerance derived from the **prompt** probe. `1.5` reproduces the shipped behaviour, `1.0` switches that loosening off, below `1.0` tightens. It is measured **inert in practice**: across 31 runs on two models the tolerance those runs actually consumed never moved. It is here because a knob that is not reachable cannot be tested. |

Both order settings still obey the safety rules described above, and a pixel the
sampler has just reset stays at order 1.

### A-FloPS Error Report

Optional. Writes the sampler's per-step decisions to a JSON file.

### A-FloPS A-Euler (paper) / A-FloPS A-Euler Schedule (paper)

The reference implementation of Algorithm 2 from Jin et al., AAAI 2026
(arXiv:2509.00036), included so the adaptive sampler can be compared against it.
These are plain samplers: they do not use the probes and they make one model call
per step. They are not needed for normal use.

## Wiring

![Wiring example: BasicScheduler -> A-FloPS Autotuner -> the A-FloPS Sampler -> SamplerCustomAdvanced](docs/wiring-example.png)

1. Build a normal `SamplerCustomAdvanced` graph (model, noise, guider, sampler,
   sigmas).
2. Add **A-FloPS Autotuner**: connect your guider to `guider`, and your scheduler's
   output (for example BasicScheduler "simple") to `sigmas`.
3. Connect the Autotuner's `sigmas` output to `SamplerCustomAdvanced.sigmas`.
4. Add **A-FloPS Sampler**: connect the Autotuner's `options` output to its
   `options+` input, and its `sampler` output to `SamplerCustomAdvanced.sampler`.
   The Sampler's own `model` input is optional: connecting the model there lets it
   read the model's type and settings directly. The Autotuner needs no such input,
   because it receives the model through the guider.
5. Optional: add **A-FloPS Debug Options** and chain it in front of the Sampler
   (Autotuner `options` -> Debug Options `options+`, Debug Options `options` ->
   Sampler `options+`) when you want the behaviour switches on a node of their own.
   With it unconnected, every switch sits at its shipped default.
6. Run. The first generation probes the model and the prompt and stores the
   measurements; later generations with the same model and settings reuse them.

The steps above are the wiring. The screenshot was taken with the shipped defaults
(`order_by_bound` on, `order_floor` at 6, `lf_ab` on, `auto_warp` on); in a graph
saved before this release, those switches sit on the Sampler node rather than on the
Debug Options node.

## What changed in this release

- **The five finished switches moved off the Sampler onto a new `A-FloPS Debug
  Options` node** (`lf_ab`, `order_by_bound`, `order_floor`, `an_enable`,
  `wmax_eg_dir`). Saved workflows keep working: the Sampler still honours its old
  inputs, and no saved value moves to a different setting. `an_enable` is new to the
  interface.
- **`ord_ema` ships at 0.37** instead of 0.5 - the measured optimum, and the value
  the maintainer's eye preferred in a controlled A/B. It lives on the Debug Options
  node.
- **The Sampler node now shows a single setting**, `log_errors`.
- Engine-side: the per-pixel order criterion no longer carries two thresholds its
  own measurements could never reach, and a set of dead code paths was removed. Both
  are behaviour-identical on the models tested.

## Notes

- **Higher is not automatically better.** Looking further ahead follows the
  trajectory's direction harder, and on some models and prompts that gives a
  cleaner, more finished image, while on others it does not. The sampler measures
  per pixel and refuses what is not safe; the settings above let you push further,
  and the result should be judged by eye.
- **The first run costs more than later ones**, because of the model probe. It is
  cached per model, settings and schedule, so it is paid once per model
  configuration rather than once per generation.
- **A couple of the Debug Options settings are experimental.** They say so in their
  own tooltips, together with what has been measured about them, rather than being
  hidden.
- **The noise control was removed.** It was measured to have no effect on the
  output on the models we tested, so the setting is gone rather than left in place
  doing nothing.

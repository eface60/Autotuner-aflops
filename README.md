# A-FloPS Autotuner for ComfyUI

An adaptive sampler for ComfyUI: it probes the model's trajectory and
**fine-tunes whatever sigma schedule you plug in** (e.g. BasicScheduler
"simple") so the steps land where the measured curvature needs them.

> **Tested only with Krea 2 and Anima (preview) models.** Other flow models
> are expected to work but have not been verified.

## Nodes

- **A-FloPS Autotuner** — probes + schedule fine-tune in one node.
  - Inputs: `guider` (the run's guider), `sigmas` (any scheduler's output).
  - Outputs: `sigmas` (fine-tuned), `options` (for the Sampler).
  - Knobs: `probe_steps` (min 5), `probe_resolution`, `shift` (1.0 = pure
    error-optimal placement; >1 = more high-sigma structure steps, <1 = more
    low-sigma detail steps).
- **A-FloPS Sampler** — the sampling engine. Knobs: `stochasticity`,
  `log_errors`. Connect the Autotuner's `options` to its `options+` input.
- **A-FloPS Error Report** — optional; dumps per-step decisions as JSON.

## Wiring

![Wiring example: BasicScheduler → A-FloPS Autotuner → SamplerCustomAdvanced, with the A-FloPS Sampler fed by the Autotuner's options](docs/wiring-example.png)

Step by step:

1. Build a normal `SamplerCustomAdvanced` graph (model → noise → guider → sampler → sigmas).
2. Add **A-FloPS Autotuner**: connect your guider to `guider`, connect your
   scheduler's output (e.g. BasicScheduler "simple") to `sigmas`.
3. Connect Autotuner `sigmas` → `SamplerCustomAdvanced.sigmas`.
4. Add **A-FloPS Sampler**: connect Autotuner `options` → Sampler `options+`,
   and Sampler `sampler` → `SamplerCustomAdvanced.sampler`.
5. The first run probes the model + prompt (tiny latents, cached per
   model/prompt/schedule); later runs reuse the measurements.

In the screenshot: a Text Encoding (guider) node feeds the Autotuner's
`guider` and the Sampler's `model`; BasicScheduler ("simple", 50 steps,
denoise 1.00) feeds the Autotuner's `sigmas`; the Autotuner's `sigmas` goes
to `SamplerCustomAdvanced.sigmas` and its `options` goes to the A-FloPS
Sampler's `options+`; the A-FloPS Sampler supplies
`SamplerCustomAdvanced.sampler`.

With no usable probe evidence the Autotuner passes your schedule through
untouched — your scheduler is always the baseline.

## Install

Copy this folder into `ComfyUI/custom_nodes/`.

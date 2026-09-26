# A-FloPS for ComfyUI

A-FloPS (Adaptive Flow Path Sampler): a ComfyUI custom sampler that probes
the model's trajectory and fine-tunes your sigma schedule to it.

## Nodes

- **A-FloPS Autotuner** — probes + schedule fine-tune in one node.
  Connect the guider and any scheduler's sigmas (e.g. BasicScheduler
  "simple"); it measures the model and the prompt's trajectory at the
  schedule's noise levels, then re-allocates the steps of your sigma list
  where the measured curvature needs them. Flat curvature reproduces your
  list exactly. Elementary knobs only: probe steps, probe resolution, and
  the shift bias (1.0 = pure error-optimal placement).
- **A-FloPS Sampler** — the sampling engine (stochasticity + log toggle).
- **A-FloPS Error Report** — per-step internal decisions as JSON.

## Wiring

```
BasicScheduler ("simple") ──sigmas──▶ Autotuner ──sigmas──▶ SamplerCustomAdvanced
guider ─────────────────────────────▶ Autotuner ──options─▶ A-FloPS Sampler (options+)
```

## Install

Copy this folder into `ComfyUI/custom_nodes/`.

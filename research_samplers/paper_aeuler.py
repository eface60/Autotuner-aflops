# -*- coding: utf-8 -*-
"""Faithful A-FloPS "A-Euler" for native flow models (Algorithm 2 of
Jin et al., AAAI 2026, arXiv:2509.00036), as a ComfyUI KSampler engine.

This reproduces the AUTHORS' reference implementation
(github.com/jinc7461/AFloPs, class AEuler), i.e. the native-flow sampler
that generated the paper's SD3.5 ablation:

    flow time    t = 1 - sigma            (LINEAR in sigma, noise -> clean)
    velocity     v = -d,  d = (x - denoised)/sigma
    first step   x + d*(sigma_next - sigma)        <- the sigma-Euler, exact
    later steps  adaptive exponential integrator:
        c      = <v - v_prev, x - x_prev> / (||x - x_prev||^2 + 1e-8)  (Eq.12)
        h_now  = v - c*x ;  h_prev = v_prev - c*x_prev
        dt     = t_next - t_now
        x'     = x*e^{c*dt} + A*h_now + B*(h_now - h_prev)/(t_now - t_prev)
        A      = (e^{c*dt} - 1)/c ,  B = (e^{c*dt} - 1 - c*dt)/c^2   (Eq.13/15)

Why t = 1 - sigma and not the model's internal time: ComfyUI's CONST flow
sampling interpolates x = (1-sigma)*x0 + sigma*noise, so sigma IS the
interpolation parameter and the flow time is linear in it.  Two consequences:

  * the first (flow-time Euler) step is algebraically IDENTICAL to ComfyUI's
    own sigma-Euler step, so there is no truncation error at the start;
  * the velocity v = -d stays bounded, and because dt is an actual sigma
    difference, ANY sigma schedule integrates correctly.

`c` is per-batch-element (reduced over the trailing dims), matching the
reference.  The reference does NOT clamp `c` for this sampler; pass
lam_min/lam_max to clamp it (testing / A-B only).

Note: the sigma<->flow-time map helpers at the top of this file are for the
companion scheduling node (uniform-flow-time grids for non-flow models); the
sampler itself needs no map.
"""
import math

import torch


_FLUX_CLASS = "ModelSamplingFlux"
_COSMOS_CLASS = "ModelSamplingCosmosRFlow"


# ---------------------------------------------------------------------------
# sigma <-> model-flow-time map (used by the companion scheduling node)
# ---------------------------------------------------------------------------
def _flow_time_of_sigma(ms, sigma):
    """Model flow time tau (clean=0 .. noise=tau_max) for a sigma."""
    cls = type(ms).__name__
    sigma = float(sigma)
    if cls == _FLUX_CLASS:
        e = math.exp(float(getattr(ms, "shift", 1.15)))
        return sigma / (e * (1.0 - sigma) + sigma)
    if cls == _COSMOS_CLASS:
        return sigma / (sigma + 1.0)
    try:  # generic: model's own inverse map
        return float(ms.timestep(torch.tensor(sigma)))
    except Exception:
        return sigma / (sigma + 1.0)


def _sigma_of_flow_time(ms, tau):
    cls = type(ms).__name__
    tau = float(tau)
    if cls == _COSMOS_CLASS:
        smax = float(ms.sigma_max)
        if tau >= smax / (smax + 1.0):
            return smax
        return tau / (1.0 - tau)
    try:  # Flux and generic: model's own forward map
        return float(ms.sigma(torch.tensor(tau)))
    except Exception:
        return tau / (1.0 - tau)


def _uniform_flow_time_sigmas(ms, steps):
    """The paper's OWN grid (Algorithm 2): t = n/N (noise -> clean), uniform
    flow time over [0, 1]; sigma = sigma(tau = 1 - t).  The scheduler node
    emits this; the sampler builds the same uniform flow-time grid internally
    from the step count + [sigma_0, sigma_last] endpoints."""
    out = []
    for n in range(steps + 1):
        t = n / steps
        out.append(_sigma_of_flow_time(ms, 1.0 - t))
    return torch.FloatTensor(out)


# ---------------------------------------------------------------------------
# exponential-integrator phi-functions (tensor form: c is per batch element)
# ---------------------------------------------------------------------------
def _phi1(z):
    """phi1(z) = (e^z - 1)/z, with a Taylor branch at z ~ 0."""
    small = z.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(z), z)
    return torch.where(small,
                       1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)),
                       torch.expm1(safe) / safe)


def _phi2(z):
    """phi2(z) = (e^z - 1 - z)/z^2, with a Taylor branch at z ~ 0."""
    small = z.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(z), z)
    return torch.where(small,
                       0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z / 120.0)),
                       (torch.expm1(safe) - safe) / (safe * safe))


def _lambda_per_sample(dv, dx, eps=1e-8):
    """Per-batch-element least-squares lambda (paper Eq.12 / reference)."""
    fv = dv.reshape(dv.shape[0], -1)
    fx = dx.reshape(dx.shape[0], -1)
    num = (fv * fx).sum(dim=1)
    den = (fx * fx).sum(dim=1) + eps
    return (num / den).view(-1, *([1] * (dv.ndim - 1)))


def _aeuler_step(x, v, x_prev, v_prev, t_now, t_prev, t_next,
                 lam_min=None, lam_max=None):
    """One adaptive step (paper Eq.12-16, reference AEuler.step)."""
    c = _lambda_per_sample(v - v_prev, x - x_prev)
    if lam_min is not None and lam_max is not None:
        c = c.clamp(lam_min, lam_max)
    t_s = t_next - t_now
    h_now = v - c * x
    h_prev = v_prev - c * x_prev
    d2 = (h_now - h_prev) / max(t_now - t_prev, 1e-8)
    z = c * t_s
    A = _phi1(z) * t_s
    B = _phi2(z) * t_s * t_s
    return x * torch.exp(z) + A * h_now + B * d2, c


@torch.no_grad()
def sample_paper_aeuler(model, x, sigmas, extra_args=None, callback=None,
                        disable=None, lam_min=None, lam_max=None, **kwargs):
    """A-FloPS A-Euler on the caller's sigma list (any schedule).

    lam_min/lam_max are TESTING knobs: the reference does not clamp lambda, so
    the default is no clamp.  Set them (e.g. -1.0/1.0) to A/B the paper's
    stated stability clamp.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    n_steps = len(sigmas) - 1
    if n_steps < 1:
        return x

    x_prev = v_prev = None
    t_prev = None

    for i in range(n_steps):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        # dx/dsigma (the Karras derivative); t-velocity is v = -d since
        # t = 1 - sigma.
        if sigma > 0.0:
            d = (x - denoised) / sigma
        else:
            d = torch.zeros_like(x)
        v = -d

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
                      "denoised": denoised})

        t_now = 1.0 - sigma
        t_next = 1.0 - sigma_next
        x_before = x

        have_prev = x_prev is not None
        valid_prev = have_prev and (t_now - t_prev) > 0.0 and t_next > t_now

        if not valid_prev:
            # First step (reference AEuler): plain Euler in sigma -- which IS
            # the flow-time Euler step because t = 1 - sigma is linear.
            x = x + d * (sigma_next - sigma)
        else:
            x, _c = _aeuler_step(x, v, x_prev, v_prev, t_now, t_prev, t_next,
                                 lam_min, lam_max)

        x_prev = x_before
        v_prev = v
        t_prev = t_now

    return x

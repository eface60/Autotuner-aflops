import math
import logging
import hashlib
import functools
import os
import torch
import torch.nn.functional as F
import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler
try:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
except Exception:
    pass
try:
    torch.backends.cudnn.benchmark = False
except Exception:
    pass
try:
    torch.backends.cudnn.deterministic = True
except Exception:
    pass
_AFLOPS_TF32_OFF = os.environ.get("AFLOPS_TF32_OFF", "1") == "1"
if _AFLOPS_TF32_OFF:
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    except Exception:
        pass
try:
    from .common import exact_step as _exact_step_ext
except Exception:
    _exact_step_ext = None
_LAST_ERRORS = []
_LAST_CFG = {}
_PROBE_PROFILES = {}       # "<model_key>|s:<sched_hash>" -> profile
_PROFILE_BY_MODEL = {}     # "<model_key>" -> most recent profile (scheduler view)
_COND_PROBE_PROFILES = {}  # (model_key, cond_sig) -> per-prompt profile
_COND_PROBE_PROFILES_SCHED = {}
_NODE_PROBE_PROFILES = {}  # (model_key, param_sig, cfg_sig) -> node-side profile
_EXACT_STEP_STATUS = "unverified"

def _exact_step_local(x, x0, s, s2):
    r = float(s2) / max(float(s), 1e-12)
    return r * x + (1.0 - r) * x0

def _validate_exact_step():
    global exact_step, _EXACT_STEP_STATUS
    if _exact_step_ext is None:
        exact_step = _exact_step_local
        _EXACT_STEP_STATUS = "fallback: .common.exact_step unavailable"
        logging.warning("[A-FloPS] %s", _EXACT_STEP_STATUS)
        return
    try:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, 8, 8, generator=g).double()
        x0 = torch.randn(2, 3, 8, 8, generator=g).double()
        worst = 0.0
        for s, s2 in ((14.0, 7.0), (1.0, 0.3), (0.5, 0.0), (2.0, 1.999)):
            got = _exact_step_ext(x, x0, s, s2)
            want = _exact_step_local(x, x0, s, s2)
            if not torch.isfinite(got).all():
                raise ValueError("non-finite output for s=%s s2=%s" % (s, s2))
            err = float((got - want).abs().max() /
                        want.abs().max().clamp_min(1e-6))
            worst = max(worst, err)
        if worst <= 1e-4:
            exact_step = _exact_step_ext
            _EXACT_STEP_STATUS = ("ok (matches (s2/s)x+(1-s2/s)x0, "
                                  "worst rel dev %.2e)" % worst)
        else:
            exact_step = _exact_step_local
            _EXACT_STEP_STATUS = ("fallback: .common.exact_step deviates %.2e "
                                  "from (s2/s)x+(1-s2/s)x0" % worst)
            logging.warning("[A-FloPS] %s", _EXACT_STEP_STATUS)
    except Exception as e:
        exact_step = _exact_step_local
        _EXACT_STEP_STATUS = "fallback: validation error (%s)" % e
        logging.warning("[A-FloPS] %s", _EXACT_STEP_STATUS)

_validate_exact_step()

def _detect_space(sigmas, mode):
    if mode != "auto":
        return mode
    return "flow" if float(sigmas[0]) <= 1.5 else "ve_edm"

def _ancestral_split(s, sn, eta, space):
    if eta <= 0.0 or sn <= 1e-8:
        return sn, 0.0
    if space == "flow":
        tf = s / max(1.0 - s, 1e-6)
        tt = sn / max(1.0 - sn, 1e-6)
        up_t = eta * math.sqrt(max(tt * tt * (tf * tf - tt * tt) / (tf * tf), 0.0))
        down_t = math.sqrt(max(tt * tt - up_t * up_t, 0.0))
        s_down = down_t / (1.0 + down_t)
        k = down_t / tt
        s_up = sn * math.sqrt(max(1.0 - k * k, 0.0))
        return s_down, s_up
    else:
        up = min(sn, eta * math.sqrt(sn * sn * (s * s - sn * sn) / (s * s)))
        down = math.sqrt(max(sn * sn - up * up, 0.0))
        return down, up

_NOISE_COLOR_PRESETS = {
    "white": 0.0,
    "pink": 1.0,
    "red": 1.5,
    "brown": 2.0,
    "blue": -1.0,
    "violet": -2.0,
}

def _colored_noise(shape, beta, device, dtype, generator=None,
                   decay_floor=1e-3, temporal_beta=0.0, dc_zero=True):
    eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    if beta == 0.0 and temporal_beta == 0.0:
        return eps
    if len(shape) < 2 or shape[-1] < 2 or shape[-2] < 2:
        return eps
    H, W = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.fftfreq(W, device=device)
    fmag = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
    fmag = fmag.clamp_min(float(decay_floor))
    weight = fmag.pow(-float(beta) / 2.0)
    if dc_zero:
        weight = weight.clone()
        weight[0, 0] = 0.0
    weight_r = weight[:, :W // 2 + 1]
    shp = [1] * len(shape)
    shp[-2], shp[-1] = H, W // 2 + 1
    weight_r = weight_r.view(*shp).to(dtype=dtype)
    eps_fft = torch.fft.rfft2(eps, norm="ortho")
    eps_fft = eps_fft * weight_r
    out = torch.fft.irfft2(eps_fft, s=(H, W), norm="ortho")
    if temporal_beta != 0.0 and len(shape) == 5 and shape[2] > 2:
        T = shape[2]
        ft = torch.fft.fftfreq(T, device=device).clamp_min(float(decay_floor))
        wt = ft.pow(-float(temporal_beta) / 2.0)
        if dc_zero:
            wt = wt.clone()
            wt[0] = 0.0
        wt = wt.view(1, 1, -1, 1, 1)[:, :, :T // 2 + 1, :, :].to(dtype=dtype)
        out_fft = torch.fft.rfft(out, dim=2, norm="ortho")
        out_fft = out_fft * wt
        out = torch.fft.irfft(out_fft, n=T, dim=2, norm="ortho")
    lead = 1
    for d in shape[:-2]:
        lead *= d
    flat = out.reshape(lead, -1)
    std = flat.std(dim=-1, keepdim=True).clamp_min(1e-8)
    out = out / std.reshape(*shape[:-2], 1, 1)
    return out

def _spectral_concentration(beta, r_min=1.0 / 64.0, r_max=0.5):
    beta = float(beta)
    if abs(beta) < 1e-6:
        return 1.0

    def _radial_pow_int(exp):
        if abs(exp + 1.0) < 1e-9:
            return math.log(r_max / r_min)
        return (r_max ** (exp + 1.0) - r_min ** (exp + 1.0)) / (exp + 1.0)

    total = 2.0 * math.pi * _radial_pow_int(1.0 - beta)
    num_modes = math.pi * (r_max * r_max - r_min * r_min)
    mean_pm = total / max(num_modes, 1e-12)
    peak = r_min ** (-beta) if beta > 0 else r_max ** (-beta)
    return max(peak / max(mean_pm, 1e-12), 1.0)

class _ColorAwareNoiseSampler:

    def __init__(self, x, seed, noise_color="white", noise_beta=1.0,
                 color_schedule="constant", color_beta_max=1.5,
                 color_beta_min=-0.5, color_temporal=False,
                 color_temporal_beta=0.0, color_decay_floor=1e-3,
                 color_dc_zero=True):
        self.shape = x.shape
        self.device = x.device
        self.dtype = x.dtype
        self.color = str(noise_color)
        self.beta = float(noise_beta)
        self.schedule = str(color_schedule)
        self.beta_max = float(color_beta_max)
        self.beta_min = float(color_beta_min)
        self.temporal = bool(color_temporal)
        self.temporal_beta = float(color_temporal_beta)
        self.decay_floor = float(color_decay_floor)
        self.dc_zero = bool(color_dc_zero)
        self.last_beta = 0.0
        self.generator = torch.Generator(device=x.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.s_max = None
        self.s_min = None

    def _base_beta(self):
        if self.color == "custom":
            return self.beta
        return _NOISE_COLOR_PRESETS.get(self.color, 0.0)

    def _beta_at(self, sigma):
        b0 = self._base_beta()
        if self.schedule == "constant":
            return b0
        s_max = self.s_max if self.s_max is not None else max(sigma, 1e-8)
        s_min = self.s_min if self.s_min is not None else max(s_max * 1e-3, 1e-8)
        ls_hi = math.log(max(s_max, 1e-8))
        ls_lo = math.log(max(s_min, 1e-8))
        ls = math.log(max(sigma, 1e-8))
        frac = (ls_hi - ls) / max(ls_hi - ls_lo, 1e-8)
        frac = min(max(frac, 0.0), 1.0)
        if self.schedule == "fade_to_white":
            return b0 * (1.0 - frac)
        if self.schedule == "linear":
            return self.beta_max + (self.beta_min - self.beta_max) * frac
        if self.schedule == "cns_approx":
            if frac < 0.5:
                return b0 * 0.2 * (1.0 - frac / 0.5)
            t = (frac - 0.5) / 0.5
            return (b0 * 0.1) * (1.0 - t) + self.beta_min * t
        return b0

    def __call__(self, sigma, sigma_next, override_beta=None):
        sig = float(sigma) if sigma is not None else 1.0
        sig_n = float(sigma_next) if sigma_next is not None else 0.0
        if self.s_max is None:
            self.s_max = sig
        self.s_max = max(self.s_max, sig)
        if sig_n > 1e-8:
            self.s_min = min(self.s_min if self.s_min is not None else sig_n, sig_n)
        if self.s_min is None:
            self.s_min = max(sig_n, sig * 1e-3, 1e-8)
        if override_beta is not None:
            beta = float(override_beta)
        else:
            beta = self._beta_at(sig)
        self.last_beta = beta
        out = _colored_noise(
            self.shape, beta, self.device, self.dtype,
            generator=self.generator, decay_floor=self.decay_floor,
            temporal_beta=self.temporal_beta if self.temporal else 0.0,
            dc_zero=self.dc_zero)
        return out

def _make_noise_sampler(x, extra_args, cfg):
    color = str(cfg.get("noise_color", "white"))
    schedule = str(cfg.get("color_schedule", "constant"))
    needs_color = color != "white"
    if not needs_color and schedule != "constant":
        if schedule in ("linear", "cns_approx") and (
                float(cfg.get("color_beta_max", 0.0)) != 0.0 or
                float(cfg.get("color_beta_min", 0.0)) != 0.0):
            needs_color = True
    _seed = extra_args.get("seed", None)
    if _seed is None:
        _seed = 0
    if not needs_color:
        return default_noise_sampler(x, _seed)
    return _ColorAwareNoiseSampler(
        x, _seed,
        noise_color=color,
        noise_beta=float(cfg.get("noise_beta", 1.0)),
        color_schedule=schedule,
        color_beta_max=float(cfg.get("color_beta_max", 1.5)),
        color_beta_min=float(cfg.get("color_beta_min", -0.5)),
        color_temporal=bool(cfg.get("color_temporal", False)),
        color_temporal_beta=float(cfg.get("color_temporal_beta", 0.0)),
        color_decay_floor=float(cfg.get("color_decay_floor", 1e-3)),
        color_dc_zero=bool(cfg.get("color_dc_zero", True)),
    )

_CENTROID_BETA_LUT = None

def _build_centroid_lut():
    global _CENTROID_BETA_LUT
    r_min, r_max = 1.0 / 64.0, 0.5
    betas = []
    centroids = []
    for i in range(81):
        beta = -2.0 + i * 0.05
        if abs(beta - 2.0) < 1e-6:
            c = (r_max - r_min) / math.log(r_max / r_min)
        elif abs(beta - 3.0) < 1e-6:
            c = math.log(r_max / r_min) * r_min * r_max / (r_max - r_min)
        else:
            num = (r_max ** (3.0 - beta) - r_min ** (3.0 - beta)) / (3.0 - beta)
            den = (r_max ** (2.0 - beta) - r_min ** (2.0 - beta)) / (2.0 - beta)
            c = num / den if den > 1e-12 else 0.3
        betas.append(beta)
        centroids.append(c)
    _CENTROID_BETA_LUT = (centroids, betas)

def _centroid_to_beta(centroid):
    if _CENTROID_BETA_LUT is None:
        _build_centroid_lut()
    centroids, betas = _CENTROID_BETA_LUT
    c = float(centroid)
    if c >= centroids[0]:
        return -2.0
    if c <= centroids[-1]:
        return 2.0
    lo, hi = 0, len(centroids) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if centroids[mid] >= c:
            lo = mid
        else:
            hi = mid
    c_lo, c_hi = centroids[lo], centroids[hi]
    b_lo, b_hi = betas[lo], betas[hi]
    if abs(c_lo - c_hi) < 1e-8:
        return b_lo
    t = (c - c_lo) / (c_hi - c_lo)
    return b_lo + t * (b_hi - b_lo)

def _blackman_harris_window_1d(N, device=None, dtype=torch.float32):
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    n = torch.arange(N, device=device, dtype=dtype)
    c = (2.0 * math.pi / float(N)) * n
    return a0 - a1 * torch.cos(c) + a2 * torch.cos(2.0 * c) - a3 * torch.cos(3.0 * c)

def _spectral_centroid(jmap):
    if jmap is None:
        return None
    sh = jmap.shape
    if len(sh) < 3:
        return None
    H, W = sh[-2], sh[-1]
    if H < 4 or W < 4:
        return None
    j = jmap.reshape(-1, H, W).float()
    j = j - j.mean(dim=(-2, -1), keepdim=True)
    win_y = _blackman_harris_window_1d(H, device=j.device, dtype=j.dtype)
    win_x = _blackman_harris_window_1d(W, device=j.device, dtype=j.dtype)
    win_2d = (win_y.view(-1, 1) * win_x.view(1, -1)).unsqueeze(0)  # (1, H, W)
    j = j * win_2d
    fft = torch.fft.rfft2(j, norm="ortho")
    power = (fft.real ** 2 + fft.imag ** 2)
    power = power.mean(dim=0)
    fy = torch.fft.fftfreq(H, device=jmap.device)
    fx = torch.fft.rfftfreq(W, device=jmap.device)
    R = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
    w = torch.full((W // 2 + 1,), 2.0, device=jmap.device)
    w[0] = 1.0
    if W % 2 == 0:
        w[-1] = 1.0
    w = w.view(1, -1)
    pw = power * w
    total = pw.sum()
    if total < 1e-12:
        return 0.33
    centroid = float((R * pw).sum() / total)
    return centroid

def _whitepoint_from_x0s(x0s):
    """Spectral centroid of the mid-trajectory x0's detail map (the probe
    'whitepoint'): measures whether the model's natural output structure is
    blue- or red-heavy.  Shared by the model probe, the cond probe and the
    focused probe so EVERY profile carries it (the engine's model-probe
    fallback via _latest_profile may legitimately serve a per-prompt cond
    profile, and the adaptive-noise prior wants the whitepoint either way)."""
    try:
        if x0s:
            mid = len(x0s) // 2
            jm = x0s[mid].pow(2).mean(dim=1).sqrt()
            c = _spectral_centroid(jm)
            if c is not None:
                return float(c)
    except Exception:
        pass
    return 0.33

def _predict_adaptive_beta(jmap, err, eta_eff, s, space, sigma_data,
                           consistency, cfg, corr_used=None, fresh_frac=None):
    if jmap is None:
        return None
    centroid = _spectral_centroid(jmap)
    if centroid is None:
        return None
    beta_raw = min(_centroid_to_beta(centroid), 0.0)
    if fresh_frac is None:
        fresh_frac = 0.25 * float(eta_eff) * float(eta_eff)
    tau_eta = float(cfg.get("adaptive_tau_eta", 0.15))
    s_eta = math.exp(-max(float(fresh_frac), 0.0) / max(tau_eta, 1e-3))
    s_eta = max(0.0, min(s_eta, 1.0))
    tau_err = float(cfg.get("adaptive_tau_err", 0.5))
    err_norm = (err or 0.0) / max(float(s), 0.01)
    s_err = math.exp(-err_norm / max(tau_err, 1e-4))
    s_err = max(0.0, min(s_err, 1.0))
    snr = _compute_snr(float(s), space, sigma_data)
    log_snr = math.log(max(snr, 1e-8))
    tau_snr = float(cfg.get("adaptive_tau_snr", 2.0))
    s_snr = math.exp(-log_snr * log_snr / (2.0 * tau_snr * tau_snr))
    s_snr = max(0.0, min(s_snr, 1.0))
    s_cons = max(0.0, min(float(consistency), 1.0))
    gamma_cons = float(cfg.get("adaptive_cons_gamma", 1.5))
    s_cons = s_cons ** gamma_cons
    s_corr = 0.7
    safety = s_eta * s_err * s_snr * s_cons * s_corr
    user_strength = float(cfg.get("color_strength", 1.0))
    user_strength = max(0.0, min(user_strength, 1.0))
    beta = safety * user_strength * beta_raw
    beta_min = float(cfg.get("color_beta_min", -2.0))
    beta_max = float(cfg.get("color_beta_max", 2.0))
    beta = max(beta_min, min(beta, beta_max))
    return beta

def _lf_get(lf, name, default):
    v = getattr(lf, name, default)
    if callable(v):
        try:
            v = v()
        except Exception:
            return default
    return v if v is not None else default

def _extract_model_info(model):
    if model is None:
        return None
    try:
        patcher = model
        if hasattr(model, "inner_model"):
            inner = model.inner_model
            if hasattr(inner, "model_patcher"):
                patcher = inner.model_patcher
            elif hasattr(inner, "inner_model") and hasattr(inner.inner_model, "model_patcher"):
                patcher = inner.inner_model.model_patcher
        elif hasattr(model, "model_patcher"):
            patcher = model.model_patcher
        ms = patcher.get_model_object("model_sampling")
        lf = patcher.get_model_object("latent_format")
        mt = patcher.get_model_object("model_type")
        mc = patcher.get_model_object("model_config")
        uc = getattr(mc, "unet_config", {}) or {}
        try:
            import comfy.model_sampling as _ms
            flow_classes = (
                _ms.ModelSamplingDiscreteFlow,
                _ms.ModelSamplingFlux,
                _ms.ModelSamplingAV,
                _ms.ModelSamplingCosmosRFlow,
            )
        except Exception:
            flow_classes = ()
        is_flow = isinstance(ms, flow_classes) if flow_classes else None
        if is_flow is None:
            try:
                from comfy.model_base import ModelType as _MT
                is_flow = int(mt) in (
                    int(_MT.FLOW), int(_MT.FLUX), int(_MT.FLOW_AV),
                    int(_MT.FLOW_COSMOS), int(_MT.IMG_TO_IMG_FLOW),
                )
            except Exception:
                is_flow = None
        image_model = uc.get("image_model")
        guidance_embed = uc.get("guidance_embed")
        is_distilled = None
        try:
            diff_model = patcher.get_model_object("diffusion_model")
            if image_model == "flux" and not guidance_embed:
                n_double = len(getattr(diff_model, "double_blocks", []))
                is_distilled = n_double <= 6
        except Exception:
            pass
        if is_distilled is None:
            try:
                init = getattr(patcher, "cached_patcher_init", None)
                if init and len(init) > 1 and init[1]:
                    fname = str(init[1][0]).lower()
                    is_distilled = any(k in fname for k in
                                       ("turbo", "schnell", "lightning",
                                        "dmd", "lcm", "hyper"))
            except Exception:
                pass
        return {
            "is_flow": is_flow,
            "model_type": int(mt) if mt is not None else None,
            "sigma_min": float(getattr(ms, "sigma_min", 0.0)),
            "sigma_max": float(getattr(ms, "sigma_max", 1.0)),
            "sigma_data": float(getattr(ms, "sigma_data", 1.0)),
            "shift": getattr(ms, "shift", None),
            "multiplier": getattr(ms, "multiplier", None),
            "latent_channels": int(_lf_get(lf, "latent_channels", 4)),
            "latent_dimensions": int(_lf_get(lf, "latent_dimensions", 2)),
            "spacial_downscale_ratio": int(_lf_get(lf, "spacial_downscale_ratio", 8)),
            "temporal_downscale_ratio": int(_lf_get(lf, "temporal_downscale_ratio", 1)),
            "image_model": image_model,
            "in_channels": uc.get("in_channels"),
            "out_channels": uc.get("out_channels"),
            "guidance_embed": guidance_embed,
            "is_distilled": is_distilled,
            "is_video": int(_lf_get(lf, "latent_dimensions", 2)) >= 3,
        }
    except Exception:
        return None

_PROBE_RANK_CACHE = {}

def _probe_rank_key(patcher):
    try:
        if patcher is None:
            return None
        return _model_cache_key(patcher)
    except Exception:
        return None

def _remember_probe_rank(patcher, rank):
    try:
        if rank not in (4, 5):
            return
        k = _probe_rank_key(patcher)
        if k is not None:
            _PROBE_RANK_CACHE[k] = int(rank)
    except Exception:
        pass

def _detect_probe_rank(patcher, mi=None):
    if patcher is None:
        return None
    try:
        if mi is None:
            try:
                mi = _extract_model_info(patcher)
            except Exception:
                mi = None
        if mi is not None:
            try:
                ld = int(mi.get("latent_dimensions") or 0)
            except (TypeError, ValueError):
                ld = 0
            if ld >= 3:
                return 5
    except Exception:
        pass
    try:
        uc = patcher.get_model_object("model_config").unet_config or {}
        if isinstance(uc, dict) and (("patch_temporal" in uc)
                                     or ("max_frames" in uc)
                                     or ("temporal_patch_size" in uc)):
            return 5
    except Exception:
        pass
    try:
        dm = patcher.get_model_object("diffusion_model")
        for _n, mod in dm.named_modules():
            pat = getattr(mod, "pattern", None)
            if isinstance(pat, str) and "(t" in pat and "(h" in pat \
                    and "(w" in pat:
                return 5
    except Exception:
        pass
    return None

def _resolve_probe_rank(patcher, ref_x=None, mi=None):
    try:
        k = _probe_rank_key(patcher)
        if k is not None:
            v = _PROBE_RANK_CACHE.get(k)
            if v in (4, 5):
                return int(v)
    except Exception:
        pass
    try:
        if torch.is_tensor(ref_x):
            d = int(ref_x.dim())
            if d in (4, 5):
                return d
    except Exception:
        pass
    try:
        r = _detect_probe_rank(patcher, mi)
        if r in (4, 5):
            return r
    except Exception:
        pass
    return 4

def _probe_rank_ladder(patcher, ref_x=None, mi=None):
    ladder = [_resolve_probe_rank(patcher, ref_x, mi)]
    try:
        if torch.is_tensor(ref_x):
            d = int(ref_x.dim())
            if d in (4, 5) and d not in ladder:
                ladder.append(d)
    except Exception:
        pass
    try:
        r = _detect_probe_rank(patcher, mi)
        if r in (4, 5) and r not in ladder:
            ladder.append(r)
    except Exception:
        pass
    for r in (4, 5):
        if r not in ladder:
            ladder.append(r)
    return ladder

def _probe_latent_shape(ref_x=None, pres=16, batch=1, rank=None,
                        channels=None, patcher=None, mi=None):
    try:
        batch = int(batch)
        if batch < 1:
            batch = 1
    except Exception:
        batch = 1
    try:
        pres = max(1, int(pres))
    except Exception:
        pres = 16
    if rank not in (4, 5):
        rank = _resolve_probe_rank(patcher, ref_x, mi)
    if channels is None:
        try:
            channels = int(ref_x.shape[1]) if torch.is_tensor(ref_x) else 0
        except Exception:
            channels = 0
    try:
        channels = int(channels)
    except Exception:
        channels = 4
    if channels < 1:
        channels = 4
    if rank == 5:
        try:
            if torch.is_tensor(ref_x) and ref_x.dim() == 5:
                T = int(ref_x.shape[2])
                if 0 < T <= 64:
                    return (batch, channels, T, pres, pres)
        except Exception:
            pass
        return (batch, channels, 1, pres, pres)
    return (batch, channels, pres, pres)

def _is_shape_error(e):
    try:
        if _is_device_error(e):
            return False
        if isinstance(e, IndexError):
            return True
        msg = str(e).lower()
    except Exception:
        return False
    for token in ("tuple index out of range", "index out of range",
                  "number of dimensions", "dimension mismatch",
                  "dimensions mismatch", "expected 3d", "expected 4d",
                  "expected 5d", "got 3d", "got 4d", "got 5d",
                  "expected 3-dimensional", "expected 4-dimensional",
                  "expected 5-dimensional", "got 3-dimensional",
                  "got 4-dimensional", "got 5-dimensional",
                  "invalid shape", "shape mismatch", "size mismatch",
                  "mat1 and mat2", "calculated padded input size",
                  "output size is too small", "expected input batch_size",
                  "einops", "rearrange", "weight of size"):
        if token in msg:
            return True
    return False

def _navigate_to_patcher(model):
    if model is None:
        return None
    try:
        patcher = model
        if hasattr(model, "inner_model"):
            inner = model.inner_model
            if hasattr(inner, "model_patcher"):
                patcher = inner.model_patcher
            elif hasattr(inner, "inner_model") and hasattr(inner.inner_model, "model_patcher"):
                patcher = inner.inner_model.model_patcher
        elif hasattr(model, "model_patcher"):
            patcher = model.model_patcher
        if hasattr(patcher, "get_model_object"):
            return patcher
        return None
    except Exception:
        return None

def _model_cache_key(model):
    patcher = _navigate_to_patcher(model)
    if patcher is not None:
        try:
            mh = getattr(patcher, "model_hash", None)
            if callable(mh):
                v = mh()
                if v:
                    return "mh:" + str(v)
        except Exception:
            pass
        try:
            dm = patcher.get_model_object("diffusion_model")
            acc = 0.0
            n = 0
            for p in list(dm.parameters())[:4]:
                flat = p.detach().float().flatten()
                if flat.numel() > 0:
                    acc += float(flat[:: max(1, flat.numel() // 64)][:64].sum())
                    n += 1
            if n > 0:
                return "wc:%.6f:%d" % (acc, n)
        except Exception:
            pass
        base = getattr(patcher, "model", None)
        if base is not None:
            return "bm:" + str(id(base))
    return "id:" + str(id(model))

def _schedule_sig(x, sigmas, cfg):
    try:
        h = hashlib.sha1()
        h.update(str(tuple(int(v) for v in x.shape[1:])).encode())
        try:
            st = sigmas.detach().to("cpu", torch.float32).contiguous().flatten()
            st_np = st.numpy()
            h.update(st_np.tobytes())
        except Exception:
            st = sigmas.detach().to("cpu", torch.float32).contiguous().flatten()
            h.update(str([round(float(v), 8) for v in st]).encode())
        h.update(("ps=%d,pr=%d" % (int(cfg.get("probe_steps", 10)),
                                   int(cfg.get("probe_resolution", 48)))).encode())
        return h.hexdigest()[:16]
    except Exception:
        return "nologic"

def _prune_probe_cache(mk, keep=8):
    keys = [k for k in _PROBE_PROFILES if k == mk or k.startswith(mk + "|s:")]
    for k in keys[:-keep] if len(keys) > keep else []:
        _PROBE_PROFILES.pop(k, None)

def _cond_signature(positive, negative):
    def _tensor_part(t):
        tf = t.detach().float()
        return "T%s:%.6e:%.6e:%.6e" % (
            tuple(tf.shape), float(tf.sum()),
            float(tf.norm()), float(tf.mean()))

    def _entry_part(entry):
        if isinstance(entry, dict):
            keys = sorted(str(k) for k in entry.keys() if k != "uuid")
            parts.append("D%d:%s" % (len(keys), ",".join(keys)))
            for k in keys:
                v = entry.get(k)
                if torch.is_tensor(v):
                    parts.append(str(k) + "=" + _tensor_part(v))
        elif isinstance(entry, (list, tuple)) and entry:
            t = entry[0]
            if torch.is_tensor(t):
                parts.append(_tensor_part(t))
            else:
                parts.append(repr(t)[:64])
            if len(entry) > 1 and isinstance(entry[1], dict):
                parts.append(",".join(sorted(entry[1].keys())))
        else:
            parts.append(repr(entry)[:64])

    parts = []
    try:
        for cond in (positive, negative):
            parts.append("#%d" % (len(cond) if cond else 0))
            for entry in (cond or []):
                _entry_part(entry)
    except Exception:
        parts.append("err")
    return hashlib.sha256("|".join(parts).encode("utf-8", "replace")).hexdigest()[:16]

def _extra_cond_signature(extra_args):
    try:
        if not isinstance(extra_args, dict):
            return None
        pos = extra_args.get("cond")
        neg = extra_args.get("uncond")
        if pos is None and neg is None:
            return None
        return _cond_signature(pos, neg)
    except Exception:
        return None

def _cond_cache_key(cfg, extra_args, model_key):
    cond_sig = cfg.get("cond_probe_sig")
    if not cond_sig:
        cond_sig = _extra_cond_signature(extra_args)
    return (model_key, str(cond_sig)) if cond_sig else None

def _latest_profile(model):
    try:
        mk = _model_cache_key(model)
    except Exception:
        return None
    prof = _PROFILE_BY_MODEL.get(mk)
    if prof is not None:
        return prof
    hit = None
    for k, v in _PROBE_PROFILES.items():
        if k == mk or k.startswith(mk + "|s:"):
            hit = v
    return hit

def _lagrange_weights(us, u_t):
    n = len(us)
    lam = []
    for j in range(n):
        d = 1.0
        for k in range(n):
            if k == j:
                continue
            d *= (us[j] - us[k])
        lam.append(1.0 / d if d != 0.0 else 0.0)
    for j in range(n):
        if abs(u_t - us[j]) < 1e-15:
            ws = [0.0] * n
            ws[j] = 1.0
            return ws
    cs = []
    for j in range(n):
        cs.append(lam[j] / (u_t - us[j]))
    s = sum(cs)
    if abs(s) < 1e-300:
        return [1.0 / n] * n
    return [c / s for c in cs]

def _mid_x0(pts, u_mid, order):
    sel = pts[-min(order, len(pts)):]
    us = [p[0] for p in sel]
    if len(set(us)) != len(us):
        return sel[-1][1], 0.0
    ws = _lagrange_weights(us, u_mid)
    out = sel[0][1] * ws[0]
    for w, p in zip(ws[1:], sel[1:]):
        out = out + p[1] * w
    return out, max(abs(w) for w in ws)
_ORD_BIG = 1e30
_AFLOPS_LAM_MIN = -3.0
_AFLOPS_LAM_MAX = 0.0
_AFLOPS_X0M_ENV = 1.3
_ORD_EMA = 0.5        # innovation-stack EMA weight (half-life: one step)
_ORD_Z = 3.5          # robust outlier threshold (modified z-score)
_ORD_R_FLOOR = 1.5    # novelty must ALSO beat the pixel's own envelope
_ORD_ENV_DECAY = 0.7  # motion-envelope decay (was the order_jema_decay knob)
_GATE_ETA = 0.15      # integral rate (per steered step)
_GATE_TARGET = 0.10   # required net efficacy for full authority
_GATE_FLOOR = 0.05    # exploration floor when gated down
_LF_ACT_M = 0.3       # |m| > 0.3 * amp counts as steered (same as C1)
_LF_ACT_RHO = 0.05    # refresh fraction that counts as re-drawn material

def _pixel_innovations(hist_pts, u_t, x0_cur, p_max):
    out = []
    n = len(hist_pts)
    for p in range(1, int(p_max) + 1):
        if n < p:
            out.append(None)
            continue
        sel = hist_pts[-p:]
        us = [q[0] for q in sel]
        if len(set(us)) != len(us):
            out.append(None)
            continue
        ws = _lagrange_weights(us, u_t)
        pred = sel[0][1].float() * ws[0]
        for w, q in zip(ws[1:], sel[1:]):
            pred = pred + q[1].float() * w
        out.append((pred - x0_cur.float()).pow(2).mean(dim=1, keepdim=True)
                   .sqrt())
    return out

def _grad_weighted_norm(diff, x0, strength=5.0):
    gx = (x0[..., 1:, :] - x0[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    gy = (x0[..., :, 1:] - x0[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    gx = F.pad(gx, (0, 0, 0, 1))
    gy = F.pad(gy, (0, 1, 0, 0))
    g = gx + gy
    g = g / g.mean().clamp_min(1e-8)
    w = (1.0 + float(strength) * g).clamp(1.0, 10.0)
    dw = diff * w
    return float(dw.norm() / (x0 * w).norm().clamp_min(1e-8))
_PERCEPTUAL_DOG_GAIN = math.sqrt(327680.0 / 329.0)


def _perceptual_error_norm(diff, ref):
    sp = diff.shape
    d2 = diff.reshape(-1, 1, sp[-2], sp[-1]).float()
    g_small = _gaussian_blur_separable(d2, 3)   # [1,2,1]/4       (radius 1)
    g_large = _gaussian_blur_separable(d2, 5)   # [1,4,6,4,1]/16  (radius 2)
    d_hp = ((g_small - g_large) * _PERCEPTUAL_DOG_GAIN).reshape(sp)
    err_lo = diff.flatten().float().norm()                  # ||diff||
    err_hi = d_hp.flatten().float().norm()                  # ||DoG(diff)||
    ref_norm = ref.flatten().float().norm().clamp_min(1e-8)
    return float(torch.sqrt(err_lo * err_lo + 0.5 * err_hi * err_hi) / ref_norm)

def _err_norm(a, b, mode, x0_ref=None, strength=5.0):
    diff = (a - b).float()
    if mode == "grad_l2" and x0_ref is not None:
        return _grad_weighted_norm(diff, x0_ref.float(), strength=strength)
    if mode == "perceptual":
        return _perceptual_error_norm(diff, b)
    diff_flat = diff.flatten()
    if mode == "abs_l2":
        return float(diff_flat.norm() / math.sqrt(max(1, diff_flat.numel())))
    return float(diff_flat.norm() / b.flatten().float().norm().clamp_min(1e-8))

def _eta_schedule(eta, mode, frac, s, s_max, falloff, eta_min):
    if eta <= 0.0:
        return 0.0
    if mode == "linear":
        e = eta * (1.0 - frac)
    elif mode == "exponential":
        e = eta * math.exp(-max(float(falloff), 0.0) * frac)
    elif mode == "cosine":
        e = eta * 0.5 * (1.0 + math.cos(math.pi * frac))
    elif mode == "sigma_power":
        base = min(s / max(s_max, 1e-8), 1.0)
        e = eta * (base ** max(float(falloff), 1e-4))
    else:
        e = eta
    return max(e, min(float(eta_min), eta))

_ETA_ESCAPE_JUMP = 0.005    # fallback: median jump / median |x0| below this => stalled
_ETA_ESCAPE_DETAIL = 0.25   # fallback: ||highpass(x0)|| / ||x0|| below this => flat
_ETA_FADE_FLOOR = 0.25      # fraction of the baseline eta retained after the late fade
_ETA_FADE_SIGMA = 0.35      # fallback convergence sigma (used when the probe gave none)
_ETA_FADE_LO_RATIO = 0.25   # the fade completes at this fraction of the convergence sigma


def _eta_late_fade(s, cfg=None):
    """Decay the baseline ancestral noise through the late stage, anchored at
    the sigma where the probe measured the model's convergence (eta_fade_sigma
    from the profile's conv_step).  The noise is needed early/mid (to stop
    blotchy regions forming), but a constant level keeps oscillating the x0
    trajectory through the endgame; fading it lets structure lock in.  The
    escape (local-minimum) nudge is NOT faded -- a stuck trajectory still gets
    its noise."""
    c = cfg or {}
    conv_sigma = c.get("eta_fade_sigma")
    if conv_sigma is None:
        conv_sigma = _ETA_FADE_SIGMA
    conv_sigma = max(float(conv_sigma), 1e-6)
    floor = float(c.get("eta_fade_floor", _ETA_FADE_FLOOR) or _ETA_FADE_FLOOR)
    lo = conv_sigma * _ETA_FADE_LO_RATIO
    t = (conv_sigma - float(s)) / max(conv_sigma - lo, 1e-9)
    t = min(max(t, 0.0), 1.0)
    ss = t * t * (3.0 - 2.0 * t)  # smoothstep
    return 1.0 - (1.0 - floor) * ss


def _stuck_components(x0, jmap):
    """Return (jump_rel, detail_abs) for the global stuck detector, or
    (None, None) when unavailable.  Exposed separately so the engine can log
    the raw values per step (needed to calibrate the thresholds)."""
    if jmap is None or x0 is None:
        return None, None
    try:
        xm = x0.float().pow(2).mean(dim=1).sqrt()
        med_xm = float(xm.median().clamp_min(1e-8))
        med_j = float(jmap.float().median().clamp_min(1e-8))
        jump_rel = med_j / med_xm
        x4 = x0.float().reshape(-1, x0.shape[1], *x0.shape[-2:])
        hp = x4 - F.avg_pool2d(x4, 3, stride=1, padding=1)
        detail_abs = float(hp.norm() / x4.norm().clamp_min(1e-8))
        return jump_rel, detail_abs
    except Exception:
        return None, None


def _global_stuck_signal(x0, jmap, cfg=None, dir_cos=None, s=None):
    """Global local-minimum detector, in [0, 1].

    Fires on EITHER of two stuck states:

    1. STALLED: the median per-pixel x0 jump has collapsed (relative to the x0
       magnitude) AND the image is still flat (low absolute high-frequency
       energy).  The flatness condition keeps the nudge from firing at normal
       convergence, where the jump also collapses but the image is already
       detailed.

    2. OSCILLATING: the x0 update reverses the previous update (dir_cos < 0),
       and only BELOW the probe-measured oscillation onset (eta_fade_sigma =
       the profile's osc_sigma).  A trajectory can be stuck in a local minimum
       while its median jump is still non-trivial -- every pixel moves, just
       back and forth.  The early "commit" also reverses direction, but that
       is the model committing, not a wobble, so the reversal is only a stuck
       signal after the oscillation onset.  It scales with the reversal
       strength: monotone contributes nothing, a full reversal saturates.

    Thresholds are cfg-tunable (eta_escape_jump / eta_escape_detail) and
    deliberately absolute rather than median-relative so a global stall (every
    pixel stops at once) is still detected.
    """
    jump_rel, detail_abs = _stuck_components(x0, jmap)
    if jump_rel is None or detail_abs is None:
        return 0.0
    c = cfg or {}

    def _thr(key, fallback):
        v = c.get(key, fallback)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = fallback
        # a negative value is the "auto" sentinel; the engine derives the real
        # threshold from the probe before the loop, so here it only means "no
        # probe measurement was available" -> use the constant fallback.
        if v is None or v < 0:
            v = fallback
        return max(v, 1e-9)

    j_thr = _thr("eta_escape_jump", _ETA_ESCAPE_JUMP)
    d_thr = _thr("eta_escape_detail", _ETA_ESCAPE_DETAIL)
    stag_score = 1.0 - min(max(jump_rel / j_thr, 0.0), 1.0)
    flat_score = 1.0 - min(max(detail_abs / d_thr, 0.0), 1.0)
    stall = stag_score * flat_score
    osc = 0.0
    if dir_cos is not None and dir_cos < 0.0:
        osc = min(1.0, -float(dir_cos))
        _osc_on = c.get("eta_fade_sigma")
        if _osc_on is not None and s is not None and float(s) > float(_osc_on):
            # above the oscillation onset the reversal is the commit, not a
            # wobble; keep the nudge for the late phase only.
            osc = 0.0
    return max(stall, osc)


def _adaptive_eta(err, consistency, s, sn, s_max, s_min, space, sigma_data,
                  s_noise, model_info, eta_ceiling, cfg, few_step=False,
                  stuck=0.0):
    """Adaptive stochasticity (v15): minimal noise + local-minimum escape.

    The base level is eta_min (default 0 = fully deterministic), so the
    trajectory is free to lock structure in.  Noise is only raised -- toward
    the stochasticity ceiling -- while `stuck` (the global local-minimum
    signal) is non-zero.  The escape is self-limiting: the injected noise makes
    the trajectory move again, which lowers `stuck` on the next step.

    The base (floor) noise is faded through the late stage (see
    _eta_late_fade, anchored at the probe-measured convergence sigma) so the
    endgame can lock in; the escape nudge rides on top of the faded floor and
    is NOT faded, so a genuinely stuck trajectory still gets its noise.
    Replaces the old burn-in / error / SNR / stability multipliers, which
    pegged eta at the ceiling for flow models.
    """
    floor = max(0.0, float(cfg.get("eta_min", 0.0) or 0.0))
    # eta_min is a FLOOR: if the user sets it above the stochasticity slider the
    # effective level is eta_min (constant, no escape headroom) rather than
    # silently collapsing to 0.
    ceil = max(float(eta_ceiling), floor)
    floor = floor * _eta_late_fade(float(s), cfg)
    stuck = max(0.0, min(float(stuck), 1.0))
    return floor + (ceil - floor) * stuck

def _jump_signal(x0, prev_x0, x, q=0.99):
    diff = (x0 - prev_x0).pow(2).mean(dim=1)
    resid = (x - x0).pow(2).mean(dim=1)
    ref = x0.detach().pow(2).mean(dim=1)
    r = (diff + 1e-12).sqrt() / ((resid + 1e-12).sqrt() + 0.02 * (ref + 1e-12).sqrt())
    flat = r.flatten().float()
    n = flat.numel()
    if n > (1 << 22):
        flat = flat[:: n // (1 << 22) + 1]
    return float(torch.quantile(flat, q))

class _OutlierGuard:

    def __init__(self, window=8, z=3.0, floor=0.05, ratio=3.0, hold=2,
                 warmup=4, mad_floor=0.1):
        self.window = max(4, int(window))
        self.z = float(z)
        self.floor = float(floor)
        self.ratio = float(ratio)
        self.hold = int(hold)
        self.warmup = int(warmup)
        self.mad_floor = float(mad_floor)
        self.hist = []
        self.deltas = []
        self.left = 0
        self.last_sev = 1.0

    def step(self, signal, z=None):
        z = self.z if z is None else float(z)
        trip = False
        self.last_sev = 1.0
        if self.hist:
            delta = signal - self.hist[-1]
            if len(self.deltas) >= self.warmup:
                arr = sorted(self.deltas)
                dmed = arr[len(arr) // 2]
                dmad = sorted(abs(d - dmed) for d in self.deltas)[len(self.deltas) // 2]
                level = sorted(self.hist)[len(self.hist) // 2]
                lvl = max(level, 1e-6)
                dmad_eff = max(dmad, self.mad_floor * lvl)
                dthr = max(dmed + z * 1.4826 * dmad_eff, self.floor * lvl)
                trip = delta > dthr or signal > self.ratio * lvl
                if trip:
                    self.last_sev = max(signal / max(dthr, 1e-8), 1.0)
            self.deltas.append(delta)
            if len(self.deltas) > self.window:
                self.deltas.pop(0)
        if trip:
            self.left = self.hold + 1
        active = self.left > 0
        self.hist.append(signal)
        if len(self.hist) > self.window:
            self.hist.pop(0)
        if self.left > 0:
            self.left -= 1
        return trip, active

def _gaussian_kernel1d(radius, dtype=torch.float32, device=None):
    from math import comb
    n = 2 * int(radius)
    coeffs = [float(comb(n, k)) for k in range(n + 1)]
    s = sum(coeffs)
    t = torch.tensor(coeffs, device=device, dtype=dtype) / s
    return t

def _gaussian_blur_separable(g, win):
    sh = g.shape
    g4 = g.reshape(-1, 1, sh[-2], sh[-1]).float()
    radius = max(1, int(win) // 2)
    k = _gaussian_kernel1d(radius, dtype=g4.dtype, device=g4.device)
    pad = radius
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    g4 = F.pad(g4, (pad, pad, 0, 0), mode="replicate")
    g4 = F.conv2d(g4, kx)
    g4 = F.pad(g4, (0, 0, pad, pad), mode="replicate")
    g4 = F.conv2d(g4, ky)
    return g4.reshape(sh)

def _local_mean(g, win):
    return _gaussian_blur_separable(g, win)

def _clean_mask(m, erode=1, dilate=2):
    sh = m.shape

    def pool(x, k):
        return F.max_pool2d(x.reshape(-1, 1, sh[-2], sh[-1]).float(), k,
                            stride=1, padding=k // 2)

    if erode > 0:
        m = ~(pool(~m, 2 * erode + 1) > 0)
    if dilate > 0:
        m = pool(m, 2 * dilate + 1) > 0
    return m.reshape(sh)

def _neighbor_count(m):
    sh = m.shape
    m4 = m.reshape(-1, 1, sh[-2], sh[-1]).float()
    cnt = F.avg_pool2d(m4, 3, stride=1, padding=1) * 9.0
    return cnt.reshape(sh)

def _anomaly_mask(x0, prev_x0, j_prev_max, win, s_thr, n_thr, v_thr,
                  jmap=None, region_z=12.0):
    if jmap is None:
        jmap = (x0 - prev_x0).pow(2).mean(dim=1).sqrt()
    scale = float(jmap.mean()) + 1e-8
    loc = _local_mean(jmap, win)
    spatial = (jmap / (loc + 0.1 * scale)) > float(s_thr)
    v = x0.pow(2).mean(dim=1).sqrt()
    vloc = _local_mean(v, win)
    value = (v / (vloc + 1e-6)) > float(v_thr)
    if j_prev_max is not None:
        nov = (jmap / (j_prev_max + 0.1 * scale)) > float(n_thr)
        m_small = spatial & nov & value
        flat = jmap.reshape(jmap.shape[0], -1).float()
        med = flat.median(dim=1).values.unsqueeze(1)
        mad = (flat - med).abs().median(dim=1).values.unsqueeze(1)
        zr = ((flat - med) / (1.4826 * mad + 0.05 * med + 1e-6)) > float(region_z)
        zr2d = zr.reshape(jmap.shape) & nov
        cnt = _neighbor_count(zr2d)
        m = m_small | (zr2d & (cnt >= 3.0))
    else:
        m = spatial & value
    return _clean_mask(m, erode=1, dilate=2)

def _elevation_mask(jmap, win, v_thr, x0, mult=3.0):
    flat = jmap.reshape(jmap.shape[0], -1).float()
    med = flat.median(dim=1).values.view(-1, *([1] * (jmap.dim() - 1)))
    scale = float(jmap.mean()) + 1e-8
    m = jmap > (float(mult) * med + 0.1 * scale)
    v = x0.pow(2).mean(dim=1).sqrt()
    vloc = _local_mean(v, win)
    return m & ((v / (vloc + 1e-6)) > float(v_thr))
_LF_CENTERING = "weighted_const"
_LF_V_MAX = 1.70
_LF_DRIFT_RELAX = 0.30
_LF_BAND_RHO = 0.05     # fallback band refresh (when no band_curve baseline is available)
_LF_FRAG_GAIN = 0.75    # fallback fragility gain (when no local-error measurement exists)
_FRAG_ERR_REF = 0.006   # reference real local error: a model at this gets frag_gain ~ 1.0
_FRAG_GAIN_MIN = 0.15
_FRAG_GAIN_MAX = 1.5
_BAND_RHO_MAX = 0.10    # max band refresh at 2x the measured hi-band baseline
_BAND_RHO_REF = 0.25    # excess (fraction above baseline) where the refresh starts

def _lf_v_step(v, vmul):
    return min(float(v) * float(vmul), _LF_V_MAX)

def _lf_dir_weight(cos, db=None):
    cos = cos if torch.is_tensor(cos) else torch.as_tensor(cos)
    if db is None:
        db = 0.25
    db = min(max(float(db), 0.0), 0.9)
    return (1.0 + 0.6 * ((-cos - db) / max(1.0 - db, 1e-6)).clamp(0.0, 1.0)
            - 0.2 * ((cos - 0.7) / 0.3).clamp(0.0, 1.0)).clamp(0.8, 1.6)

def _lf_map_median(m):
    flat = m.reshape(m.shape[0], -1).float()
    med = flat.median(dim=1).values.view(-1, *([1] * (m.dim() - 1)))
    return med.clamp_min(1e-6)

def _lf_detail_map(x0):
    sh = x0.shape
    x4 = x0.reshape(-1, sh[1], *sh[-2:]).float()
    hp = x4 - F.avg_pool2d(x4, 3, stride=1, padding=1)
    d = hp.pow(2).mean(dim=1).sqrt()
    med = d.reshape(d.shape[0], -1).median(dim=1).values.view(-1, 1, 1).clamp_min(1e-6)
    d = (d / med).clamp(1e-4, 1e4).log().clamp(-4.0, 4.0)
    return d.reshape(sh[0:1] + (1,) + sh[2:])

def _sstep(t):
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)

def _lf_env_fade(kn, eg_n=0.0):
    _fade_floor = 0.15 + 0.85 * float(eg_n)  # 0.15 at eg_n=0, 1.0 at eg_n=1
    return _fade_floor + (1.0 - _fade_floor) * (1.0 - _sstep((float(kn) - 0.6) / 0.3))

def _lf_block_scale(jmap, block=4):
    sh = jmap.shape
    j4 = (jmap.reshape(-1, 1, *sh[-2:]).float() + 1e-12).log()
    blk = F.avg_pool2d(j4, int(block))
    full = F.interpolate(blk, size=sh[-2:], mode="bilinear", align_corners=False)
    return blk, full.reshape(sh)

def _lf_block_delta_bands(jmap, block=4):
    sh = jmap.shape
    j4 = jmap.reshape(-1, 1, *sh[-2:]).float()
    b1 = F.avg_pool2d(j4, 3, stride=1, padding=1)
    b2 = F.avg_pool2d(b1, 3, stride=1, padding=1)
    p = lambda t: F.avg_pool2d(t.pow(2), int(block))
    e_hi, e_mid, e_lo = p(j4 - b1), p(b1 - b2), p(b2)
    tot = (e_hi + e_mid + e_lo).clamp_min(1e-20)
    return e_lo / tot, tot

def _lf_rescue_window(kn, start=0.55, full=0.72):
    t_in = _sstep((float(kn) - float(start)) / max(float(full) - float(start), 1e-6))
    t_out = 1.0 - 0.5 * _sstep((float(kn) - 0.92) / 0.08)
    return t_in * t_out

def _lf_band_fractions(jmap):
    sh = jmap.shape
    j4 = jmap.reshape(-1, 1, *sh[-2:]).float()
    b1 = _gaussian_blur_separable(j4, 5)
    b2 = _gaussian_blur_separable(b1, 5)
    hi = j4 - b1
    mid = b1 - b2
    lo = b2
    e_hi = hi.pow(2).mean(dim=(1, 2, 3))
    e_mid = mid.pow(2).mean(dim=(1, 2, 3))
    e_lo = lo.pow(2).mean(dim=(1, 2, 3))
    tot = (e_hi + e_mid + e_lo).clamp_min(1e-12)
    n_bt = e_hi.shape[0]
    b = max(1, sh[0])
    if n_bt != b:
        e_hi = e_hi.reshape(b, -1).mean(dim=1)
        e_mid = e_mid.reshape(b, -1).mean(dim=1)
        e_lo = e_lo.reshape(b, -1).mean(dim=1)
        tot = (e_hi + e_mid + e_lo).clamp_min(1e-12)
    return ((e_hi / tot).mean().item(), (e_mid / tot).mean().item(),
            (e_lo / tot).mean().item())

def _lf_band_split(c):
    sh = c.shape
    c4 = c.reshape(-1, sh[1], *sh[-2:]).float()
    b1 = _gaussian_blur_separable(c4, 5)
    b2 = _gaussian_blur_separable(b1, 5)
    hi = (c4 - b1).to(c.dtype)
    mid = (b1 - b2).to(c.dtype)
    lo = b2.to(c.dtype)
    return hi.reshape(sh), mid.reshape(sh), lo.reshape(sh)

def _lf_window(kn):
    t_in = min(max((kn - 0.15) / 0.25, 0.0), 1.0)
    t_out = 1.0 - min(max((kn - 0.60) / 0.25, 0.0), 1.0)
    return (t_in * t_in * (3.0 - 2.0 * t_in)) * (t_out * t_out * (3.0 - 2.0 * t_out))

def _lf_vol_window(kn):
    t_out = 1.0 - min(max((kn - 0.55) / 0.25, 0.0), 1.0)
    return t_out * t_out * (3.0 - 2.0 * t_out)

def _lf_step_field(x0, jmap, vol_prev, med_ref, damp, w, w_vol, gains, kn=1.0,
                   trust=None, w_dir=None, prior=None, spatial_health=False,
                   block=4, rescue=None, resc_state=None, step_d=None,
                   gdir=None, eg_n=0.0, cum=None, gate=None, yield_map=None,
                   fragility=None):
    with torch.no_grad():
        d = _lf_detail_map(x0)
        raw = d.new_zeros(d.shape)
        rho_raw = d.new_zeros(d.shape)
        vol_new = None
        env = _lf_env_fade(kn, eg_n) if gains.get("env_fade", False) else 1.0
        amp_eff = gains["amp"] * env
        ramp_eff = gains["ramp"] * env
        if jmap is not None:
            j = jmap.float().unsqueeze(1)
            med = _lf_map_median(jmap).unsqueeze(1)
            health = None
            if spatial_health and min(jmap.shape[-2:]) >= int(block):
                blk_log, blk_full = _lf_block_scale(jmap, block)
                if med_ref is None or med_ref.shape != blk_log.shape:
                    med_ref = blk_log.clone()
                else:
                    med_ref = torch.maximum(blk_log, med_ref + math.log(0.8))
                health_blk = (0.5 * (blk_log - med_ref) + 1.0).clamp(0.0, 1.0)
                hb = F.interpolate(health_blk, size=jmap.shape[-2:],
                                   mode="bilinear", align_corners=False)
                if hb.shape[0] != jmap.shape[0]:
                    health = hb.reshape(jmap.shape[0], jmap.shape[1],
                                        *jmap.shape[-2:]).unsqueeze(1)
                else:
                    health = hb  # (B,1,H,W), already channel-aligned
                while health.dim() < raw.dim():
                    health = health.unsqueeze(2)
                scale_eff = torch.maximum(med, blk_full.exp().unsqueeze(1))
            else:
                if med_ref is None:
                    med_ref = med.clone()
                else:
                    med_ref = torch.maximum(med, 0.8 * med_ref)
                health = ((med / med_ref.clamp_min(1e-12)).clamp_min(1e-8).log()
                          * 0.5 + 1.0).clamp(0.0, 1.0)
                scale_eff = med
            lj = (j / scale_eff).clamp(1e-4, 1e4).log().clamp(-4.0, 4.0)
            vol_new = lj if vol_prev is None else 0.5 * vol_prev + 0.5 * lj
            dgate = (1.0 - 0.5 * gains["d"] * torch.sigmoid(d / 0.7)).clamp(0.0, 1.0)
            vol_w = w_dir if w_dir is not None else 1.0
            raw = raw + (gains["v"] * torch.tanh(vol_new / 0.6) * dgate
                         * w_vol * health * vol_w)
            stag = (scale_eff / j.float().clamp_min(1e-6)).log().clamp(0.0, 4.0)
            bold = (torch.tanh(stag / 0.8) * torch.sigmoid(-d / 0.5)
                    * w * health)
            raw = raw - gains["s"] * bold
            rho_raw = rho_raw + gains["s"] * bold
        comm = torch.tanh(d / 0.7)
        if prior is not None:
            pw = (0.75 + 0.5 * prior.float()).clamp(0.5, 1.5)
        else:
            pw = 1.0
        raw = raw - gains["d"] * comm * w * pw
        if fragility is not None:
            _fw = fragility.float()
            while _fw.dim() < raw.dim():
                _fw = _fw.unsqueeze(2)
            # fragile regions (probe-measured high local error) take a gentler
            # step: positive raw -> positive m_off -> retain more noise, i.e.
            # trust the model less exactly where it is least reliable.  The
            # authority scales with the model's measured local error.
            _frag_gain = float(gains.get("frag_gain", _LF_FRAG_GAIN))
            raw = raw + _frag_gain * _fw.to(raw.dtype)
        raw = raw * gains["k"]
        if damp is not None:
            keep = 1.0 - damp.float()
            raw = raw * keep
            rho_raw = rho_raw * keep
        trust_t = None
        if trust is not None:
            trust_t = trust.float().unsqueeze(1).clamp(0.2, 1.0)
            if gate is not None:
                if torch.is_tensor(gate):
                    trust_t = trust_t * gate.to(trust_t.dtype)
                else:
                    trust_t = trust_t * trust_t.new_full((), float(gate))
            raw = raw * trust_t
            rho_raw = rho_raw * trust_t
        if yield_map is not None:
            _y = yield_map.float().clamp(0.0, 1.0)
            while _y.dim() < raw.dim():
                _y = _y.unsqueeze(2)
            _keep = (1.0 - _y).to(raw.dtype)
            raw = raw * _keep
            rho_raw = rho_raw * _keep
        rescue_term = None
        resc_rho = None
        hi_boost = None
        resc_out = None
        _bound = amp_eff
        if (rescue is not None and rescue.get("on")
                and min(x0.shape[-2:]) >= int(rescue.get("block", 4))):
            _armed = kn >= float(rescue["start"])
            _wr = (_lf_rescue_window(kn, float(rescue["start"]),
                                     float(rescue["full"])) if _armed else 0.0)
            _st = resc_state if isinstance(resc_state, dict) else None
            if _armed and jmap is not None:
                _blk = int(rescue.get("block", 4))
                lf_share, tot_blk = _lf_block_delta_bands(jmap, _blk)
                blk_log, _ = _lf_block_scale(jmap, _blk)
                d_blk = F.avg_pool2d(d.reshape(-1, 1, *d.shape[-2:]), _blk)
                n_bt = blk_log.shape[0]
                ref_tot = torch.quantile(tot_blk.reshape(n_bt, -1).float(),
                                         0.10, dim=1).view(-1, 1, 1, 1)
                floor_mag = (float(rescue.get("floor", 0.02))
                             * float(rescue.get("s", 0.0) or 0.0))
                ref = _st.get("ref") if _st else None
                if ref is None or ref.shape != blk_log.shape:
                    ref = blk_log.clone()
                else:
                    ref = torch.maximum(blk_log, ref - 0.2231)  # log(1.25)
                # The rescue targets blocks that are still moving and where the
                # model is predicting real detail (high-frequency x0 content,
                # the d_blk term).  The old "structure static" gate (lf_share
                # < 0.5) measured the BAND OF THE x0 CHANGE, which is ~95%
                # low-frequency for these models (the structure is still
                # forming late, not static), so it blocked ~99% of blocks and
                # kept the rescue permanently off.  Drop it: d_blk already
                # verifies "the model predicts detail", and the still-moving
                # + coverage gates bound the region.
                hit = ((tot_blk > 1.5 * ref_tot)
                       & (tot_blk.sqrt() > floor_mag)
                       & ((blk_log - ref) > -0.6931)   # >= half the late peak
                       & (d_blk > (0.1 * (1.0 + 0.7 * eg_n))))
                frac = hit.float().mean(dim=(1, 2, 3), keepdim=True)
                g_cov = ((float(rescue.get("max_cov", 0.35)) - frac)
                         / 0.10).clamp(0.0, 1.0)
                need = 2.0 if rescue.get("fast") else 3.0
                credit = _st.get("credit") if _st else None
                if credit is None or credit.shape != hit.shape:
                    credit = torch.zeros_like(hit, dtype=torch.float32)
                credit = (credit + torch.where(hit, 1.0 / need,
                                               -0.5).float()).clamp(0.0, 1.0)
                _st_out = {"ref": ref, "credit": credit}
                if _wr > 0.0:
                    rw_blk = (((credit - 0.5) / 0.25).clamp(0.0, 1.0) * g_cov)
                    rw4 = F.interpolate(rw_blk, size=jmap.shape[-2:],
                                        mode="bilinear", align_corners=False)
                    if rw4.shape[0] != jmap.shape[0]:
                        rw = rw4.reshape(jmap.shape[0], jmap.shape[1],
                                         *jmap.shape[-2:]).unsqueeze(1)
                    else:
                        rw = rw4
                    while rw.dim() < raw.dim():
                        rw = rw.unsqueeze(2)
                    rw = rw.float()
                    if damp is not None:
                        rw = rw * (1.0 - damp.float())
                    if trust_t is not None:
                        rw = rw * ((trust_t - 0.4) / 0.6).clamp(0.0, 1.0)
                    rw = rw.clamp(0.0, 1.0)
                    _cov = float((rw > 0.05).float().mean())
                    resc_out = {"state": _st_out, "rw": None, "cov": _cov,
                                "hit": round(float(frac.mean()), 4),
                                "lf_share": round(float(lf_share.mean()), 4)}
                    if float(rw.max()) > 1e-4:
                        rescue_term = -(float(rescue["amp"]) * _wr) * rw * comm * pw
                        resc_rho = (float(rescue.get("ramp", 0.0)) * _wr) * rw
                        hi_boost = (_wr * rw).clamp(0.0, 1.0)
                        resc_out["rw"] = hi_boost
                else:
                    resc_out = {"state": _st_out, "rw": None, "cov": 0.0,
                                "hit": round(float(frac.mean()), 4),
                                "lf_share": round(float(lf_share.mean()), 4)}
        m_off = amp_eff * torch.tanh(raw)
        if rescue_term is not None:
            m_off = m_off + rescue_term
            _bound = amp_eff + float(rescue["amp"])
        dims = tuple(range(1, m_off.dim()))
        if step_d is not None:
            dlt = step_d.to(m_off.dtype)
            wgt = dlt.float().pow(2).mean(dim=1, keepdim=True).to(m_off.dtype)
            if _LF_CENTERING == "v4_proj":
                wsum = (wgt * wgt).sum(dim=dims, keepdim=True).clamp_min(1e-12)
                m_off = m_off - ((m_off * wgt).sum(dim=dims, keepdim=True)
                                / wsum) * wgt
            else:
                _wbar = ((m_off * wgt).sum(dim=dims, keepdim=True)
                         / wgt.sum(dim=dims, keepdim=True).clamp_min(1e-12))
                m_off = m_off - _wbar
            if gdir is not None:
                g_map = (dlt.float() * gdir.float()) \
                    .sum(dim=1, keepdim=True).to(m_off.dtype)
                gsum = (g_map * g_map).sum(dim=dims,
                                           keepdim=True).clamp_min(1e-12)
                m_off = m_off - ((m_off * g_map).sum(dim=dims, keepdim=True)
                                 / gsum) * g_map
        else:
            m_off = m_off - m_off.mean(dim=dims, keepdim=True)
        _mx_t = m_off.abs().max()
        _scale = (_bound / _mx_t.clamp_min(1e-8)).clamp(max=1.0)
        m_off = m_off * _scale
        cum_out = None
        _drift_B = float(gains.get("drift", 0.0) or 0.0)
        if _drift_B > 1e-6:
            _c4 = (cum.float().unsqueeze(1) if cum is not None
                   else torch.zeros_like(m_off))
            if _c4.shape == m_off.shape:
                _B_soft = 0.5 * _drift_B
                _room = ((_drift_B - _c4.abs())
                         / max(_drift_B - _B_soft, 1e-6)).clamp(0.0, 1.0)
                _same = (m_off * _c4) > 0
                m_off = m_off * torch.where(_same, _room, 1.0)
                m_off = m_off - _LF_DRIFT_RELAX * _c4
                m_off = (_c4 + m_off).clamp(-_drift_B, _drift_B) - _c4
                cum_out = (_c4 + m_off).squeeze(1)
        rho = ramp_eff * torch.tanh(rho_raw * gains["k"])
        if resc_rho is not None:
            rho = (rho + resc_rho).clamp(0.0, 0.95)
    return m_off, rho, vol_new, med_ref, resc_out, cum_out

def _lf_trust_update(emap, emap_prev, m_prev, amp_prev, trust):
    with torch.no_grad():
        eff = None
        n_st = None
        if emap_prev is not None and m_prev is not None:
            lr = ((emap + 1e-8).log() - (emap_prev + 1e-8).log())
            lr = torch.nan_to_num(lr, nan=0.0, posinf=4.0, neginf=-4.0)
            _b = lr.shape[0]
            _med = lr.reshape(_b, -1).median(dim=1).values.view(
                _b, *([1] * (lr.dim() - 1)))
            lr_c = lr - _med
            steered = m_prev.abs() > (0.3 * amp_prev)
            bad = steered & (lr_c > 0.35)
            good = steered & (lr_c < -0.15)
            t = trust if trust is not None else torch.ones_like(lr_c)
            t = torch.where(bad, t * 0.55,
                            torch.where(good, (t * 1.15).clamp(max=1.0), t))
            trust = (t + 0.02).clamp(0.2, 1.0)
            n_f = steered.float().sum()
            eff = (good.float().sum() - bad.float().sum()) / n_f.clamp_min(1.0)
            n_st = n_f.detach().clone()
    return trust, emap.detach(), eff, n_st

def _lf_refresh(c, rho, surgical=False, ramp_cap=1.0, hi_boost=None, gen=None):
    with torch.no_grad():
        _rho_max = rho.max()
        _rho_active = _rho_max > 1e-4
        if not bool(_rho_active.item()):
            return c
        _hb = None
        if hi_boost is not None:
            _hb = hi_boost.to(c.dtype)
            if float(_hb.max()) <= 1e-4:
                _hb = None
        if surgical or _hb is not None:
            hi, mid, lo = _lf_band_split(c)

            def _one(band, f, scale):
                r2 = (f * f).clamp(0.0, 0.99)
                if gen is not None:
                    _noise = torch.randn(band.shape, device=band.device,
                                         dtype=band.dtype, generator=gen)
                else:
                    _fb_gen = torch.Generator(device=band.device)
                    _fb_gen.manual_seed(20260915)
                    _noise = torch.randn(band.shape, device=band.device,
                                         dtype=band.dtype, generator=_fb_gen)
                return torch.sqrt(1.0 - r2) * band + f * scale * _noise

            def _rms(band):
                dims = tuple(range(1, band.dim()))
                return band.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)

            def _pmag(band):
                return band.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)

            base_hi = 1.5 if surgical else 1.0
            base_lo = 0.5 if surgical else 1.0
            if _hb is not None:
                f_hi = (rho * base_hi * (1.0 + _hb)).clamp(0.0, 0.75)
                f_lo = (rho * base_lo * (1.0 - 0.5 * _hb)).clamp(0.0, 0.75)
                return (_one(hi, f_hi, _pmag(hi)) + _one(mid, rho, _rms(mid))
                        + _one(lo, f_lo, _pmag(lo))).to(c.dtype)
            _cap = float(ramp_cap)
            return (_one(hi, (rho * base_hi).clamp(max=_cap), _rms(hi))
                    + _one(mid, rho, _rms(mid))
                    + _one(lo, (rho * base_lo).clamp(max=_cap), _rms(lo))).to(c.dtype)
        dims = tuple(range(1, c.dim()))
        rms = c.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
        r2 = (rho * rho).clamp_max(0.99)
        if gen is not None:
            _noise = torch.randn(c.shape, device=c.device,
                                 dtype=c.dtype, generator=gen)
        else:
            _fb_gen = torch.Generator(device=c.device)
            _fb_gen.manual_seed(20260915)
            _noise = torch.randn(c.shape, device=c.device,
                                 dtype=c.dtype, generator=_fb_gen)
        return torch.sqrt(1.0 - r2) * c + rho * rms * _noise


def _pchip_slopes(xs, ys):
    n = len(xs)
    if n == 1:
        return [0.0]
    if n == 2:
        d = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-12)
        return [d, d]
    hs = [xs[i + 1] - xs[i] for i in range(n - 1)]
    ds = [(ys[i + 1] - ys[i]) / max(hs[i], 1e-12) for i in range(n - 1)]
    ms = [0.0] * n
    for i in range(1, n - 1):
        if ds[i - 1] * ds[i] <= 0.0:
            ms[i] = 0.0
        else:
            w_left  = 2.0 * hs[i] + hs[i - 1]   # weight for ds[i-1] (LEFT slope)
            w_right = hs[i] + 2.0 * hs[i - 1]   # weight for ds[i]   (RIGHT slope)
            ms[i] = (w_left + w_right) / (w_left / ds[i - 1] + w_right / ds[i])
    ms[0] = ((2.0 * hs[0] + hs[1]) * ds[0] - hs[0] * ds[1]) / (hs[0] + hs[1])
    if ms[0] * ds[0] <= 0.0:
        ms[0] = 0.0
    elif ds[0] * ds[1] < 0.0 and abs(ms[0]) > abs(3.0 * ds[0]):
        ms[0] = 3.0 * ds[0]
    ms[-1] = ((2.0 * hs[-1] + hs[-2]) * ds[-1] - hs[-1] * ds[-2]) / (hs[-1] + hs[-2])
    if ms[-1] * ds[-1] <= 0.0:
        ms[-1] = 0.0
    elif ds[-2] * ds[-1] < 0.0 and abs(ms[-1]) > abs(3.0 * ds[-1]):
        ms[-1] = 3.0 * ds[-1]
    return ms

def _pchip_eval(x0, x1, y0, y1, m0, m1, t):
    h = x1 - x0
    if h < 1e-12:
        return y0
    s = (t - x0) / h
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return (h00 * y0 + h10 * m0 * h + h01 * y1 + h11 * m1 * h)

def _pchip_interp(curve, xs, grid):
    n = len(curve)
    if n == 0:
        return []
    if n == 1:
        return [float(curve[0]) for _ in grid]
    if n == 2:
        ys = [float(curve[0]), float(curve[-1])]
        x_lo, x_hi = xs[0], xs[-1]
        out = []
        for t in grid:
            t = min(max(float(t), 0.0), 1.0)
            if t <= x_lo:
                out.append(ys[0]); continue
            if t >= x_hi:
                out.append(ys[1]); continue
            a = (t - x_lo) / max(x_hi - x_lo, 1e-9)
            out.append(ys[0] * (1.0 - a) + ys[1] * a)
        return out
    ys = [float(v) for v in curve]
    ms = _pchip_slopes(xs, ys)
    out = []
    x_lo, x_hi = xs[0], xs[-1]
    y_lo, y_hi = ys[0], ys[-1]
    j = 0
    for t in grid:
        t = min(max(float(t), 0.0), 1.0)
        if t <= x_lo:
            out.append(y_lo); continue
        if t >= x_hi:
            out.append(y_hi); continue
        while j + 1 < n and xs[j + 1] < t:
            j += 1
        out.append(_pchip_eval(xs[j], xs[j + 1], ys[j], ys[j + 1],
                              ms[j], ms[j + 1], t))
    return out

def _interp_curve_x(curve, xs, grid):
    return _pchip_interp(curve, xs, grid)


def _real_step_positions(sigmas):
    try:
        raw = [float(v) for v in (sigmas if sigmas is not None else [])]
        rv = [v for v in raw if v > 1e-6]
    except Exception:
        return None
    if len(rv) < 2:
        return None
    lo = math.log(max(rv[-1], 1e-8))
    hi = math.log(max(rv[0], 1e-8))
    if hi - lo < 1e-9:
        return None
    out = []
    for v in raw[1:]:
        if v <= 1e-6:
            out.append(1.0)
        else:
            out.append(min(max((hi - math.log(v)) / (hi - lo), 0.0), 1.0))
    return out


def _prof_step_curve(prof, slope, band):
    if not isinstance(prof, dict):
        return None, None
    if band:
        c = prof.get("band_curve") or []
    else:
        c = prof.get("slope_curve") or []
        if not c:
            jc = prof.get("jump_curve") or []
            gc = prof.get("gap_curve") or []
            if jc and len(gc) == len(jc):
                c = [j / max(g, 1e-9) for j, g in zip(jc, gc)]
            else:
                c = jc
    if not c:
        return None, None
    sc = prof.get("sigma_curve") or []
    pos = None
    if len(sc) == len(c) + 1:
        try:
            hi = math.log(max(float(sc[0]), 1e-8))
            lo = math.log(max(float(sc[-1]), 1e-8))
            if hi - lo > 1e-9:
                pos = [min(max((hi - math.log(max(float(v), 1e-8)))
                               / (hi - lo), 0.0), 1.0) for v in sc[1:]]
        except Exception:
            pos = None
    return c, pos


def _build_step_curve(prof_m, prof_c, w, n_steps, sigmas, slope=False,
                      band=False):
    cm, pm = _prof_step_curve(prof_m, slope, band)
    cc, pc = _prof_step_curve(prof_c, slope, band)
    if not cm and not cc:
        return None
    grid = _real_step_positions(sigmas)
    if not grid or len(grid) < 3:
        n_out = max(8, int(n_steps))
        grid = [(i + 1.0) / n_out for i in range(n_out)]
    n_out = len(grid)

    def _resample(curve, pos):
        if not curve:
            return None
        if pos and len(pos) == len(curve):
            xs = pos
        else:
            n = len(curve)
            xs = [(i + 0.5) / n for i in range(n)]
        if isinstance(curve[0], (list, tuple)):
            k = len(curve[0])
            return [_interp_curve_x([float(r[i]) for r in curve], xs, grid)
                    for i in range(k)]
        return _interp_curve_x(curve, xs, grid)

    rm = _resample(cm, pm)
    rc = _resample(cc, pc)
    if rm and rc:
        if isinstance(rm[0], (list, tuple)):
            out = [[(1.0 - w) * a + w * b for a, b in zip(mr, rr)]
                   for mr, rr in zip(rm, rc)]
        else:
            out = [(1.0 - w) * a + w * b for a, b in zip(rm, rc)]
    else:
        out = rm if rm else rc
    if not out:
        return None
    if band:
        if isinstance(out[0], (list, tuple)):
            return [round(float(x), 4) for x in out[0]]
        return [round(float(x), 4) for x in out]
    if len(out) < 4:
        return None
    sm3 = [sum(out[max(0, i - 1):i + 2]) / len(out[max(0, i - 1):i + 2])
           for i in range(len(out))]
    if any(x <= 1e-12 for x in sm3):
        return None
    gm = math.exp(sum(math.log(x) for x in sm3) / len(sm3))
    raw = [math.sqrt(x / gm) for x in sm3]
    qs = sorted(raw)
    q_lo = qs[min(len(qs) - 1, int(0.10 * len(qs)))]
    q_hi = qs[min(len(qs) - 1, int(0.90 * len(qs)))]
    spread = max(q_hi / max(q_lo, 1e-6), 1.0)
    rail_lo = max(0.45, 1.0 / spread)
    rail_hi = min(2.20, spread)
    sched = [min(max(r, min(rail_lo, rail_hi)), max(rail_lo, rail_hi))
             for r in raw]
    return [round(sched[i], 3) for i in range(len(sched))]


def _lf_drift_budget(is_distilled):
    return 0.06 if bool(is_distilled) else 0.10


def _probe_degenerate(prof):
    """True when a probe profile carries no usable dynamics evidence: a
    few-step/distilled model can converge in ~1 probe step, so its measured
    jump/curvature collapse to ~0 and averaging them into the blend only
    dilutes the meaningful per-prompt evidence."""
    if not isinstance(prof, dict):
        return True
    mj = float(prof.get("med_jump", 0.0) or 0.0)
    mc = float(prof.get("med_curv", 0.0) or 0.0)
    return mj < 1e-4 and mc < 1e-4


def _lf_auto_gains(prof_m, prof_c, cond_w, n_steps, sigmas=None, cfg=None):
    def _interp_curve(curve, grid):
        n = len(curve)
        if n == 0:
            return []
        if n == 1:
            return [float(curve[0]) for _ in grid]
        xs = [(i + 0.5) / n for i in range(n)]
        return _pchip_interp(curve, xs, grid)

    def _blend_curve(key, w, n_out):
        g = [(i + 1.0) / n_out for i in range(n_out)]

        def _get(prof):
            if not isinstance(prof, dict):
                return []
            c = prof.get(key) or []
            if c and isinstance(c[0], (list, tuple)):
                k = len(c[0])
                return [_interp_curve([float(r[i]) for r in c], g)
                        for i in range(k)]
            return _interp_curve(c, g)

        cm = _get(prof_m)
        cc = _get(prof_c)
        if cm and cc:
            if isinstance(cm[0], (list, tuple)):
                return [[(1.0 - w) * a + w * b for a, b in zip(rm, rc)]
                        for rm, rc in zip(cm, cc)]
            return [(1.0 - w) * a + w * b for a, b in zip(cm, cc)]
        return cm or cc

    def _prof_n(prof):
        jc = prof.get("jump_curve") or []
        n_jump = int(prof.get("n_jump") or len(jc) or 0)
        n_err = int(prof.get("n_err") or max(0, len(jc) - 1))
        n_guard = int(prof.get("n_guard") or len(jc) or 0)
        ns = [x for x in (n_jump, n_err, n_guard) if x > 0]
        return min(ns) if ns else 0

    def _prof_eg(prof):
        eg = prof.get("endgame")
        if not isinstance(eg, dict):
            return None
        vals = []
        for lv in (eg.get("levels") or []):
            for key in ("rec_q", "rec2_q", "rec_p50"):
                try:
                    v = lv.get(key)
                    if isinstance(v, (list, tuple)):
                        vals.append(float(v[1]))
                    else:
                        vals.append(float(v))
                except (TypeError, ValueError, IndexError, KeyError):
                    pass
        return max(vals) if vals else None

    def _prof_dir(prof):
        dc = prof.get("dir_cons_valid") or []
        if not dc:
            dc = prof.get("dir_cons") or []
        if not dc:
            return None
        late = dc[len(dc) // 3:] or dc
        return sum(late) / float(len(late))

    def _one(prof):
        ts = prof.get("trajectory_state")
        if ts is None and isinstance(prof, dict) and prof.get("jump_curve"):
            try:
                ts = _compute_trajectory_state(prof)
            except Exception:
                ts = None
        ts = ts or {}
        try:
            err = float(ts.get("med_local_err_real"))
        except (TypeError, ValueError):
            err = -1.0
        if not (err >= 0.0):
            err = float(prof.get("med_local_err", 0.0) or 0.0)
            gp = float(prof.get("log_gap_probe", 0.0) or 0.0)
            gr = float(prof.get("log_gap_real", 0.0) or 0.0)
            if err > 0 and gp > 1e-9 and gr > 1e-9:
                err = err * min(max((gr / gp) ** 2, 0.04), 1.0)
        sm = float(prof.get("guard_sig_med", 0.0) or 0.0)
        sp = float(prof.get("guard_sig_p90", 0.0) or 0.0)
        spike = (sp / sm) if sm > 1e-9 else 1.0
        return {"ov": min(max(float(ts.get("oversteer", 0.0) or 0.0), 0.0), 1.0),
                "osm": min(max(float(ts.get("oversmooth", 0.0) or 0.0), 0.0), 1.0),
                "err": max(err, 0.0),
                "spike": max(spike, 1.0),
                "dist": bool(prof.get("is_distilled_eff", False)),
                "n": _prof_n(prof),
                "eg": _prof_eg(prof),
                "dir": _prof_dir(prof)}

    sig_m = _one(prof_m) if isinstance(prof_m, dict) else None
    sig_c = _one(prof_c) if isinstance(prof_c, dict) else None
    if sig_m is not None and sig_c is not None:
        w = min(max(float(cond_w), 0.0), 1.0)
        # Degenerate model probe -> trust the per-prompt probe (see
        # _probe_degenerate) instead of averaging in ~0 evidence.
        if _probe_degenerate(prof_m):
            w = 1.0
        sig = {key: (1.0 - w) * sig_m[key] + w * sig_c[key]
               for key in ("ov", "osm", "err", "spike")}
        sig["dist"] = sig_m["dist"] or sig_c["dist"]
        sig["n"] = (1.0 - w) * sig_m["n"] + w * sig_c["n"]
        eg_m, eg_c = sig_m["eg"], sig_c["eg"]
        if eg_m is not None and eg_c is not None:
            sig["eg"] = (1.0 - w) * eg_m + w * eg_c
        else:
            sig["eg"] = eg_m if eg_m is not None else eg_c
        dir_m, dir_c = sig_m["dir"], sig_c["dir"]
        if dir_m is not None and dir_c is not None:
            sig["dir"] = (1.0 - w) * dir_m + w * dir_c
        else:
            sig["dir"] = dir_m if dir_m is not None else dir_c
        src = "model+cond(w=%.2f)" % w
    elif sig_m is not None:
        sig, src = sig_m, "model"
    elif sig_c is not None:
        sig, src = sig_c, "cond"
    else:
        ev = {"source": "none", "ov": 0.0, "osm": 0.0, "err": 0.0,
              "spike": 1.0, "dist": False, "n_steps": int(n_steps)}
        return None, ev
    ov, osm = sig["ov"], sig["osm"]
    err, spike = sig["err"], sig["spike"]
    dir_late = sig.get("dir")
    dir_real = None
    if cfg is None or bool(cfg.get("lf_dir_real_feedback", False)):
        if isinstance(prof_c, dict):
            _dr = prof_c.get("dir_real")
            if isinstance(_dr, list) and _dr:
                dir_real = _dr
        if dir_real is None and isinstance(prof_m, dict):
            _dr = prof_m.get("dir_real")
            if isinstance(_dr, list) and _dr:
                dir_real = _dr
    if dir_real is not None:
        _use = dir_real[-3:]
        dir_late = sum(float(x) for x in _use) / len(_use)
        dir_src = "run"
    else:
        dir_src = "probe"
    a4_bump = 0.0
    if dir_late is not None:
        _ratio = min(1.0, max(0.0, (0.2 - dir_late) / 0.2))
        if _ratio > 0.0:
            # Direction consistency (a mean cosine) is scale-invariant: it
            # directly flags a late-stage x0 oscillation and should always
            # raise the anti-artifact authority.  The old gap-ratio gate here
            # conflated the magnitude gap with the direction and zeroed this
            # signal for probes whose log_gap_probe/log_gap_real > 2.
            a4_bump = 0.15 * _ratio
            ov = min(1.0, ov + a4_bump)
    ev = max(ov, osm)
    few = min(max((9.0 - float(n_steps)) / 6.0, 0.0), 1.0)
    err_n = 1.0 - math.exp(-err / 0.03)
    # same log mapping as s_ovs_spike: the spike ratio spans orders of
    # magnitude, so an exponential saturates and loses all resolution.
    spk_n = min(1.0, max(0.0, math.log(max(spike, 1.0)) / math.log(30.0)))
    k = min(max(0.55 + 0.95 * ev, 0.50), 1.50)
    v = min(max(0.40 + 1.30 * ov, 0.30), 1.70)
    s = min(max(0.25 + 1.25 * osm, 0.20), 1.50)
    d = min(max((0.75 + 0.75 * few) * (1.0 - 0.45 * ov), 0.30), 1.60)
    clamp = min(max(0.38 * (1.0 - 0.45 * err_n) * (1.0 - 0.30 * spk_n)
                    + 0.10 * ev, 0.10), 0.48)
    eg_err = sig.get("eg")
    eg_n = 0.0
    if eg_err is not None and eg_err > 0.0:
        eg_n = 1.0 - math.exp(-eg_err / 0.06)
        clamp = clamp * (1.0 - 0.35 * eg_n)
        v = v * min(1.3, 1.0 + 0.30 * eg_n)
    v = min(v, _LF_V_MAX)
    n_eff = float(sig.get("n") or 0.0)
    conf = math.sqrt(n_eff / (n_eff + 2.0)) if n_eff > 0 else 1.0
    k = k * conf
    clamp = clamp * conf
    k = min(max(k, 0.50), 1.50)
    clamp = min(max(clamp, 0.10), 0.48)
    v_sched = None
    try:
        v_sched = _build_step_curve(
            prof_m, prof_c,
            (min(max(float(cond_w), 0.0), 1.0)
             if (sig_m is not None and sig_c is not None) else 0.0),
            n_steps, sigmas, slope=True)
    except Exception:
        v_sched = None
    hi_ref = None
    try:
        hi_ref = _build_step_curve(
            prof_m, prof_c,
            (min(max(float(cond_w), 0.0), 1.0)
             if (sig_m is not None and sig_c is not None) else 0.0),
            n_steps, sigmas, slope=False, band=True)
    except Exception:
        hi_ref = None
    gains = {"k": round(k, 3), "v": round(v, 3), "d": round(d, 3),
             "s": round(s, 3), "clamp": round(clamp, 3),
             "v_sched": v_sched, "hi_ref": hi_ref, "env_fade": True,
             "drift": round(_lf_drift_budget(sig["dist"]), 3)}
    evidence = {"source": src, "ov": round(ov, 3), "osm": round(osm, 3),
                "err": round(err, 5), "spike": round(spike, 2),
                "dist": sig["dist"], "n_steps": int(n_steps),
                "n": (round(n_eff, 1) if n_eff > 0 else None),
                "conf": round(conf, 3),
                "eg": (round(float(eg_err), 5) if eg_err is not None else None),
                "eg_n": (round(eg_n, 3) if eg_err is not None else None),
                "dir": (round(float(dir_late), 3) if dir_late is not None
                        else None),
                "dir_src": dir_src,
                "bump": round(a4_bump, 3),
                "drift": gains["drift"],
                "v_sched": v_sched is not None,
                "band_ref": hi_ref is not None}
    return gains, evidence

def _compute_snr(s, space, sigma_data):
    sd = float(sigma_data) if sigma_data is not None else 1.0
    if space == "flow":
        return max((1.0 - float(s)) / max(float(s), 1e-6), 1e-6)
    return sd * sd / max(float(s) * float(s), 1e-8)

def _consistency_step(s, s2, space):
    """Step size (in the space the dynamics actually live in) used to normalize
    the per-step x0 change into a derivative for the consistency measure.

    Flow dynamics evolve in DDIM time t = s/(1-s) (that is the variable the
    ancestral split already works in); measuring the x0 change against a
    log-sigma step collapses the step to ~0 near s=1 on a shift-bunched flow
    schedule, which made the consistency signal inert.  VE dynamics evolve in
    log-sigma, so the plain log step is correct there.
    """
    s = float(s)
    s2 = float(s2)
    if space == "flow":
        tf = s / max(1.0 - s, 1e-6)
        tt = s2 / max(1.0 - s2, 1e-6)
        return math.log(max(tf, 1e-8)) - math.log(max(tt, 1e-8))
    return math.log(max(s, 1e-8)) - math.log(max(s2, 1e-8))


def _compute_consistency(derivative_norm, x0_norm, step_size):
    if x0_norm < 1e-8:
        return 0.0
    step = max(float(step_size), 1e-8)
    # derivative_norm is the raw per-step x0 change; divide by the step to get
    # a true derivative before comparing against the x0 magnitude.  The old
    # code *multiplied* by step_size, which drove the term to 0 (consistency
    # -> 1, i.e. "always stable") on flow models with tiny log-sigma steps.
    return 1.0 / (1.0 + float(derivative_norm) / (float(x0_norm) * step))

def _aflops_step(x, x0, hist, s, sn, order=2, lam_min=None, lam_max=None):
    if len(hist) < 1:
        return None, None
    du = math.log(max(float(s), 1e-8)) - math.log(max(float(sn), 1e-8))
    if du <= 0.0:
        return None, None
    x_prev = hist[-1][3]
    x0_prev = hist[-1][2]
    s_prev_log = hist[-1][0]
    v_n = x0 - x
    v_prev = x0_prev - x_prev
    delta_v = (v_n - v_prev).flatten().float()
    delta_x = (x - x_prev).flatten().float()
    dx_norm_sq = float(delta_x.dot(delta_x))
    if dx_norm_sq < 1e-12:
        return None, None
    lam = float(delta_v.dot(delta_x) / dx_norm_sq)
    _lo = _AFLOPS_LAM_MIN if lam_min is None else float(lam_min)
    _hi = _AFLOPS_LAM_MAX if lam_max is None else float(lam_max)
    lam = max(_lo, min(_hi, lam))
    z = lam * du
    az = abs(z)
    if az > 10.0:
        return None, None
    h_n = x0 - (1.0 + lam) * x
    if az < 1e-4:
        exp_z    = 1.0 + z * (1.0 + z * (0.5 + z / 6.0))
        phi1_du  = du * (1.0 + z * (0.5 + z / 6.0))
        phi2_du2 = du * du * (0.5 + z / 6.0)
    else:
        exp_z    = math.exp(z)
        em1      = math.expm1(z)
        phi1_du  = em1 / lam
        phi2_du2 = (em1 - z) / (lam * lam)
    x_new = exp_z * x + phi1_du * h_n
    if order >= 2:
        h_prev = x0_prev - (1.0 + lam) * x_prev
        u_log_cur = math.log(max(float(s), 1e-8))
        du_h = s_prev_log - u_log_cur
        if du_h > 1e-8:
            slope = (h_n - h_prev) / du_h
            x_new = x_new + phi2_du2 * slope
    return x_new, lam

# ---------------------------------------------------------------------------
# Model-internal cache isolation for probe calls
#
# Modern ComfyUI models increasingly keep per-run caches on the model
# instance (Qwen-Image-2.1's prefix/KV cache, Wan Animate pose-branch cache,
# ...).  Those caches are keyed by full-resolution sequence shapes: slots
# selected/created during a probe pass (tiny probe latents, synthetic
# trajectories) do not match -- and on some ComfyUI revisions cannot even be
# *compared* against -- the real run's sequences, which crashes or poisons
# the actual sampling that follows.  Probes therefore always run with the
# model's own caches switched OFF: {"device": "off"} is the explicit opt-out
# contract of ComfyUI's cache machinery itself (see e.g.
# QwenImage21Transformer2DModel.select_prefix_cache), cache entries that are
# not dicts are dropped outright, and the real sampling loop is untouched.
# ---------------------------------------------------------------------------
_CACHE_OPT_OUT = {"device": "off"}
_CACHE_ISOLATION_LOGGED = [False]


def _probe_model_options(extra_args):
    """Model options copy with model-internal caches disabled, for probe
    calls.  Returns None when there is nothing cache-shaped to isolate (the
    caller then keeps using the run's own model_options, zero overhead)."""
    mo = (extra_args or {}).get("model_options")
    if not isinstance(mo, dict):
        return None
    try:
        to = mo.get("transformer_options")
        if not isinstance(to, dict):
            return None
        cache_keys = [k for k in to.keys()
                      if isinstance(k, str) and "cache" in k.lower()]
        if not cache_keys:
            return None
        to2 = dict(to)
        off_keys = []
        for k in cache_keys:
            v = to2.get(k)
            if isinstance(v, dict):
                if v.get("device") == "off":
                    continue
                v2 = dict(v)
                v2["device"] = "off"
                to2[k] = v2
            else:
                to2.pop(k, None)
            off_keys.append(k)
        if not off_keys:
            return None
        mo2 = dict(mo)
        mo2["transformer_options"] = to2
        if not _CACHE_ISOLATION_LOGGED[0]:
            _CACHE_ISOLATION_LOGGED[0] = True
            logging.info("[A-FloPS-probe] model-internal cache options "
                         "disabled for probe calls (%s): probe-shaped "
                         "sequences must never read or fill the model's "
                         "own caches", ", ".join(sorted(off_keys)))
        return mo2
    except Exception:
        return None


# Instance-level cache isolation: several architectures auto-enable their
# per-run cache directly on the model instance at pre_run time (Qwen-Image-2.1
# arms its prefix cache whenever the sampler starts, no transformer_options
# entry required), so the options-level opt-out above is not sufficient for
# them.  Models of that family expose a `reset_prefix_cache(enabled)` switch
# on the diffusion model -- probes flip it off for the duration of a
# measurement and re-arm it afterwards.  The next real sampling run re-arms
# its own cache via the runtime's pre_run anyway, and a re-armed cache is
# transparent (it simply refills), so this can never change the run's
# numerical results -- only who is allowed to fill the slots.
_PROBE_CACHE_DISABLE_DEPTH = [0]


def _find_prefix_reset(model):
    """Locate the model instance's per-run cache switch (reset_prefix_cache),
    if the diffusion model exposes one.  Returns the callable or None."""
    try:
        patcher = _navigate_to_patcher(model)
        dm = None
        if patcher is not None:
            m = getattr(patcher, "model", None)
            dm = getattr(m, "diffusion_model", None) if m is not None else None
        reset = getattr(dm, "reset_prefix_cache", None) if dm is not None \
            else None
        return reset if callable(reset) else None
    except Exception:
        return None


def _probe_model_cache_guard(model, disable=True):
    """Flip the model instance's own per-run cache off (disable=True) around
    probe measurements, re-arming it on disable=False.  No-op for models
    without such a switch (guarded by hasattr, exceptions swallowed)."""
    try:
        reset = _find_prefix_reset(model)
        if reset is None:
            return
        if disable:
            _PROBE_CACHE_DISABLE_DEPTH[0] += 1
            if _PROBE_CACHE_DISABLE_DEPTH[0] == 1:
                reset(False)
                if not _CACHE_ISOLATION_LOGGED[0]:
                    _CACHE_ISOLATION_LOGGED[0] = True
                    logging.info("[A-FloPS-probe] model-internal per-run "
                                 "cache disabled for the probe measurement "
                                 "(instance switch), re-armed afterwards")
        else:
            if _PROBE_CACHE_DISABLE_DEPTH[0] > 0:
                _PROBE_CACHE_DISABLE_DEPTH[0] -= 1
            if _PROBE_CACHE_DISABLE_DEPTH[0] == 0:
                reset(True)
    except Exception:
        pass


def _probe_model_call(model, x, sigma, extra_args):
    """Revision-agnostic model call for every probe measurement.

    Two contracts are handled here so the probes stay compatible with the
    ComfyUI revision that happens to be running (this is REVISION
    compatibility, not a per-model handler -- the model itself is still
    invoked exactly as the chain invokes it):

    1. denoise_mask placement.  Current ComfyUI requires it as a positional
       argument of KSamplerX0Inpaint.__call__ (older revisions took it
       through extra_args).  When extra_args does not carry it, the first
       call would fail with "missing 1 required positional argument:
       'denoise_mask'" -- so the call is retried with an explicit None.
    2. Probe trajectories are synthetic and UNMASKED.  A real denoise_mask
       tensor belongs to the user's latent; on current ComfyUI it feeds the
       inpaint math inside KSamplerX0Inpaint.__call__ and would not
       broadcast against the probe latent.  It is therefore always dropped
       (logged) for probe calls.

    Both probe latents and cache state are shape-private: the call rides
    the chain with the model's internal caches disabled (see
    _probe_model_options) so probing can never read or pollute them,
    whatever the architecture.
    """
    ea = dict(extra_args or {})
    _mopt = _probe_model_options(ea)
    if _mopt is not None:
        ea["model_options"] = _mopt
    dropped = ea.get("denoise_mask")
    if dropped is not None:
        ea["denoise_mask"] = None
        try:
            _shape = tuple(dropped.shape)
        except Exception:
            _shape = None
        logging.info("[A-FloPS-probe] probing unmasked: the run's "
                     "denoise_mask does not apply to the synthetic probe "
                     "latent (%s dropped)",
                     "shape %s" % (_shape,) if _shape is not None
                     else type(dropped).__name__)
    try:
        return model(x, sigma, **ea)
    except TypeError as e1:
        msg = str(e1)
        attempts = []
        if "denoise_mask" in msg:
            # Required positionally by this revision and absent from
            # extra_args (e.g. the node-side model probe, whose extra_args
            # start empty).
            attempts.append(lambda: model(x, sigma, None, **ea))
            attempts.append(lambda: model(x, sigma, None,
                                          model_options=ea.get(
                                              "model_options", {}),
                                          seed=ea.get("seed")))
        if "keyword" in msg:
            # The chain model only accepts the explicit kwargs of this
            # revision; retry with progressively less.
            attempts.append(lambda: model(x, sigma))
        for att in attempts:
            try:
                return att()
            except TypeError:
                continue
        raise e1

def _probe_noise_survival(model, extra_args, xs, x0s, grid, ones, gen=None):
    try:
        m = len(x0s)
        if m >= 6:
            probe_idxs = [m // 4, 3 * m // 4]
        elif m >= 3:
            probe_idxs = [m // 2]
        else:
            return 0.1
        safe = 0.35  # start permissive; min() tightens below
        for i in probe_idxs:
            if i + 1 >= m:
                i = max(0, m - 2)
            s_m = grid[i]
            sn_m = grid[i + 1]
            x_at = xs[i]
            x0_ref = x0s[i]
            x_next_clean = exact_step(x_at, x0_ref, s_m, sn_m)
            x0_next_clean = _probe_model_call(model, x_next_clean,
                                              sn_m * ones, extra_args)
            ref_norm = float(x0_next_clean.float().norm().clamp_min(1e-8))
            safe_i = 0.1
            for f in (0.2, 0.35):
                noise = torch.randn(x_at.shape, device=x_at.device,
                                    dtype=torch.float32,
                                    generator=gen).to(x_at.dtype)
                amp = math.sqrt(f) * float(s_m)
                xpert = x_at + noise * amp
                x0_pert = _probe_model_call(model, xpert, s_m * ones,
                                            extra_args)
                x_next_pert = exact_step(xpert, x0_pert, s_m, sn_m)
                x0_next_pert = _probe_model_call(model, x_next_pert,
                                                 sn_m * ones, extra_args)
                div = float((x0_next_pert - x0_next_clean).float().norm()) / ref_norm
                if div < 0.35:
                    safe_i = f
                else:
                    break
            safe = min(safe, safe_i)
        return max(safe, 0.1)
    except Exception:
        return 0.15

def _derive_calibrations(profile):
    calib = {}
    med_curv = profile.get("med_curv", 0.0)
    safe_f = profile.get("safe_fresh_frac", 0.15)
    med_err = float(profile.get("med_local_err", 0.0) or 0.0)
    gap_p = float(profile.get("log_gap_probe", 0.0) or 0.0)
    gap_r = float(profile.get("log_gap_real", 0.0) or 0.0)
    if med_err > 0:
        scale = 1.0
        if gap_p > 1e-9 and gap_r > 1e-9:
            scale = min(max((gap_r / gap_p) ** 2, 0.04), 1.0)
        calib["rtol"] = round(min(0.04, max(0.005, 3.0 * med_err * scale)), 4)
    sig_med = float(profile.get("guard_sig_med", 0.0) or 0.0)
    sig_p90 = float(profile.get("guard_sig_p90", 0.0) or 0.0)
    if sig_med > 1e-9:
        r = sig_p90 / sig_med
        calib["guard_floor"] = round(
            min(0.5, max(0.05, 0.15 * math.sqrt(max(r, 0.25) / 4.0))), 4)
    if med_curv > 0:
        lc = math.log(max(med_curv, 1e-8) + 1.0)
        if lc < math.log(1.5):  # med_curv < 0.5
            wb = 3.0
        elif lc > math.log(5.0):  # med_curv > 4.0
            wb = 1.8
        else:
            t = (lc - math.log(1.5)) / (math.log(5.0) - math.log(1.5))
            wb = 3.0 - t * (3.0 - 1.8)
        calib["wmax_base"] = round(max(1.5, min(2.5, wb)), 3)
    calib["safe_eta"] = round(min(1.0, max(0.1, math.sqrt(max(safe_f, 1e-4)))), 3)
    calib.update(_derive_anomaly_calibrations(profile))
    return calib

def _derive_anomaly_calibrations(profile):
    """Translate probe evidence into the anomaly-repair thresholds.

    The probe already measures everything the anomaly detector needs to know
    about the model's LEGITIMATE update statistics:

    - jump_q (per-step p10/p50/p90 of the normalized per-pixel jump; fallback
      guard_sig_med/p90) says how heavy-tailed the model's normal jump
      distribution is.  Heavy tails = large per-pixel jumps are structure,
      not anomalies, so the spatial/novelty z-thresholds must rise to avoid
      false flags; distributions tighter than a Gaussian reference can afford
      more sensitivity.
    - err_q (per-step p10/p50/p90 of the local error) tunes the value-error
      threshold the same way.
    - conv_step / n_jump say when the bulk of the trajectory work is done, so
      anomaly detection can arm relative to the measured convergence point
      instead of a fixed 15% of the run (0.15 default is reproduced exactly
      at conv_step = n_jump/3).
    - is_distilled_eff softens repair across the board (the promised
      behaviour of the probe_tune_anomaly toggle for distilled models).
    """
    calib = {}

    def _tail_ratio(quantile_curve):
        rs = []
        for q in (quantile_curve or []):
            try:
                p50, p90 = float(q[1]), float(q[2])
                if p50 > 1e-9:
                    rs.append(p90 / p50)
            except (TypeError, ValueError, IndexError):
                continue
        if not rs:
            return None
        return sorted(rs)[len(rs) // 2]

    spike = _tail_ratio(profile.get("jump_q"))
    if spike is None:
        sig_med = float(profile.get("guard_sig_med", 0.0) or 0.0)
        sig_p90 = float(profile.get("guard_sig_p90", 0.0) or 0.0)
        if sig_med > 1e-9:
            spike = sig_p90 / sig_med
    if spike is not None and spike > 0:
        # Gaussian reference: p90/p50 = 1.815.  Thresholds sit at 1.0x for a
        # reference-shaped distribution, rise up to +60% for very heavy
        # tails, drop to -25% for unusually tight ones.
        mult = 1.0 + 0.6 * (spike - 1.815) / 1.815
        mult = min(max(mult, 0.75), 1.60)
        if bool(profile.get("is_distilled_eff", False)):
            mult = min(mult * 1.2, 1.9)  # soften repair for distilled models
        calib["an_spatial"] = round(5.0 * mult, 3)
        calib["an_novelty"] = round(4.0 * mult, 3)

    err_spike = _tail_ratio(profile.get("err_q"))
    if err_spike is not None and err_spike > 0:
        vmult = 1.0 + 0.5 * (err_spike - 1.815) / 1.815
        vmult = min(max(vmult, 0.80), 1.50)
        if bool(profile.get("is_distilled_eff", False)):
            vmult = min(vmult * 1.1, 1.6)
        calib["an_value"] = round(2.5 * vmult, 3)

    try:
        conv_step = int(profile.get("conv_step", -1))
        n_jump = int(profile.get("n_jump", 0))
    except (TypeError, ValueError):
        conv_step, n_jump = -1, 0
    if 0 <= conv_step < n_jump:
        frac = conv_step / float(n_jump)
        calib["an_from"] = round(min(max(0.05 + 0.30 * frac, 0.05), 0.35), 3)
    return calib

def _probe_gen(x):
    gen = torch.Generator(device=x.device)
    gen.manual_seed(20260915)
    return gen

def _lf_renoise(x0, s, space, gen):
    eps = torch.randn(x0.shape, device=x0.device, dtype=torch.float32,
                      generator=gen).to(x0.dtype)
    if space == "flow":
        sf = min(max(float(s), 0.0), 0.999)
        return (1.0 - sf) * x0 + sf * eps
    return x0 + float(s) * eps

def _probe_endgame(model, extra_args, x0_final, grid_last, pos_all, ones,
                   space, gen):
    try:
        cands = [float(v) for v in pos_all if 1e-6 < float(v) < float(grid_last) * 0.9]
        if len(cands) >= 2:
            s_a = cands[max(0, int(round(len(cands) * 0.45)))]
            s_b = cands[min(len(cands) - 1, int(round(len(cands) * 0.85)))]
        else:
            s_a = float(grid_last) * 0.30
            s_b = float(grid_last) * 0.10
        if not (s_b < s_a):
            s_b = s_a * 0.4
        ref_rms = float(x0_final.float().pow(2).mean().sqrt().clamp_min(1e-8))

        def _qs(map_t):
            flat = map_t.reshape(-1).float()
            q = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9],
                                                  device=flat.device))
            return [round(float(v), 6) for v in q]

        levels = []
        for s_t in (s_a, s_b):
            x_t = _lf_renoise(x0_final, s_t, space, gen)
            x0_a = _probe_model_call(model, x_t,
                                     x_t.new_full([x_t.shape[0]], s_t),
                                     extra_args)
            s2_t = s_t * 0.5
            x1_t = exact_step(x_t, x0_a, s_t, s2_t)
            x0_b = _probe_model_call(model, x1_t,
                                     x1_t.new_full([x1_t.shape[0]], s2_t),
                                     extra_args)
            a_rms = float(x0_a.float().pow(2).mean().sqrt().clamp_min(1e-8))
            rec_a = (x0_a.float() - x0_final.float()).pow(2).mean(dim=1).sqrt() / ref_rms
            rec_b = (x0_b.float() - x0_final.float()).pow(2).mean(dim=1).sqrt() / ref_rms
            jmp = (x0_b.float() - x0_a.float()).pow(2).mean(dim=1).sqrt() / a_rms
            levels.append({"s": round(float(s_t), 6),
                           "s2": round(float(s2_t), 6),
                           "rec_q": _qs(rec_a), "rec2_q": _qs(rec_b),
                           "jmp_q": _qs(jmp)})
        return {"levels": levels, "n": int(rec_a.numel())}
    except Exception as e:
        logging.warning("[A-FloPS-probe] endgame measurement skipped: %s", e)
        return None


def _probe_distilled_mode(cfg):
    """Few-step (distilled) detection for probe-grid shaping, available before
    the probe has produced its own is_distilled_eff: guidance-free operation
    (live cfg <= 1) or an explicit distilled model hint."""
    try:
        pc = cfg.get("_guider_live_cfg")
        if pc is not None and float(pc) <= 1.0 + 1e-6:
            return True
    except (TypeError, ValueError):
        pass
    try:
        mi = cfg.get("_model_info")
        if isinstance(mi, dict) and mi.get("is_distilled"):
            return True
    except Exception:
        pass
    return False


_PROBE_SMIN_FLOOR = 0.10    # flow: floor the probe grid at 10% of sigma_max


def _probe_grid(sigmas, K, distilled_mode=False):
    """Build the probe trajectory's sigma grid, GEOMETRIC (log-uniform) in
    sigma.  An index-uniform subsample of a high-shift schedule collapses most
    points into a tiny high-sigma band, producing degenerate (~0) jumps and
    curvature that fool the is_distilled_eff heuristic (a cfg>1 model with a
    bunched schedule reads as "converged in 2 steps").  Log-uniform spacing
    keeps the probe points meaningfully spread regardless of the shift.  For
    few-step (distilled) models the top half of log-sigma gets half the points,
    since that is where all the dynamics happen.

    The low bound is FLOORED for flow schedules: the schedule's smallest
    positive sigma is shift-dependent and, on a high-shift flow schedule, can
    collapse toward ~0 (the logged run gave min(g)=0.0023).  Anchoring the
    geometric grid at that min(g) spans a huge log-range, the jump curve
    decays abruptly, and is_distilled_eff falsely flags a cfg>1 model as
    distilled.  Clamping at 10% of sigma_max keeps the probe inside the
    model's active sigma regime and makes the measurement comparable across
    schedules."""
    g = [float(s) for s in sigmas if float(s) > 1e-6]
    if len(g) < 3:
        return None
    if len(g) <= K:
        return sorted(g, reverse=True)
    smax = max(g)
    smin = min(g)
    if smax <= 1.5:
        # flow: floor the low bound (VE schedules have a natural, sane min(g)
        # and do not suffer the high-shift collapse).
        smin = max(smin, smax * _PROBE_SMIN_FLOOR)
    lmax = math.log(max(smax, 1e-8))
    lmin = math.log(max(smin, 1e-8))
    if distilled_mode:
        n_hi = max(2, K // 2)
        lsplit = 0.5 * (lmax + lmin)
        hi = [math.exp(lmax + (lsplit - lmax) * (i / max(n_hi - 1, 1)))
              for i in range(n_hi)]
        lo = [math.exp(lsplit + (lmin - lsplit) * (i / max(K - n_hi - 1, 1)))
              for i in range(K - n_hi)]
        grid = hi + lo
    else:
        grid = [math.exp(lmax + (lmin - lmax) * (i / (K - 1))) for i in range(K)]
    grid = sorted(set(round(float(v), 8) for v in grid), reverse=True)
    return grid if len(grid) >= 3 else None


def _fragility_map(frag_acc, size=8):
    """Downsample the accumulated per-pixel max local error into a coarse
    [0,1] fragility map (which regions the linear velocity model predicts
    least well -- the model's "weak spots").  Feeds the local field's spatial
    prior so it can steer volatility/dwell toward exactly those regions."""
    if frag_acc is None:
        return None
    try:
        fm = frag_acc.float().reshape(-1, 1, *frag_acc.shape[-2:])
        fm = F.adaptive_avg_pool2d(fm, (int(size), int(size)))
        fm = fm.mean(dim=0, keepdim=True)
        mx = float(fm.max().clamp_min(1e-8))
        fm = (fm / mx).clamp(0.0, 1.0)
        return {"h": int(size), "w": int(size),
                "v": [round(float(v), 4) for v in fm.reshape(-1)]}
    except Exception:
        return None


def _run_probe(model, x, sigmas, extra_args, cfg, callback=None):
    try:
        pres = max(16, int(cfg.get("probe_resolution", 48)))
        K = max(4, int(cfg.get("probe_steps", 10)))
        B = 1
        C = x.shape[1]
        if B < 1 or C < 1:
            return None
        shape = _probe_latent_shape(ref_x=x, pres=pres, batch=B)
        px = torch.randn(shape, device=x.device, dtype=torch.float32,
                         generator=_probe_gen(x)).to(x.dtype)
        ones = px.new_ones([B])
        grid = _probe_grid(sigmas, K, _probe_distilled_mode(cfg))
        if grid is None:
            return None
        px = px * float(grid[0])
        pos_all = [float(v) for v in sigmas if float(v) > 1e-6]
        gap_curve = [abs(math.log(max(grid[i], 1e-8))
                         - math.log(max(grid[i + 1], 1e-8)))
                     for i in range(len(grid) - 1)]
        real_gap_curve = []
        if len(pos_all) >= 2:
            pos_idx = []
            _j = 0
            for gv in grid:
                while _j < len(pos_all) and pos_all[_j] > gv * (1.0 + 1e-9):
                    _j += 1
                pos_idx.append(min(_j, len(pos_all) - 1))
            real_gaps = [abs(math.log(max(pos_all[t], 1e-8))
                             - math.log(max(pos_all[t + 1], 1e-8)))
                         for t in range(len(pos_all) - 1)]
            for i in range(len(grid) - 1):
                a, b = pos_idx[i], pos_idx[i + 1]
                if b <= a:
                    lo, hi = max(a - 1, 0), min(a + 1, len(pos_all) - 1)
                else:
                    lo, hi = a, b
                seg = real_gaps[lo:hi]
                real_gap_curve.append(sum(seg) / len(seg) if seg
                                      else (sum(real_gaps) / len(real_gaps)))
        else:
            real_gap_curve = list(gap_curve)
        space = _detect_space(torch.tensor(grid), cfg.get("noise_space", "auto"))
        xs = []
        x0s = []
        cur = px
        _dev_retry_done = False
        for i, s in enumerate(grid):
            _progress(callback, "probe", "progress", i, len(grid) + 4,
                      "A-FloPS model probe: step %d/%d" % (i + 1, len(grid)))
            x0 = None
            try:
                x0 = _probe_model_call(model, cur, s * ones, extra_args)
            except Exception as e:
                _pp = None if _dev_retry_done else _navigate_to_patcher(model)
                if (not _dev_retry_done and _pp is not None
                        and _is_device_error(e)):
                    _dev_retry_done = True
                    try:
                        _dev = _probe_call_device(_pp)
                        if cur.device != _dev:
                            cur = cur.to(_dev)
                        if ones.device != _dev:
                            ones = ones.to(_dev)
                        x0 = _probe_model_call(model, cur, s * ones, extra_args)
                        logging.info("[A-FloPS-probe] recovered from a "
                                     "device mismatch at s=%.4g (model "
                                     "re-secured on %s)", s, _dev)
                    except Exception:
                        x0 = None
                if x0 is None:
                    import traceback as _tb
                    logging.warning("[A-FloPS-probe] model call failed at s=%.4g: %s", s, e)
                    logging.warning("[A-FloPS-probe] failure traceback:\n%s", _tb.format_exc())
                    logging.warning("[A-FloPS-probe] failure context: %s",
                                    _probe_failure_diagnostics(model, cur, ones, extra_args, None))
                    break
            xs.append(cur)
            x0s.append(x0.detach())
            if i < len(grid) - 1:
                cur = exact_step(cur, x0, s, grid[i + 1])
        m = len(x0s)
        if m < 3:
            return None
        norms = [float(t.float().norm().clamp_min(1e-8)) for t in x0s]
        jumps = []
        for i in range(m - 1):
            d = float((x0s[i + 1] - x0s[i]).float().norm())
            jumps.append(d / max(max(norms[i], norms[i + 1]), 1e-8))
        dist_on = bool(cfg.get("probe_distributions", True))
        jump_q = []
        ojf_curve = []
        band_curve = []
        jrel_curve = []
        dabs_curve = []
        if dist_on:
            _qt = torch.tensor([0.1, 0.5, 0.9])
            for i in range(m - 1):
                jpx = (x0s[i + 1] - x0s[i]).float().pow(2).mean(dim=1).sqrt()
                sc = max(float(x0s[i].float().pow(2).mean().sqrt()),
                         float(x0s[i + 1].float().pow(2).mean().sqrt()))
                jpx_n = (jpx / max(sc, 1e-8)).reshape(-1)
                qs = torch.quantile(jpx_n, _qt.to(jpx_n.device))
                jump_q.append([round(float(v), 6) for v in qs])
                p50 = float(qs[1])
                if p50 > 1e-9:
                    ojf_curve.append(round(
                        float((jpx_n > 2.5 * p50).float().mean()), 4))
                try:
                    band_curve.append([round(v, 4) for v in
                                       _lf_band_fractions(jpx)])
                except Exception:
                    band_curve.append([0.33, 0.34, 0.33])
                # runtime-scale per-pixel relative jump: the SAME quantity the
                # escape's stagnation detector uses (median jump / median
                # per-pixel magnitude).  Stored so the escape threshold can be
                # derived from the model's own measured jump level.
                xm = x0s[i].float().pow(2).mean(dim=1).sqrt()
                jrel_curve.append(round(
                    float(jpx.median().clamp_min(0.0)
                          / xm.median().clamp_min(1e-8)), 6))
            for i in range(m):
                x4 = x0s[i].float().reshape(-1, x0s[i].shape[1],
                                            *x0s[i].shape[-2:])
                hp = x4 - F.avg_pool2d(x4, 3, stride=1, padding=1)
                dabs_curve.append(round(
                    float(hp.norm() / x4.norm().clamp_min(1e-8)), 6))
        med_jump_rel = (sorted(jrel_curve)[len(jrel_curve) // 2]
                        if jrel_curve else None)
        med_detail_abs = (sorted(dabs_curve)[len(dabs_curve) // 2]
                          if dabs_curve else None)
        dir_cons = []
        if bool(cfg.get("probe_dir_cons", True)):
            for i in range(1, m - 1):
                d1 = (x0s[i] - x0s[i - 1]).float()
                d2 = (x0s[i + 1] - x0s[i]).float()
                num = (d1 * d2).sum(dim=1)
                den = d1.pow(2).sum(dim=1).sqrt() * d2.pow(2).sum(dim=1).sqrt()
                okm = den > 1e-8
                if bool(okm.any()):
                    dir_cons.append(round(float((num[okm] / den[okm]).mean()), 4))
        dir_cons_valid = []
        if dir_cons and len(gap_curve) >= 2:
            _med_g = sorted(gap_curve)[len(gap_curve) // 2]
            for _k in range(len(dir_cons)):
                _ga, _gb = gap_curve[_k], gap_curve[_k + 1]
                if (_med_g / 3.0) <= _ga <= 3.0 * _med_g and \
                        (_med_g / 3.0) <= _gb <= 3.0 * _med_g:
                    dir_cons_valid.append(dir_cons[_k])
            if len(dir_cons_valid) < 2:
                dir_cons_valid = []
        # oscillation onset: the sigma where the x0 trajectory first reverses
        # direction (dir_cons < 0) -- where the late noise fade should anchor
        # (the noise stops helping and starts oscillating the trajectory).
        osc_sigma = None
        for _k, _dc in enumerate(dir_cons):
            if _dc < 0 and _k + 1 < len(grid):
                osc_sigma = float(grid[_k + 1])
                break
        curvs = []
        for i in range(1, m - 1):
            dl_prev = math.log(max(grid[i - 1], 1e-8)) - math.log(max(grid[i], 1e-8))
            dl_next = math.log(max(grid[i], 1e-8)) - math.log(max(grid[i + 1], 1e-8))
            dl = 0.5 * (abs(dl_prev) + abs(dl_next))
            if dl < 1e-6:
                continue
            dd = (x0s[i + 1] - 2.0 * x0s[i] + x0s[i - 1]).float().norm()
            curvs.append(float(dd) / (dl * dl)
                         / max(max(norms[i - 1], norms[i], norms[i + 1]), 1e-8))
        local_errs = []
        err_q = []
        guard_sigs = []
        frag_acc = None
        for i in range(1, m - 1):
            s_i = float(grid[i])
            s_n = float(grid[i + 1])
            s_p = float(grid[i - 1])
            if s_n <= 1e-8 or s_i <= 1e-8:
                continue
            x1_i = exact_step(xs[i], x0s[i], s_i, s_n)
            ls = math.log(max(s_i, 1e-8))
            ls2 = math.log(max(s_n, 1e-8))
            lsp = math.log(max(s_p, 1e-8))
            lmid = 0.5 * (ls + ls2)
            t = (lmid - ls) / (lsp - ls)
            x0_mid = x0s[i] + t * (x0s[i - 1] - x0s[i])
            x_pred_i = exact_step(xs[i], x0_mid, s_i, s_n)
            local_errs.append(float((x_pred_i - x1_i).float().norm()
                                    / x1_i.float().norm().clamp_min(1e-8)))
            if dist_on:
                epx = ((x_pred_i - x1_i).float().pow(2).mean(dim=1).sqrt()
                       / float(x1_i.float().pow(2).mean().sqrt().clamp_min(1e-8)))
                eq = torch.quantile(epx.reshape(-1),
                                    torch.tensor([0.1, 0.5, 0.9],
                                                 device=epx.device))
                err_q.append([round(float(v), 6) for v in eq])
                # per-pixel fragility: max local error over the trajectory
                if frag_acc is None:
                    frag_acc = epx.detach().float().unsqueeze(1)
                else:
                    frag_acc = torch.maximum(frag_acc, epx.detach().float().unsqueeze(1))
        local_errs_real = []
        _err_probe_gaps = gap_curve[1:1 + len(local_errs)]
        _err_real_gaps = real_gap_curve[1:1 + len(local_errs)]
        for _e, _gp, _gr in zip(local_errs, _err_probe_gaps, _err_real_gaps):
            if _gp > 1e-9 and _gr > 1e-9:
                local_errs_real.append(
                    _e * min(max((_gr / _gp) ** 2, 0.04), 1.0))
            else:
                local_errs_real.append(_e)
        for i in range(1, m):
            guard_sigs.append(float(_jump_signal(x0s[i], x0s[i - 1], xs[i])))
        g_pos = [max(float(v), 1e-8) for v in grid]
        gap_probe = (sum(abs(math.log(g_pos[i]) - math.log(g_pos[i + 1]))
                         for i in range(len(g_pos) - 1))
                     / max(1, len(g_pos) - 1))
        pos_all = [float(v) for v in sigmas if float(v) > 1e-6]
        gap_real = 0.0
        if len(pos_all) > 1:
            gap_real = (sum(abs(math.log(max(pos_all[i], 1e-8))
                                - math.log(max(pos_all[i + 1], 1e-8)))
                            for i in range(len(pos_all) - 1))
                        / (len(pos_all) - 1))

        def _pct(arr, p):
            if not arr:
                return 0.0
            a = sorted(arr)
            return a[min(len(a) - 1, int(p * len(a)))]

        med_jump = _pct(jumps, 0.5)
        med_curv = _pct(curvs, 0.5)
        total_jump = sum(jumps) + 1e-12
        early = jumps[: max(1, len(jumps) // 3)]
        early_frac = sum(early) / total_jump
        conv_step = len(jumps)
        thr = 0.05 * (max(jumps) if jumps else 1.0)
        for i, jv in enumerate(jumps):
            if jv < thr:
                conv_step = i
                break
        # Distilled = the x0 trajectory stops moving within the first few
        # steps AND most of its movement happens in that front window.  We do
        # NOT flag "last jump tiny relative to the max" as distilled: any
        # smooth fast-converging model (jump ~ sigma^p with p > 1) has a
        # small final jump once the grid reaches low sigma, so that clause
        # turned every cfg > 1 flow model into a false distilled positive.
        # The cfg override in _resolve_distilled_override already catches
        # true cfg <= 1 distilled models.
        is_distilled_eff = bool(early_frac > 0.75 and
                                conv_step <= max(2, len(jumps) // 3))
        is_distilled_eff = _resolve_distilled_override(
            cfg.get("probe_is_distilled", "auto"),
            cfg.get("_guider_live_cfg"),
            is_distilled_eff)
        _progress(callback, "probe", "progress", len(grid), len(grid) + 4,
                  "A-FloPS model probe: noise survival")
        safe_f = _probe_noise_survival(model, extra_args, xs, x0s, grid, ones,
                                       gen=_probe_gen(x))
        whitepoint = _whitepoint_from_x0s(x0s)
        endgame = None
        if bool(cfg.get("probe_endgame", True)):
            _progress(callback, "probe", "progress", len(grid) + 1, len(grid) + 4,
                      "A-FloPS model probe: endgame")
            pos_all_eg = [float(v) for v in sigmas if float(v) > 1e-6]
            endgame = _probe_endgame(model, extra_args, x0s[-1], grid[-1],
                                     pos_all_eg, ones, space, _probe_gen(x))
        profile = {
            "version": 7,
            "space": space,
            "n_probe_steps": m,
            "sigma_curve": [round(v, 6) for v in grid],
            "jump_curve": [round(v, 6) for v in jumps],
            "gap_curve": [round(v, 6) for v in gap_curve],
            "real_gap_curve": [round(v, 6) for v in real_gap_curve],
            "local_err_curve": [round(v, 6) for v in local_errs],
            "slope_curve": [round(j / max(gv, 1e-9), 8)
                            for j, gv in zip(jumps, gap_curve)],
            "med_local_err_real": (_pct(local_errs_real, 0.5)
                                   if local_errs_real else 0.0),
            "curv_curve": [round(v, 6) for v in curvs],
            "med_jump": med_jump,
            "med_curv": med_curv,
            "med_jump_rel": med_jump_rel,
            "med_detail_abs": med_detail_abs,
            "early_frac": early_frac,
            "conv_step": conv_step,
            "conv_sigma": (round(float(grid[int(conv_step)]), 6)
                           if 0 <= conv_step < len(grid) else None),
            "osc_sigma": (round(float(osc_sigma), 6)
                          if osc_sigma is not None else None),
            "is_distilled_eff": is_distilled_eff,
            "safe_fresh_frac": safe_f,
            "whitepoint": whitepoint,
            "med_local_err": (_pct(local_errs, 0.5) if local_errs else 0.0),
            "guard_sig_med": (_pct(guard_sigs, 0.5) if guard_sigs else 0.0),
            "guard_sig_p90": (_pct(guard_sigs, 0.9) if guard_sigs else 0.0),
            "log_gap_probe": gap_probe,
            "log_gap_real": gap_real,
            "n_jump": len(jumps),
            "n_err": len(local_errs),
            "n_guard": len(guard_sigs),
            "jump_q": jump_q,
            "err_q": err_q,
            "fragility": _fragility_map(frag_acc),
            "ojf_curve": ojf_curve,
            "band_curve": band_curve,
            "dir_cons": dir_cons,
            "dir_cons_valid": dir_cons_valid,
            "endgame": endgame,
            "probe_cfg": _probe_effective_cfg(model, extra_args),
        }
        profile["calib"] = _derive_calibrations(profile)
        profile["trajectory_state"] = _compute_trajectory_state(profile)
        return profile
    except Exception as e:
        logging.warning("[A-FloPS-probe] probe failed: %s", e)
        return None

def _apply_probe_profile(cfg, profile, tune, auto_tunable=None):
    if not profile:
        return
    calib = profile.get("calib", {})
    if auto_tunable is None:
        auto_tunable = {k for k in ENGINE_DEFAULTS
                        if cfg.get(k) == ENGINE_DEFAULTS[k]}

    def _set(key, val):
        if key not in auto_tunable:
            return
        try:
            cur = float(cfg.get(key))
            val_f = float(val)
            if cur > 0 and val_f > 0 and key != "warmup":
                val_f = max(0.25 * cur, min(4.0 * cur, val_f))
                cfg[key] = val_f
            else:
                cfg[key] = val
        except (TypeError, ValueError):
            cfg[key] = val

    if tune.get("guard", True) and "guard_floor" in calib:
        _set("guard_floor", calib["guard_floor"])
    if tune.get("tol", True) and "rtol" in calib:
        _set("rtol", calib["rtol"])
    if tune.get("wmax", True) and "wmax_base" in calib:
        _set("wmax_base", calib["wmax_base"])
    if tune.get("anomaly", True):
        for _ak in ("an_from", "an_spatial", "an_novelty", "an_value"):
            if _ak in calib:
                _set(_ak, calib[_ak])
    if tune.get("eta", False):
        se = calib.get("safe_eta")
        if se is not None and cfg.get("eta") is not None:
            if "eta" in auto_tunable:
                cfg["eta"] = min(float(cfg["eta"]), float(se))
    if profile.get("is_distilled_eff"):
        _set("warmup", 1)
    # The adaptive noise path ("auto") can only pick a beta once a jump map
    # exists -- the very first fresh-noise draw of a run had none and fell
    # back to plain white.  The probe already measured the model's spectral
    # structure (whitepoint); seed that first draw from it via the same
    # centroid->beta LUT the runtime uses, at half strength (evidence prior,
    # not a user override).
    wp = profile.get("whitepoint")
    if wp is not None and str(cfg.get("noise_color", "white")) == "auto":
        try:
            _bp = min(_centroid_to_beta(float(wp)), 0.0) * 0.5
            _cb_min = float(cfg.get("color_beta_min", -0.5) or 0.0)
            cfg["_noise_beta_prior"] = round(min(max(_bp, _cb_min), 0.0), 4)
        except (TypeError, ValueError):
            pass

def _model_weights_device(model_patcher):
    try:
        base_model = getattr(model_patcher, "model", None)
        if base_model is not None:
            try:
                return next(base_model.parameters()).device
            except (StopIteration, RuntimeError):
                pass
        ld = getattr(model_patcher, "load_device", None)
        if ld is not None:
            return torch.device(ld)
    except Exception:
        pass
    return torch.device("cpu")

def _ensure_model_gpu(model_patcher):
    try:
        ld = getattr(model_patcher, "load_device", None)
        if ld is None:
            return False
        ld = torch.device(ld)
        if ld.type == "cpu":
            return False
        try:
            if float(getattr(model_patcher, "loaded_size", lambda: 0.0)()
                     or 0.0) > 0.0:
                return True
        except (TypeError, ValueError, AttributeError):
            pass
        import comfy.model_management as _mm
        _load = getattr(_mm, "load_models_gpu", None)
        if _load is None:
            return False
        _load([model_patcher])
        logging.info("[A-FloPS-probe] model nudged onto %s before probing "
                     "(ComfyUI had left it offloaded)", ld)
        return True
    except Exception as e:
        logging.info("[A-FloPS-probe] could not nudge model onto the GPU "
                     "before probing (continuing on current devices): %s", e)
        return False

def _probe_call_device(model_patcher):
    try:
        ld = getattr(model_patcher, "load_device", None)
        if ld is not None:
            ld = torch.device(ld)
            if ld.type != "cpu":
                try:
                    resident = float(getattr(model_patcher, "loaded_size",
                                             lambda: 0.0)() or 0.0) > 0.0
                except (TypeError, ValueError, AttributeError):
                    resident = False
                if resident or _ensure_model_gpu(model_patcher):
                    return ld
    except Exception:
        pass
    return _model_weights_device(model_patcher)

def _is_device_error(e):
    try:
        msg = str(e).lower()
    except Exception:
        return False
    return ("device" in msg) or ("cuda" in msg) or ("cpu" in msg)

def _probe_failure_diagnostics(model, cur, ones, extra_args, gap_ctx=None):
    try:
        parts = []
        if torch.is_tensor(cur):
            parts.append("x=%s" % cur.device)
            parts.append("x_shape=%s" % (tuple(cur.shape),))
        if torch.is_tensor(ones):
            parts.append("t=%s" % ones.device)
        if isinstance(extra_args, dict):
            for k, v in list(extra_args.items())[:8]:
                if torch.is_tensor(v):
                    parts.append("%s=%s" % (k, v.device))
        if torch.cuda.is_available():
            parts.append("cur_dev=cuda:%d" % torch.cuda.current_device())
            parts.append("n_gpu=%d" % torch.cuda.device_count())
        patcher = None
        if isinstance(gap_ctx, dict):
            patcher = gap_ctx.get("patcher")
            if patcher is None:
                guider = gap_ctx.get("guider")
                if guider is not None:
                    patcher = getattr(guider, "model_patcher", None)
        if patcher is None:
            patcher = _navigate_to_patcher(model)
        if patcher is not None:
            ld = getattr(patcher, "load_device", None)
            try:
                ls = float(getattr(patcher, "loaded_size",
                                   lambda: 0.0)() or 0.0)
            except Exception:
                ls = -1.0
            parts.append("load_device=%s loaded_size=%.4g" % (ld, ls))
            base = getattr(patcher, "model", None)
            if base is not None:
                hist = {}
                try:
                    for _pn, _pp in base.named_parameters():
                        _d = str(_pp.device)
                        hist[_d] = hist.get(_d, 0) + 1
                except Exception:
                    pass
                if hist:
                    parts.append("weights=%s" % hist)
        return " ".join(parts) if parts else "(no device info)"
    except Exception:
        return "(diagnostics unavailable)"

def _guider_probe_cfg(guider):
    for attr in ("cfg1", "cfg"):
        v = _safe_getattr(guider, attr, None)
        if not isinstance(v, bool) and isinstance(v, (int, float)):
            return float(v)
    return None

def _guider_cond_signature(guider, probe_cfg=None):
    try:
        pos, neg = _guider_conds(guider)
        sig = _cond_signature(pos, neg)
        if probe_cfg is None:
            probe_cfg = _guider_probe_cfg(guider)
        if probe_cfg is not None:
            try:
                sig += "|cfg=%.6g" % float(probe_cfg)
            except (TypeError, ValueError):
                pass
        return sig
    except Exception:
        return None

def _generate_provisional_sigmas(model_patcher, steps=20):
    try:
        model_sampling = model_patcher.get_model_object("model_sampling")
        shift = getattr(model_sampling, "shift", None)
        if shift is None:
            shift = 1.0
        shift = float(shift)
        sigmas = aflops_scheduler_sigmas(model_sampling, int(steps),
                                          "log_uniform", shift=shift,
                                          denoise=1.0, tail_subdivide=1,
                                          tail_steps=1)
        return sigmas
    except Exception as e:
        logging.warning("[A-FloPS-probe] could not generate provisional "
                        "sigmas: %s", e)
        return None

def _set_probe_guider_conds(guider, positive, negative):
    def _list(c):
        if c is None:
            return []
        try:
            return list(c)
        except TypeError:
            return []

    def _is_converted(c):
        return bool(c) and all(isinstance(e, dict) for e in c)

    try:
        pos = _list(positive)
        neg = _list(negative)
        pos_conv = _is_converted(pos)
        neg_conv = _is_converted(neg)
        if pos_conv or neg_conv:
            try:
                import comfy.samplers as _cs
                try:
                    import comfy.sampler_helpers as _sh
                    _conv = getattr(_sh, "convert_cond", None)
                except Exception:
                    _conv = getattr(_cs, "convert_cond", None)
            except Exception:
                _conv = None

            def _norm(c, is_conv):
                if is_conv:
                    return [dict(e) if isinstance(e, dict) else e
                            for e in c]
                if not c:
                    return []
                if _conv is not None:
                    try:
                        r = _conv(list(c))
                        if r and all(isinstance(e, dict) for e in r):
                            return r
                    except Exception:
                        pass
                return list(c)

            pos_n = _norm(pos, pos_conv)
            neg_n = _norm(neg, neg_conv)
            guider.conds = {"positive": pos_n, "negative": neg_n}
            guider.original_conds = {
                "positive": list(pos_n), "negative": list(neg_n)}
            return True
        guider.set_conds(pos, neg)
        return True
    except Exception:
        return False

def _engine_cond_probe_run(model_patcher, positive, negative, cfg_scale,
                           cfg, sigmas, shape, callback=None, probe_fn=None,
                           probe_kind="cond"):
    """Run a probing pass as a REAL ComfyUI sampling run.

    This is the only way A-FloPS probes ever touch a model: the probe
    trajectory executes as the sampler function of a genuine
    CFGGuider + KSAMPLER run, so every model evaluation goes through
    ComfyUI's own sampling chain (prepare_sampling, process_conds,
    calc_cond_batch, cfg_function, model_function wrappers, additional
    models, guidance/concat/ref model_conds, ...) and the runtime prepares
    all device state for us (load_models_gpu, cuda_device_context,
    cast_to_load_options, pre_run).  That makes the probes MODEL-AGNOSTIC:
    SD/SDXL eps & v-prediction, Flux/Wan/Chroma-style flow models, video
    latents, guidance embeds, custom sampler_cfg_function hooks -- whatever
    ComfyUI can sample, the probe can measure, with no per-model handler
    code and nothing to re-implement when the next architecture lands.

    probe_fn(model, x, extra_args) -> profile overrides the default
    cond-probe call so the same machinery can also execute the MODEL probe
    node-side.  `model` is the KSamplerX0Inpaint wrapping the probe guider:
    calling it is exactly like calling the model during a real sampling
    step, and it returns the denoised (x0) prediction under the run's full
    CFG configuration.  `extra_args` is the run's own sampler kwargs
    (model_options, seed, denoise_mask) -- probes forward them so the chain
    model is invoked the way the revision requires (see _probe_model_call).
    """
    try:
        import comfy.samplers as _cs
        _Guider = getattr(_cs, "CFGGuider", None)
        _KS = getattr(_cs, "KSAMPLER", None)
        if _Guider is None or _KS is None:
            return None
        try:
            cfg_scale = float(cfg_scale)
        except (TypeError, ValueError):
            cfg_scale = 7.0
        prov_sigmas = sigmas
        try:
            s_max = float(prov_sigmas.reshape(-1)[0])
        except Exception:
            try:
                s_max = float(prov_sigmas[0])
            except Exception:
                return None
        if not (s_max > 1e-6):
            return None
        clone = model_patcher.clone()
        # The probing run is a shape-private execution: keep it away from the
        # model's internal caches exactly like every probe model call (see
        # _probe_model_options).  clone() shares the transformer_options dict
        # with the original patcher, so the sanitized dicts replace (never
        # mutate) the clone's model_options.
        try:
            _clone_mopt = _probe_model_options(
                {"model_options": getattr(clone, "model_options", None)})
            if _clone_mopt is not None:
                clone.model_options = _clone_mopt
        except Exception:
            pass
        guider = _Guider(clone)
        if not _set_probe_guider_conds(guider, positive, negative):
            return None
        guider.set_cfg(float(cfg_scale))

        holder = {"profile": None}

        def _probe_sampler_fn(model, x, sigmas, extra_args=None,
                              callback=None, disable=False):
            try:
                _probe_model_cache_guard(model, True)
                try:
                    if probe_fn is not None:
                        holder["profile"] = probe_fn(model, x,
                                                     dict(extra_args or {}))
                    else:
                        holder["profile"] = _run_cond_probe(
                            model, x, prov_sigmas, dict(extra_args or {}), cfg,
                            gap_ctx={"guider": guider, "cfg": float(cfg_scale),
                                     "patcher": model_patcher},
                            callback=None)
                finally:
                    _probe_model_cache_guard(model, False)
            except Exception as e:
                import traceback as _tb
                logging.warning("[A-FloPS-%s-probe] probing run failed "
                                "inside the sampling window: %s\n%s",
                                probe_kind, e, _tb.format_exc())
            return x

        sampler = _KS(_probe_sampler_fn)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260915)
        noise = torch.randn(shape, generator=gen, dtype=torch.float32)
        empty_latent = torch.zeros(shape, dtype=torch.float32)
        run_sigmas = torch.tensor([s_max, 0.0], dtype=torch.float32)
        # Guiders in the wild have drifted sample() signatures (custom
        # guiders, older ComfyUI revisions).  Try progressively simpler
        # invocations instead of giving up: there is no hand-rolled
        # model-call fallback anymore -- the chain IS the path.
        _sample_err = None
        for _kwargs in ({"denoise_mask": None, "callback": None,
                         "disable_pbar": True, "seed": 20260915},
                        {"denoise_mask": None, "callback": None,
                         "disable_pbar": True},
                        {"callback": None, "disable_pbar": True},
                        {}):
            try:
                guider.sample(noise, empty_latent, sampler, run_sigmas,
                              **_kwargs)
                _sample_err = None
                break
            except TypeError as _te:
                _sample_err = _te
            except Exception as _se:
                _sample_err = _se
                break
        if _sample_err is not None:
            logging.warning("[A-FloPS-%s-probe] probing run could not start "
                            "through the ComfyUI sampling chain: %s",
                            probe_kind, _sample_err)
            return None
        if holder["profile"] is not None:
            logging.info("[A-FloPS-%s-probe] probing run executed as a "
                         "real ComfyUI sampling run (device state fully "
                         "prepared by the runtime: load_models_gpu + "
                         "additional models, cuda_device_context, "
                         "cast_to_load_options, pre_run)", probe_kind)
        return holder["profile"]
    except Exception as e:
        import traceback as _tb
        logging.info("[A-FloPS-%s-probe] engine-run probe unavailable "
                     "(falling back to the direct path): %s\n%s",
                     probe_kind, e, _tb.format_exc())
        return None

def _sig_list_sig(lst):
    """Compact signature of a sigma list (for schedule-keyed probe caches)."""
    try:
        return hashlib.md5(
            ",".join("%.5f" % float(v) for v in lst).encode("utf-8")
        ).hexdigest()[:12]
    except Exception:
        return "?"


def _norm_sig_list(sigmas):
    """External sigma input -> descending float list, or None."""
    if sigmas is None:
        return None
    try:
        lst = [float(v) for v in sigmas]
    except Exception:
        return None
    if len(lst) < 4 or lst[0] <= lst[1] or lst[0] <= 0.0:
        return None
    return lst


def _run_node_side_cond_probe(model_patcher, positive, negative, cfg,
                              cfg_scale=7.0, sigmas=None):
    try:
        mk = _model_cache_key(model_patcher)
        cond_sig = cfg.get("cond_probe_sig")
        ext = _norm_sig_list(sigmas)
        _key = (mk, str(cond_sig) + ("|g:" + _sig_list_sig(ext)
                                    if ext is not None else ""))
        if cond_sig is not None and not bool(cfg.get("cond_probe_force", False)):
            cached = _COND_PROBE_PROFILES.get(_key)
            if cached is not None:
                _PROFILE_BY_MODEL[mk] = cached
                logging.info("[A-FloPS-cond-probe] node-side probe: model+prompt "
                             "unchanged, using cached profile (skip re-probe)")
                return cached, True
        if ext is not None:
            sigmas = ext
        else:
            sigmas = _generate_provisional_sigmas(model_patcher, steps=20)
        if sigmas is None or len(sigmas) < 4:
            return None, False
        pres = max(8, int(cfg.get("cond_probe_resolution", 16)))
        mi = _extract_model_info(model_patcher)
        B = 1
        def _fallback_latent_channels():
            try:
                lf = model_patcher.get_model_object("latent_format")
                v = _lf_get(lf, "latent_channels", 4)
                return int(v)
            except Exception:
                return 4
        if mi and mi.get("latent_channels"):
            C = int(mi["latent_channels"])
        else:
            C = _fallback_latent_channels()
        shape = _probe_latent_shape(ref_x=None, pres=pres, batch=B,
                                    channels=C, patcher=model_patcher, mi=mi)
        profile = _engine_cond_probe_run(model_patcher, positive, negative,
                                         cfg_scale, cfg, sigmas, shape)
        if profile is None:
            # No hand-rolled fallback: if the node-side chain run could not
            # execute, defer to the engine-side probe, which runs inside the
            # real sampling run through the very same ComfyUI chain.
            logging.info("[A-FloPS-cond-probe] node-side chain probe did "
                         "not execute; deferring to the engine-side probe "
                         "at sampling time")
            return None, False

        if profile is not None and cond_sig is not None:
            _COND_PROBE_PROFILES[_key] = profile
            _plain = (mk, str(cond_sig))
            if _key != _plain:
                # Plain (schedule-agnostic) slot: the engine's per-prompt
                # fallback and any legacy plain-key lookup read it; the
                # schedule-keyed slot keeps freshness per schedule.
                _COND_PROBE_PROFILES[_plain] = profile
            _PROFILE_BY_MODEL[mk] = profile
        return profile, False
    except Exception as e:
        logging.warning("[A-FloPS-cond-probe] node-side probe failed "
                        "(falling back to engine-side): %s", e)
        return None, False

def _run_node_side_cond_probe_guider(guider, cfg, cfg_scale=None,
                                     sigmas=None):
    patcher = getattr(guider, "model_patcher", None)
    if patcher is None:
        patcher = _navigate_to_patcher(guider)
    if patcher is None:
        logging.warning("[A-FloPS-cond-probe] guider exposes no ModelPatcher; "
                        "node-side probe skipped")
        return None, False
    positive, negative = _guider_conds(guider)
    if positive is None and negative is None:
        logging.warning("[A-FloPS-cond-probe] guider carries no conditioning; "
                        "node-side probe skipped")
        return None, False
    eff_cfg = _guider_probe_cfg(guider)
    if eff_cfg is None:
        try:
            eff_cfg = float(cfg_scale if cfg_scale is not None
                            else cfg.get("cond_probe_cfg", 7.0))
        except (TypeError, ValueError):
            eff_cfg = 7.0
    if negative is None or (isinstance(negative, (list, tuple))
                            and len(negative) == 0):
        eff_cfg = 1.0
    return _run_node_side_cond_probe(patcher, positive, negative, cfg,
                                     eff_cfg, sigmas=sigmas)


def _node_probe_param_sig(cfg):
    """Fingerprint of the probe parameters that are baked into a model
    profile's measurement (widget changes must invalidate the cache)."""
    return "|".join([
        str(int(cfg.get("probe_steps", 10))),
        str(int(cfg.get("probe_resolution", 48))),
        "d" if bool(cfg.get("probe_distributions", True)) else "-",
        "e" if bool(cfg.get("probe_endgame", True)) else "-",
        "c" if bool(cfg.get("probe_dir_cons", True)) else "-",
    ])


def _run_node_side_model_probe(guider, cfg, cfg_scale=None, sigmas=None):
    """Node-side MODEL probe: pre-compute the calibration profile in the
    Probe Options node BEFORE the KSampler starts.

    This is the model-probe counterpart of _run_node_side_cond_probe_guider.
    Without it the model probe only runs inside aflops_engine at SAMPLING
    time, so the profile arrives one generation too late for any node that
    executes before the KSampler -- most importantly the A-FloPS Scheduler,
    whose probe-based shift recommendation and distilled detection are blind
    on the first run.

    The probe borrows the guider's ModelPatcher, conditioning and live cfg
    scale, measures on a provisional log-uniform schedule (the consumers
    resample the profile's curves onto the real sigmas, exactly like the
    node-side cond probe), and caches the profile in _NODE_PROBE_PROFILES,
    _PROBE_PROFILES ("<mk>|s:node:<params>|<cfg>") and _PROFILE_BY_MODEL so
    the Scheduler can see it immediately and the engine can skip its own
    sampling-time probe.

    The measurement runs as a real ComfyUI sampling run through the chain
    (see _engine_cond_probe_run) -- model-agnostic, no per-model handlers.

    Returns (profile, from_cache); (None, False) means "fall back to the
    legacy engine-side probe".
    """
    try:
        patcher = getattr(guider, "model_patcher", None)
        if patcher is None:
            patcher = _navigate_to_patcher(guider)
        if patcher is None:
            logging.info("[A-FloPS-probe] guider exposes no ModelPatcher; "
                         "node-side model probe skipped")
            return None, False
        positive, negative = _guider_conds(guider)
        if positive is None and negative is None:
            logging.info("[A-FloPS-probe] guider carries no conditioning; "
                         "node-side model probe skipped")
            return None, False
        eff_cfg = _guider_probe_cfg(guider)
        if eff_cfg is None:
            eff_cfg = cfg_scale if cfg_scale is not None else 7.0
        try:
            eff_cfg = float(eff_cfg)
        except (TypeError, ValueError):
            eff_cfg = 7.0
        if negative is None or (isinstance(negative, (list, tuple))
                                and len(negative) == 0):
            eff_cfg = 1.0
        # The distilled 'auto' override keys off the live probe cfg; make it
        # visible to _run_probe (the engine recomputes it at sampling time).
        cfg.setdefault("_guider_live_cfg", eff_cfg)
        mk = _model_cache_key(patcher)
        ext = _norm_sig_list(sigmas)
        nkey = (mk, _node_probe_param_sig(cfg), "%.2f" % eff_cfg,
                _sig_list_sig(ext) if ext is not None else "provisional")
        if not bool(cfg.get("probe_force", False)):
            cached = _NODE_PROBE_PROFILES.get(nkey)
            if cached is not None and int(cached.get("version", 1) or 1) >= 7:
                _PROFILE_BY_MODEL[mk] = cached
                logging.info("[A-FloPS-probe] node-side probe: model "
                             "unchanged, using cached profile (skip "
                             "re-probe)")
                return cached, True
        if ext is not None:
            sigmas = ext
        else:
            sigmas = _generate_provisional_sigmas(patcher, steps=20)
        if sigmas is None or len(sigmas) < 4:
            return None, False
        pres = max(16, int(cfg.get("probe_resolution", 48)))
        mi = _extract_model_info(patcher)
        B = 1

        def _fallback_latent_channels():
            try:
                lf = patcher.get_model_object("latent_format")
                v = _lf_get(lf, "latent_channels", 4)
                return int(v)
            except Exception:
                return 4

        if mi and mi.get("latent_channels"):
            C = int(mi["latent_channels"])
        else:
            C = _fallback_latent_channels()
        shape = _probe_latent_shape(ref_x=None, pres=pres, batch=B,
                                    channels=C, patcher=patcher, mi=mi)

        def _probe_fn(model, x, extra_args=None):
            prof = _run_probe(model, x, sigmas, dict(extra_args or {}), cfg)
            if prof is not None:
                prof["probe_cfg"] = eff_cfg
            return prof

        profile = _engine_cond_probe_run(
            patcher, positive, negative, eff_cfg, cfg, sigmas, shape,
            probe_fn=_probe_fn, probe_kind="model")
        if profile is None:
            # No hand-rolled fallback: the engine-side probe (inside the
            # real sampling run, through the very same ComfyUI chain) takes
            # over at sampling time instead.
            logging.info("[A-FloPS-probe] node-side chain probe did not "
                         "execute; deferring to the engine-side probe at "
                         "sampling time")
            return None, False

        profile["node_side"] = True
        _NODE_PROBE_PROFILES[nkey] = profile
        try:
            _PROBE_PROFILES["%s|s:node:%s|%s" % (mk, nkey[1], nkey[2])] = \
                profile
            _prune_probe_cache(mk, keep=8)
        except Exception:
            pass
        _PROFILE_BY_MODEL[mk] = profile
        return profile, False
    except Exception as e:
        logging.warning("[A-FloPS-probe] node-side probe failed (falling "
                        "back to engine-side): %s", e)
        return None, False

def _compute_trajectory_state(profile):
    try:
        is_distilled = bool(profile.get("is_distilled_eff", False))
        jumps = profile.get("jump_curve", [])
        gaps_c = profile.get("gap_curve") or []
        if gaps_c and len(gaps_c) == len(jumps):
            slope_j = [j / max(g, 1e-9) for j, g in zip(jumps, gaps_c)]
        else:
            slope_j = jumps
        conv_ratio = 1.0  # neutral default
        if len(slope_j) >= 3:
            n3 = max(1, len(slope_j) // 3)
            early = sum(slope_j[:n3]) / n3
            late = sum(slope_j[-n3:]) / n3
            if early > 1e-8:
                conv_ratio = late / early
        trend_resid = None
        if len(slope_j) >= 6 and all(s > 1e-12 for s in slope_j):
            _k = max(2, min(3, len(slope_j) // 3))
            _tail = slope_j[-_k:]
            _mid = slope_j[-2 * _k:-_k]
            _t_med = sorted(_tail)[len(_tail) // 2]
            _m_med = sorted(_mid)[len(_mid) // 2]
            if _t_med > 1e-12 and _m_med > 1e-12:
                trend_resid = math.log(_t_med / _m_med)
        med_err_raw = float(profile.get("med_local_err", 0.0) or 0.0)
        gap_p = float(profile.get("log_gap_probe", 0.0) or 0.0)
        gap_r = float(profile.get("log_gap_real", 0.0) or 0.0)
        lec = profile.get("local_err_curve") or []
        gpc = profile.get("gap_curve") or []
        grc = profile.get("real_gap_curve") or []
        med_err = None
        if (lec and gpc and grc and len(lec) == len(gpc)
                and len(lec) == len(grc)):
            per = []
            for _e, _gp, _gr in zip(lec, gpc, grc):
                if _gp > 1e-9 and _gr > 1e-9:
                    per.append(_e * min(max((_gr / _gp) ** 2, 0.04), 1.0))
                else:
                    per.append(_e)
            if per:
                med_err = sorted(per)[len(per) // 2]
        if med_err is None:
            stored_real = profile.get("med_local_err_real")
            if stored_real is not None and med_err_raw > 0:
                med_err = float(stored_real)
            else:
                med_err = med_err_raw
                if med_err_raw > 0 and gap_p > 1e-9 and gap_r > 1e-9:
                    scale = min(max((gap_r / gap_p) ** 2, 0.04), 1.0)
                    med_err = med_err_raw * scale
        if trend_resid is not None:
            s_ovs_conv = max(
                max(0.0, min(1.0, (trend_resid - 0.3) / 0.7)),
                max(0.0, min(1.0, (conv_ratio - 1.0) / 1.0)))
        else:
            s_ovs_conv = max(0.0, min(1.0, (conv_ratio - 1.0) / 1.0))
        curvs = profile.get("curv_curve", [])
        med_curv = float(profile.get("med_curv", 0.0) or 0.0)
        late_curv_ratio = 1.0
        if len(curvs) >= 2 and med_curv > 1e-8:
            late_curv = sum(curvs[len(curvs) // 2:]) / max(1, len(curvs) - len(curvs) // 2)
            late_curv_ratio = late_curv / med_curv
        s_ovs_curv = max(0.0, min(1.0, (late_curv_ratio - 1.0) / 2.0))
        s_ovs_err = 1.0 - math.exp(-med_err / 0.03)
        gap = profile.get("gap")
        probe_cfg = None
        try:
            probe_cfg = float(profile.get("probe_cfg"))
        except (TypeError, ValueError):
            probe_cfg = None
        s_ovs_ov = 0.0
        if gap is not None and probe_cfg is not None and probe_cfg > 1.0 + 1e-6:
            rel_ov = float(gap.get("overshoot", 0.0)) / (probe_cfg - 1.0)
            s_ovs_ov = 1.0 - math.exp(-rel_ov / 1.2)
        sig_med = float(profile.get("guard_sig_med", 0.0) or 0.0)
        sig_p90 = float(profile.get("guard_sig_p90", 0.0) or 0.0)
        spike_ratio = (sig_p90 / max(sig_med, 1e-8)
                       if sig_med > 1e-8 else 1.0)
        # The guard-signal p90/median ratio is a ratio that spans orders of
        # magnitude (a hot-pixel model measures ~17, a smooth one ~1.5).  A
        # log mapping keeps it in dynamic range instead of the exponential
        # saturating to ~1 for anything above ~5 and drowning the other
        # oversteer signals.
        s_ovs_spike = min(1.0, max(0.0,
                                   math.log(max(spike_ratio, 1.0)) / math.log(30.0)))
        s_ovs_bimod = 0.0
        ojf = profile.get("ojf_curve") or []
        if ojf:
            ojf_mean = sum(float(x) for x in ojf) / len(ojf)
            s_ovs_bimod = max(0.0, min(1.0, (ojf_mean - 0.06) / 0.24))

        signals = [s_ovs_conv, s_ovs_curv, s_ovs_err, s_ovs_ov, s_ovs_spike]
        if ojf and s_ovs_bimod > 0.0:
            signals.append(s_ovs_bimod)
        not_oversteer = 1.0
        for sv in signals:
            not_oversteer *= (1.0 - sv)
        oversteer = max(0.0, min(1.0, 1.0 - not_oversteer ** (1.0 / len(signals))))
        if is_distilled:
            oversmooth = 0.0
        else:
            if trend_resid is not None:
                s_osm_conv = max(0.0, min(1.0, (-3.2 - trend_resid) / 0.8))
            else:
                s_osm_conv = max(0.0, min(1.0, (0.03 - conv_ratio) / 0.03))
            s_osm_curv = max(0.0, min(1.0, (0.3 - late_curv_ratio) / 0.3))
            s_osm_err = max(0.0, min(1.0, (0.003 - med_err) / 0.003))
            oversmooth = max(0.0, min(1.0,
                s_osm_conv * (0.7 + 0.3 * max(s_osm_curv, s_osm_err))))

        return {
            "oversteer": round(oversteer, 4),
            "oversmooth": round(oversmooth, 4),
            "med_local_err_raw": round(med_err_raw, 6),
            "med_local_err_real": round(med_err, 6),
        }
    except Exception:
        return {"oversteer": 0.0, "oversmooth": 0.0}

def _safe_getattr(o, name, default=None):
    try:
        for cls in type(o).__mro__:
            if name in cls.__dict__:
                return getattr(o, name)
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict) and name in d:
            return d[name]
    except Exception:
        pass
    return default

def _looks_like_guider(o):
    try:
        if o is None or isinstance(o, (int, float, bool, str, bytes)):
            return False
        if not callable(_safe_getattr(o, "set_cfg", None)):
            return False
        cfg = _safe_getattr(o, "cfg", None)
        if isinstance(cfg, bool) or not isinstance(cfg, (int, float)):
            return False
        conds = _safe_getattr(o, "conds", None)
        if isinstance(conds, dict) and ("positive" in conds
                                        or "negative" in conds):
            return True
        if _safe_getattr(o, "positive", None) is not None:
            return True
        orig = _safe_getattr(o, "original_conds", None)
        if isinstance(orig, dict) and ("positive" in orig
                                       or "negative" in orig):
            return True
        return False
    except Exception:
        return False

def _guider_conds(g):
    pos = neg = None
    conds = _safe_getattr(g, "conds", None)
    if isinstance(conds, dict):
        pos = conds.get("positive")
        neg = conds.get("negative")
    if pos is None:
        orig = _safe_getattr(g, "original_conds", None)
        if isinstance(orig, dict):
            pos = orig.get("positive")
            neg = orig.get("negative")
    if pos is None:
        pos = _safe_getattr(g, "positive", None)
        if neg is None:
            neg = _safe_getattr(g, "negative", None)
    return pos, neg

def _iter_obj_graph(seed, max_nodes=64):
    seen = set()
    stack = [seed]
    count = 0
    while stack and count < max_nodes:
        o = stack.pop(0)
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        count += 1
        yield o
        fn = o
        if isinstance(fn, functools.partial):
            stack.append(fn.func)
            stack.extend(fn.args)
            try:
                stack.extend(fn.keywords.values())
            except Exception:
                pass
            fn = fn.func
        if callable(fn):
            f = _safe_getattr(fn, "__func__", fn)
            for cell in (_safe_getattr(f, "__closure__", None) or ()):
                try:
                    stack.append(cell.cell_contents)
                except (ValueError, TypeError):
                    pass
        if callable(o):
            self_obj = _safe_getattr(o, "__self__", None)
            if self_obj is not None:
                stack.append(self_obj)
        d = _safe_getattr(o, "__dict__", None)
        if isinstance(d, dict):
            for v in list(d.values())[:24]:
                stack.append(v)

def _find_guider_in_stack(limit=24):
    try:
        import sys as _sys
        f = _sys._getframe(1)
        n = 0
        while f is not None and n < limit:
            for v in list(f.f_locals.values()):
                if _looks_like_guider(v):
                    return v
            f = f.f_back
            n += 1
    except Exception:
        pass
    return None

def _resolve_gap_source(model, extra_args):
    pos = neg = None
    cfg = None
    if isinstance(extra_args, dict):
        pos = extra_args.get("cond")
        neg = extra_args.get("uncond")
        raw = extra_args.get("cond_scale", extra_args.get("cfg"))
        try:
            cfg = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            cfg = None
    guider = None
    for cand in _iter_obj_graph(model):
        if _looks_like_guider(cand):
            guider = cand
            break
    if guider is None:
        guider = _find_guider_in_stack()
    if guider is not None:
        gpos, gneg = _guider_conds(guider)
        if pos is None:
            pos, neg = gpos, gneg
        gcfg = _safe_getattr(guider, "cfg", None)
        if not isinstance(gcfg, bool) and isinstance(gcfg, (int, float)):
            if cfg is None:
                cfg = float(gcfg)
    return pos, neg, cfg, guider

def _build_gap_ctx(model, extra_args, patcher=None, positive=None,
                   negative=None):
    if patcher is None:
        patcher = _navigate_to_patcher(model)
    r_pos, r_neg, r_cfg, guider = _resolve_gap_source(model, extra_args)
    if positive is None:
        positive = r_pos
    if negative is None:
        negative = r_neg
    return {"patcher": patcher, "positive": positive, "negative": negative,
            "cfg": r_cfg, "guider": guider}

def _probe_effective_cfg(model, extra_args, gap_ctx=None):
    if isinstance(gap_ctx, dict):
        v = gap_ctx.get("cfg")
        if not isinstance(v, bool) and isinstance(v, (int, float)):
            return float(v)
    try:
        v = _safe_getattr(model, "cfg", None)
        if v is not None:
            return float(v)
    except Exception:
        pass
    if isinstance(extra_args, dict):
        for key in ("cond_scale", "cfg"):
            try:
                if extra_args.get(key) is not None:
                    return float(extra_args[key])
            except (TypeError, ValueError):
                pass
    return None


def _resolve_distilled_override(mode, probe_cfg=None, heuristic=False):
    if mode is None:
        mode = "auto"
    if isinstance(mode, bool):
        return mode
    m = str(mode).strip().lower()
    if m in ("yes", "true", "on", "force", "1"):
        return True
    if m in ("no", "false", "off", "0"):
        return False
    if probe_cfg is not None:
        try:
            if float(probe_cfg) <= 1.0 + 1e-6:
                return True
        except (TypeError, ValueError):
            pass
    return bool(heuristic)

def _measure_guidance_gap(model, extra_args, xs, x0s, grid, ones,
                          gap_ctx=None):
    """Cond/uncond decomposition of the guidance gap, performed WITH
    ComfyUI's own machinery: the guider's cfg is temporarily pinned to 1.0
    (the chain's sampling_function/cfg_function then return the cond
    prediction) and to 0.0 (they return the uncond prediction).  Custom
    sampler_cfg_function hooks and the model's model_conds are honoured
    because the full sampling chain evaluates both calls -- no hand-rolled
    CFG, works for every model ComfyUI can sample."""
    try:
        probe_cfg = _probe_effective_cfg(model, extra_args, gap_ctx)
        if probe_cfg is not None and probe_cfg <= 1.0 + 1e-6:
            return None
        m = len(x0s)
        i = m // 2
        if i >= m:
            return None
        s_m = grid[i]
        x_at = xs[i]
        x0_cfg = x0s[i]
        guider = gap_ctx.get("guider") if isinstance(gap_ctx, dict) else None
        if guider is None:
            # No guider in reach -> skip the measurement.  Decomposing cond
            # vs uncond without ComfyUI's own guider would need hand-rolled
            # per-model CFG code, which A-FloPS no longer carries; the
            # consumers treat a missing gap as "unknown guidance strength".
            return None

        saved_cfg = getattr(guider, "cfg", None)
        x0_cond = x0_uncond = None
        try:
            guider.set_cfg(1.0)
            x0_cond = _probe_model_call(model, x_at, s_m * ones, extra_args)
            guider.set_cfg(0.0)
            x0_uncond = _probe_model_call(model, x_at, s_m * ones, extra_args)
        except Exception:
            x0_cond = x0_uncond = None
        finally:
            try:
                if saved_cfg is not None:
                    guider.set_cfg(saved_cfg)
            except Exception:
                pass

        if x0_cond is None or x0_uncond is None:
            return None
        return _gap_outputs(x0_cfg, x0_cond, x0_uncond)
    except Exception:
        return None

def _gap_outputs(x0_cfg, x0_cond, x0_uncond):
    uncond_norm = float(x0_uncond.float().norm().clamp_min(1e-8))
    strength = float((x0_cond - x0_uncond).float().norm()) / uncond_norm
    cond_norm = float(x0_cond.float().norm().clamp_min(1e-8))
    overshoot = float((x0_cfg - x0_cond).float().norm()) / cond_norm
    out = {"strength": round(strength, 4), "overshoot": round(overshoot, 4)}
    try:
        gd = (x0_cond.float() - x0_uncond.float())
        if gd.dim() == 5:
            gd = gd.mean(dim=2)          # collapse frames (broadcast back)
        if gd.dim() == 4 and gd.shape[0] == 1:
            c_n = int(gd.shape[1])
            h, w = gd.shape[-2:]
            if min(h, w) > 8:
                gd = F.adaptive_avg_pool2d(gd, (8, 8))
                h, w = gd.shape[-2:]
            nrm = float(gd.norm())
            if nrm > 1e-8:
                gdn = (gd / nrm)[0]
                out["prior_dir"] = {
                    "c": c_n, "h": int(h), "w": int(w),
                    "v": [round(float(t), 4) for t in gdn.reshape(-1)]}
        pm = gd.pow(2).mean(dim=1).sqrt()[0] if gd.dim() == 4 else \
            (x0_cond.float() - x0_uncond.float()).pow(2).mean(dim=1).sqrt()
        p90 = float(torch.quantile(pm.reshape(-1), 0.9))
        if p90 > 1e-8:
            pm = (pm / p90).clamp(0.0, 1.0)
            h, w = pm.shape[-2:]
            if min(h, w) > 16:
                pm = F.adaptive_avg_pool2d(pm.reshape(1, 1, h, w), (16, 16))[0, 0]
                h, w = pm.shape[-2:]
            out["prior"] = {"h": int(h), "w": int(w),
                            "v": [round(float(x), 3) for x in pm.reshape(-1)]}
    except Exception:
        pass
    return out

def _derive_cond_calibrations(profile):
    calib = {}
    med_curv = profile.get("med_curv", 0.0)
    med_err_raw = float(profile.get("med_local_err", 0.0) or 0.0)
    gap_p = float(profile.get("log_gap_probe", 0.0) or 0.0)
    gap_r = float(profile.get("log_gap_real", 0.0) or 0.0)
    med_err = med_err_raw
    if med_err_raw > 0 and gap_p > 1e-9 and gap_r > 1e-9:
        med_err = med_err_raw * min(max((gap_r / gap_p) ** 2, 0.04), 1.0)
    if med_err > 0:
        calib["rtol"] = round(min(0.04, max(0.005, 3.0 * med_err)), 4)
    sig_med = float(profile.get("guard_sig_med", 0.0) or 0.0)
    sig_p90 = float(profile.get("guard_sig_p90", 0.0) or 0.0)
    if sig_med > 1e-9:
        r = sig_p90 / sig_med
        calib["guard_floor"] = round(
            min(0.5, max(0.05, 0.15 * math.sqrt(max(r, 0.25) / 4.0))), 4)
    if med_curv > 0:
        lc = math.log(max(med_curv, 1e-8) + 1.0)
        if lc < math.log(1.5):
            wb = 3.0
        elif lc > math.log(5.0):
            wb = 1.8
        else:
            t = (lc - math.log(1.5)) / (math.log(5.0) - math.log(1.5))
            wb = 3.0 - t * (3.0 - 1.8)
        calib["wmax_base"] = round(max(1.5, min(2.5, wb)), 3)
    if med_err > 0:
        calib["safe_eta"] = round(
            max(0.1, min(1.0, 0.7 / (med_err * 20.0 + 1.0))), 3)
    gap = profile.get("gap")
    if gap is not None:
        strength = float(gap.get("strength", 1.0))
        if strength < 0.5 and "rtol" in calib:
            calib["rtol"] = round(min(0.04, calib["rtol"] * 1.5), 4)
        elif strength > 2.0 and "guard_floor" in calib:
            calib["guard_floor"] = round(max(0.05, calib["guard_floor"] * 0.7), 4)
    calib.update(_derive_anomaly_calibrations(profile))
    return calib

def _inherit_base_guard_stats(focused, base):
    if not isinstance(focused, dict) or not isinstance(base, dict):
        return focused
    bm = base.get("guard_sig_med")
    bp = base.get("guard_sig_p90")
    bf = base.get("fragility")
    if bm is None and bp is None and bf is None:
        return focused
    if (focused.get("guard_sig_med") == bm
            and focused.get("guard_sig_p90") == bp
            and focused.get("fragility") == bf):
        return focused
    out = dict(focused)
    if bm is not None:
        out["guard_sig_med"] = bm
    if bp is not None:
        out["guard_sig_p90"] = bp
    if bf is not None and out.get("fragility") is None:
        # the focused probe inherits the base probe's spatial fragility map
        # (the focused trajectory is too short to re-measure it reliably)
        out["fragility"] = bf
    try:
        out["trajectory_state"] = _compute_trajectory_state(out)
    except Exception:
        pass
    return out


def _carry_run_feedback(new_prof, old_prof):
    if not isinstance(new_prof, dict) or not isinstance(old_prof, dict):
        return new_prof
    fb = old_prof.get("dir_real")
    if isinstance(fb, list) and fb and "dir_real" not in new_prof:
        new_prof["dir_real"] = list(fb)
    return new_prof


def _detect_fragile_ranges(prof_m, prof_c):
    ranges = []
    for prof in (prof_m, prof_c):
        if not isinstance(prof, dict):
            continue
        eg = prof.get("endgame")
        if not isinstance(eg, dict):
            continue
        for lv in (eg.get("levels") or []):
            try:
                s = float(lv.get("s", 0))
                if s < 1e-6:
                    continue
                rec_q = lv.get("rec_q") or lv.get("rec_p50")
                if isinstance(rec_q, (list, tuple)) and len(rec_q) >= 2:
                    rec_p50 = float(rec_q[1])
                elif isinstance(rec_q, (int, float)):
                    rec_p50 = float(rec_q)
                else:
                    continue
                rec2_q = lv.get("rec2_q")
                if isinstance(rec2_q, (list, tuple)) and len(rec2_q) >= 2:
                    rec2_p50 = float(rec2_q[1])
                    rec_p50 = max(rec_p50, rec2_p50)
                frag = min(1.0, max(0.0, rec_p50 / 0.06))
                if frag > 0.15:  # threshold for "worth focusing on"
                    ranges.append((s, frag))
            except (TypeError, ValueError, IndexError):
                continue
    ranges.sort(key=lambda r: -r[1])
    if len(ranges) >= 2:
        sigmas_sorted = sorted([r[0] for r in ranges])
        return [(sigmas_sorted[0], sigmas_sorted[-1],
                 max(r[1] for r in ranges))]
    elif len(ranges) == 1:
        s, f = ranges[0]
        return [(s * 0.7, s * 1.3, f)]
    return []

def _rescue_window_from_fragile(sigmas, fragile_ranges):
    """Map measured endgame-fragile sigma ranges onto the detail-rescue
    window (kn space).  Returns (start, full) or (None, None).

    The rescue envelope ramps in between `start` and `full` (fractions of the
    run); by default those are fixed at 0.55/0.72 no matter WHERE the model
    actually misinterprets structured state.  The endgame probe measured
    exactly that (recovery error after re-noising the final x0 to lower
    sigmas), so place the ramp over the measured fragile band instead.
    """
    try:
        sv = [float(v) for v in (sigmas if sigmas is not None else [])
              if float(v) > 1e-6]
        if len(sv) < 4 or not fragile_ranges:
            return None, None
        n_steps = len(sv)  # engine kn = step_index / n_steps

        def _kn_of(sigma):
            for i, v in enumerate(sv):
                if v <= sigma:
                    return i / float(n_steps)
            return 1.0

        best = None
        for s_lo, s_hi, frag in fragile_ranges:
            try:
                k0 = _kn_of(float(s_hi))   # larger sigma = earlier step
                k1 = _kn_of(float(s_lo))   # smaller sigma = later step
            except (TypeError, ValueError):
                continue
            if k1 - k0 < 0.02:
                continue
            if best is None or float(frag) > best[2]:
                best = (k0, k1, float(frag))
        if best is None:
            return None, None
        start = min(max(best[0], 0.25), 0.80)
        full = min(max(best[1], start + 0.05), 0.90)
        return round(start, 3), round(full, 3)
    except Exception:
        return None, None

def _build_focused_grid(sigmas, fragile_ranges, K_base=4, K_max=8,
                        distilled_mode=False):
    base = _probe_grid(sigmas, K_base, distilled_mode) or []
    if not base:
        return []
    extra = []
    for s_lo, s_hi, frag in fragile_ranges:
        n_extra = min(3, max(1, int(frag * 3)))
        s_lo = max(s_lo, 1e-6)
        s_hi = max(s_hi, s_lo * 1.1)
        for i in range(n_extra):
            t = (i + 0.5) / n_extra
            s_extra = s_hi * (s_lo / s_hi) ** t
            extra.append(s_extra)
    grid = sorted(set(base + extra), reverse=True)
    if len(grid) > K_max:
        grid = grid[:K_max]
    return grid

def _run_focused_cond_probe(model, x, sigmas, extra_args, cfg, gap_ctx=None,
                            callback=None, fragile_ranges=None,
                            focused_grid=None, base_profile=None):
    try:
        _pf_patcher = gap_ctx.get("patcher") if isinstance(gap_ctx, dict) else None
        if _pf_patcher is not None:
            _pf_dev = _probe_call_device(_pf_patcher)
            if x.device != _pf_dev:
                x = x.to(_pf_dev)
        pres = max(8, int(cfg.get("cond_probe_resolution", 16)))
        B = 1
        C = x.shape[1]
        if B < 1 or C < 1:
            return None
        shape = _probe_latent_shape(ref_x=x, pres=pres, batch=B)
        px = torch.randn(shape, device=x.device, dtype=torch.float32,
                         generator=_probe_gen(x)).to(x.dtype)
        ones = px.new_ones([B])
        grid = focused_grid
        if not grid or len(grid) < 3:
            return None
        grid = sorted(grid, reverse=True)
        px = px * float(grid[0])
        pos_all = [float(v) for v in sigmas if float(v) > 1e-6]
        gap_curve = [abs(math.log(max(grid[i], 1e-8))
                         - math.log(max(grid[i + 1], 1e-8)))
                     for i in range(len(grid) - 1)]
        real_gap_curve = []
        if len(pos_all) >= 2:
            pos_idx = []
            _j = 0
            for gv in grid:
                while _j < len(pos_all) and pos_all[_j] > gv * (1.0 + 1e-9):
                    _j += 1
                pos_idx.append(min(_j, len(pos_all) - 1))
            real_gaps = [abs(math.log(max(pos_all[t], 1e-8))
                             - math.log(max(pos_all[t + 1], 1e-8)))
                         for t in range(len(pos_all) - 1)]
            for i in range(len(grid) - 1):
                a, b = pos_idx[i], pos_idx[i + 1]
                if b <= a:
                    lo, hi = max(a - 1, 0), min(a + 1, len(pos_all) - 1)
                else:
                    lo, hi = a, b
                seg = real_gaps[lo:hi]
                real_gap_curve.append(sum(seg) / len(seg) if seg
                                      else (sum(real_gaps) / len(real_gaps)))
        else:
            real_gap_curve = list(gap_curve)
        space = _detect_space(torch.tensor(grid), cfg.get("noise_space", "auto"))
        xs = []
        x0s = []
        cur = px
        _dev_retry_done = False
        for i, s in enumerate(grid):
            _progress(callback, "cond_probe", "progress", i, len(grid) + 4,
                      "A-FloPS focused probe: step %d/%d" % (i + 1, len(grid)))
            x0 = None
            try:
                x0 = _probe_model_call(model, cur, s * ones, extra_args)
            except Exception as e:
                if (not _dev_retry_done and _pf_patcher is not None
                        and _is_device_error(e)):
                    _dev_retry_done = True
                    try:
                        _dev = _probe_call_device(_pf_patcher)
                        if cur.device != _dev:
                            cur = cur.to(_dev)
                        if ones.device != _dev:
                            ones = ones.to(_dev)
                        x0 = _probe_model_call(model, cur, s * ones, extra_args)
                        logging.info("[A-FloPS-focused] recovered from a "
                                     "device mismatch at s=%.4g (model "
                                     "re-secured on %s)", s, _dev)
                    except Exception:
                        x0 = None
                if x0 is None:
                    import traceback as _tb
                    logging.warning("[A-FloPS-focused] model call failed at s=%.4g: %s", s, e)
                    logging.warning("[A-FloPS-focused] failure traceback:\n%s", _tb.format_exc())
                    logging.warning("[A-FloPS-focused] failure context: %s",
                                    _probe_failure_diagnostics(model, cur, ones, extra_args, gap_ctx))
                    break
            xs.append(cur)
            x0s.append(x0.detach())
            if i < len(grid) - 1:
                cur = exact_step(cur, x0, s, grid[i + 1])
        m = len(x0s)
        if m < 3:
            return None
        norms = [float(t.float().norm().clamp_min(1e-8)) for t in x0s]
        jumps = []
        for i in range(m - 1):
            d = float((x0s[i + 1] - x0s[i]).float().norm())
            jumps.append(d / max(max(norms[i], norms[i + 1]), 1e-8))
        curvs = []
        for i in range(1, m - 1):
            d1 = (x0s[i + 1] - x0s[i]).float().norm()
            d2 = (x0s[i] - x0s[i - 1]).float().norm()
            curvs.append(abs(float(d1) - float(d2)) / max(float(d1), float(d2), 1e-8))
        local_errs = []
        local_errs_real = []
        for i in range(m - 1):
            if i + 1 < m and i < len(xs) - 1:
                x_pred_mid = 0.5 * (xs[i] + xs[i + 1])
                x0_at_mid = x0s[i]
                err = float((x0s[i + 1].float() - x0_at_mid.float()).norm()
                            / max(norms[i + 1], 1e-8))
                local_errs.append(err)
                gp = gap_curve[i] if i < len(gap_curve) else 1e-3
                gr = real_gap_curve[i] if i < len(real_gap_curve) else gp
                if gp > 1e-9 and gr > 1e-9:
                    scale = min(max((gr / gp) ** 2, 0.04), 1.0)
                    local_errs_real.append(err * scale)
                else:
                    local_errs_real.append(err)
        guard_sigs = []
        for i in range(m - 1):
            sig = _jump_signal(x0s[i + 1], x0s[i], xs[i])
            guard_sigs.append(sig)
        gap = None
        probe_cfg = None
        if gap is None and gap_ctx is not None and bool(cfg.get("cond_probe_gap", True)):
            try:
                probe_cfg = float(cfg.get("cond_probe_cfg"))
            except (TypeError, ValueError):
                probe_cfg = None
            if probe_cfg is not None and probe_cfg > 1.0 + 1e-6:
                gap = _measure_guidance_gap(model, extra_args, xs, x0s,
                                            grid, ones, gap_ctx=gap_ctx)
        _fp_cfg = _probe_effective_cfg(model, extra_args, gap_ctx)
        if _fp_cfg is None:
            _fp_cfg = probe_cfg
        is_distilled_eff = _resolve_distilled_override(
            cfg.get("cond_probe_is_distilled", "auto"), _fp_cfg, False)
        endgame = None
        if isinstance(base_profile, dict):
            endgame = base_profile.get("endgame")
        if endgame is None and bool(cfg.get("probe_endgame", True)):
            _progress(callback, "cond_probe", "progress", len(grid), len(grid) + 4,
                      "A-FloPS focused probe: endgame")
            endgame = _probe_endgame(model, extra_args, x0s[-1], grid[-1],
                                     pos_all, ones, space, _probe_gen(x))
        profile = {
            "version": 7,
            "cond_sig": cfg.get("cond_probe_sig"),
            "space": space,
            "n_probe_steps": m,
            "sigma_curve": [round(v, 6) for v in grid],
            "jump_curve": [round(v, 6) for v in jumps],
            "gap_curve": [round(v, 6) for v in gap_curve],
            "real_gap_curve": [round(v, 6) for v in real_gap_curve],
            "local_err_curve": [round(v, 6) for v in local_errs],
            "med_local_err_real": (sorted(local_errs_real)[len(local_errs_real) // 2]
                                   if local_errs_real else 0.0),
            "curv_curve": [round(v, 6) for v in curvs],
            "med_jump": (sorted(jumps)[len(jumps) // 2] if jumps else 0.0),
            "med_curv": (sorted(curvs)[len(curvs) // 2] if curvs else 0.0),
            "is_distilled_eff": is_distilled_eff,
            "med_local_err": (sorted(local_errs)[len(local_errs) // 2]
                              if local_errs else 0.0),
            "guard_sig_med": (sorted(guard_sigs)[len(guard_sigs) // 2]
                              if guard_sigs else 0.0),
            "guard_sig_p90": (sorted(guard_sigs)[int(len(guard_sigs) * 0.9)]
                              if guard_sigs else 0.0),
            "log_gap_probe": (sum(gap_curve) / len(gap_curve) if gap_curve else 0.0),
            "log_gap_real": (sum(real_gap_curve) / len(real_gap_curve)
                             if real_gap_curve else 0.0),
            "gap": gap,
            "probe_cfg": probe_cfg if gap_ctx is not None else None,
            "n_jump": len(jumps),
            "n_err": len(local_errs),
            "n_guard": len(guard_sigs),
            "endgame": endgame,
            "whitepoint": _whitepoint_from_x0s(x0s),
            "focused": True,
            "fragile_ranges": [(round(s_lo, 6), round(s_hi, 6), round(f, 3))
                               for s_lo, s_hi, f in (fragile_ranges or [])],
        }
        profile["trajectory_state"] = _compute_trajectory_state(profile)
        profile["calib"] = _derive_cond_calibrations(profile)
        return profile
    except Exception as e:
        logging.warning("[A-FloPS-focused] probe failed: %s", e)
        return None

def _run_cond_probe(model, x, sigmas, extra_args, cfg, gap_ctx=None,
                    callback=None):
    try:
        _pf_patcher = gap_ctx.get("patcher") if isinstance(gap_ctx, dict) else None
        if _pf_patcher is not None:
            _pf_dev = _probe_call_device(_pf_patcher)
            if x.device != _pf_dev:
                x = x.to(_pf_dev)
        pres = max(8, int(cfg.get("cond_probe_resolution", 16)))
        K = max(3, int(cfg.get("cond_probe_steps", 4)))
        B = 1
        C = x.shape[1]
        if B < 1 or C < 1:
            return None
        grid = _probe_grid(sigmas, K, _probe_distilled_mode(cfg))
        if grid is None:
            return None
        pos_all = [float(v) for v in sigmas if float(v) > 1e-6]
        gap_curve = [abs(math.log(max(grid[i], 1e-8))
                         - math.log(max(grid[i + 1], 1e-8)))
                     for i in range(len(grid) - 1)]
        real_gap_curve = []
        if len(pos_all) >= 2:
            pos_idx = []
            _j = 0
            for gv in grid:
                while _j < len(pos_all) and pos_all[_j] > gv * (1.0 + 1e-9):
                    _j += 1
                pos_idx.append(min(_j, len(pos_all) - 1))
            real_gaps = [abs(math.log(max(pos_all[t], 1e-8))
                             - math.log(max(pos_all[t + 1], 1e-8)))
                         for t in range(len(pos_all) - 1)]
            for i in range(len(grid) - 1):
                a, b = pos_idx[i], pos_idx[i + 1]
                if b <= a:
                    lo, hi = max(a - 1, 0), min(a + 1, len(pos_all) - 1)
                else:
                    lo, hi = a, b
                seg = real_gaps[lo:hi]
                real_gap_curve.append(sum(seg) / len(seg) if seg
                                      else (sum(real_gaps) / len(real_gaps)))
        else:
            real_gap_curve = list(gap_curve)
        space = _detect_space(torch.tensor(grid), cfg.get("noise_space", "auto"))
        _rank_ladder = _probe_rank_ladder(_pf_patcher, x)
        _rank_used = None
        _ladder_i = 0
        while True:
            _rank = _rank_ladder[min(_ladder_i, len(_rank_ladder) - 1)]
            shape = _probe_latent_shape(ref_x=x, pres=pres, batch=B,
                                        rank=_rank)
            px = torch.randn(shape, device=x.device, dtype=torch.float32,
                             generator=_probe_gen(x)).to(x.dtype)
            ones = px.new_ones([B])
            px = px * float(grid[0])
            xs = []
            x0s = []
            cur = px
            _dev_retry_done = False
            _shape_retry = False
            for i, s in enumerate(grid):
                _progress(callback, "cond_probe", "progress", i, len(grid) + 4,
                          "A-FloPS cond probe: step %d/%d" % (i + 1, len(grid)))
                x0 = None
                try:
                    x0 = _probe_model_call(model, cur, s * ones, extra_args)
                except Exception as e:
                    if (not _dev_retry_done and _pf_patcher is not None
                            and _is_device_error(e)):
                        _dev_retry_done = True
                        try:
                            _dev = _probe_call_device(_pf_patcher)
                            if cur.device != _dev:
                                cur = cur.to(_dev)
                            if ones.device != _dev:
                                ones = ones.to(_dev)
                            x0 = _probe_model_call(model, cur, s * ones, extra_args)
                            logging.info("[A-FloPS-cond-probe] recovered from a "
                                         "device mismatch at s=%.4g (model "
                                         "re-secured on %s)", s, _dev)
                        except Exception:
                            x0 = None
                    if x0 is None:
                        if (_ladder_i + 1 < len(_rank_ladder)
                                and _is_shape_error(e)):
                            _shape_retry = True
                            logging.info(
                                "[A-FloPS-cond-probe] model rejected a %d-D "
                                "probe latent at s=%.4g (%s); re-running the "
                                "trajectory at %d-D",
                                cur.dim(), s, e, _rank_ladder[_ladder_i + 1])
                            break
                        import traceback as _tb
                        logging.warning("[A-FloPS-cond-probe] model call failed at s=%.4g: %s", s, e)
                        logging.warning("[A-FloPS-cond-probe] failure traceback:\n%s", _tb.format_exc())
                        logging.warning("[A-FloPS-cond-probe] failure context: %s",
                                        _probe_failure_diagnostics(model, cur, ones, extra_args, gap_ctx))
                        break
                xs.append(cur)
                x0s.append(x0.detach())
                if i < len(grid) - 1:
                    cur = exact_step(cur, x0, s, grid[i + 1])
            if not _shape_retry:
                _rank_used = _rank
                break
            _ladder_i += 1
        m = len(x0s)
        if m < 3:
            return None
        if _rank_used is not None:
            _remember_probe_rank(_pf_patcher, _rank_used)
        norms = [float(t.float().norm().clamp_min(1e-8)) for t in x0s]
        jumps = []
        for i in range(m - 1):
            d = float((x0s[i + 1] - x0s[i]).float().norm())
            jumps.append(d / max(max(norms[i], norms[i + 1]), 1e-8))
        dist_on = bool(cfg.get("probe_distributions", True))
        jump_q = []
        ojf_curve = []
        band_curve = []
        jrel_curve = []
        dabs_curve = []
        if dist_on:
            _qt = torch.tensor([0.1, 0.5, 0.9])
            for i in range(m - 1):
                jpx = (x0s[i + 1] - x0s[i]).float().pow(2).mean(dim=1).sqrt()
                sc = max(float(x0s[i].float().pow(2).mean().sqrt()),
                         float(x0s[i + 1].float().pow(2).mean().sqrt()))
                jpx_n = (jpx / max(sc, 1e-8)).reshape(-1)
                qs = torch.quantile(jpx_n, _qt.to(jpx_n.device))
                jump_q.append([round(float(v), 6) for v in qs])
                p50 = float(qs[1])
                if p50 > 1e-9:
                    ojf_curve.append(round(
                        float((jpx_n > 2.5 * p50).float().mean()), 4))
                try:
                    band_curve.append([round(v, 4) for v in
                                       _lf_band_fractions(jpx)])
                except Exception:
                    band_curve.append([0.33, 0.34, 0.33])
                # runtime-scale per-pixel relative jump: the SAME quantity the
                # escape's stagnation detector uses (median jump / median
                # per-pixel magnitude).  Stored so the escape threshold can be
                # derived from the model's own measured jump level.
                xm = x0s[i].float().pow(2).mean(dim=1).sqrt()
                jrel_curve.append(round(
                    float(jpx.median().clamp_min(0.0)
                          / xm.median().clamp_min(1e-8)), 6))
            for i in range(m):
                x4 = x0s[i].float().reshape(-1, x0s[i].shape[1],
                                            *x0s[i].shape[-2:])
                hp = x4 - F.avg_pool2d(x4, 3, stride=1, padding=1)
                dabs_curve.append(round(
                    float(hp.norm() / x4.norm().clamp_min(1e-8)), 6))
        med_jump_rel = (sorted(jrel_curve)[len(jrel_curve) // 2]
                        if jrel_curve else None)
        med_detail_abs = (sorted(dabs_curve)[len(dabs_curve) // 2]
                          if dabs_curve else None)
        dir_cons = []
        if bool(cfg.get("probe_dir_cons", True)):
            for i in range(1, m - 1):
                d1 = (x0s[i] - x0s[i - 1]).float()
                d2 = (x0s[i + 1] - x0s[i]).float()
                num = (d1 * d2).sum(dim=1)
                den = d1.pow(2).sum(dim=1).sqrt() * d2.pow(2).sum(dim=1).sqrt()
                okm = den > 1e-8
                if bool(okm.any()):
                    dir_cons.append(round(float((num[okm] / den[okm]).mean()), 4))
        dir_cons_valid = []
        if dir_cons and len(gap_curve) >= 2:
            _med_g = sorted(gap_curve)[len(gap_curve) // 2]
            for _k in range(len(dir_cons)):
                _ga, _gb = gap_curve[_k], gap_curve[_k + 1]
                if (_med_g / 3.0) <= _ga <= 3.0 * _med_g and \
                        (_med_g / 3.0) <= _gb <= 3.0 * _med_g:
                    dir_cons_valid.append(dir_cons[_k])
            if len(dir_cons_valid) < 2:
                dir_cons_valid = []
        # oscillation onset: the sigma where the x0 trajectory first reverses
        # direction (dir_cons < 0) -- where the late noise fade should anchor
        # (the noise stops helping and starts oscillating the trajectory).
        osc_sigma = None
        for _k, _dc in enumerate(dir_cons):
            if _dc < 0 and _k + 1 < len(grid):
                osc_sigma = float(grid[_k + 1])
                break
        curvs = []
        for i in range(1, m - 1):
            dl_prev = math.log(max(grid[i - 1], 1e-8)) - math.log(max(grid[i], 1e-8))
            dl_next = math.log(max(grid[i], 1e-8)) - math.log(max(grid[i + 1], 1e-8))
            dl = 0.5 * (abs(dl_prev) + abs(dl_next))
            if dl < 1e-6:
                continue
            dd = (x0s[i + 1] - 2.0 * x0s[i] + x0s[i - 1]).float().norm()
            curvs.append(float(dd) / (dl * dl)
                         / max(max(norms[i - 1], norms[i], norms[i + 1]), 1e-8))
        local_errs = []
        err_q = []
        guard_sigs = []
        frag_acc = None
        for i in range(1, m - 1):
            s_i = float(grid[i])
            s_n = float(grid[i + 1])
            s_p = float(grid[i - 1])
            if s_n <= 1e-8 or s_i <= 1e-8:
                continue
            x1_i = exact_step(xs[i], x0s[i], s_i, s_n)
            ls = math.log(max(s_i, 1e-8))
            ls2 = math.log(max(s_n, 1e-8))
            lsp = math.log(max(s_p, 1e-8))
            lmid = 0.5 * (ls + ls2)
            t = (lmid - ls) / (lsp - ls)
            x0_mid = x0s[i] + t * (x0s[i - 1] - x0s[i])
            x_pred_i = exact_step(xs[i], x0_mid, s_i, s_n)
            local_errs.append(float((x_pred_i - x1_i).float().norm()
                                    / x1_i.float().norm().clamp_min(1e-8)))
            if dist_on:
                epx = ((x_pred_i - x1_i).float().pow(2).mean(dim=1).sqrt()
                       / float(x1_i.float().pow(2).mean().sqrt().clamp_min(1e-8)))
                eq = torch.quantile(epx.reshape(-1),
                                    torch.tensor([0.1, 0.5, 0.9],
                                                 device=epx.device))
                err_q.append([round(float(v), 6) for v in eq])
                # per-pixel fragility: max local error over the trajectory
                if frag_acc is None:
                    frag_acc = epx.detach().float().unsqueeze(1)
                else:
                    frag_acc = torch.maximum(frag_acc, epx.detach().float().unsqueeze(1))
        local_errs_real = []
        _err_probe_gaps = gap_curve[1:1 + len(local_errs)]
        _err_real_gaps = real_gap_curve[1:1 + len(local_errs)]
        for _e, _gp, _gr in zip(local_errs, _err_probe_gaps, _err_real_gaps):
            if _gp > 1e-9 and _gr > 1e-9:
                local_errs_real.append(
                    _e * min(max((_gr / _gp) ** 2, 0.04), 1.0))
            else:
                local_errs_real.append(_e)
        for i in range(1, m):
            guard_sigs.append(float(_jump_signal(x0s[i], x0s[i - 1], xs[i])))
        g_pos = [max(float(v), 1e-8) for v in grid]
        gap_probe = (sum(abs(math.log(g_pos[i]) - math.log(g_pos[i + 1]))
                         for i in range(len(g_pos) - 1))
                     / max(1, len(g_pos) - 1))
        pos_all = [float(v) for v in sigmas if float(v) > 1e-6]
        gap_real = 0.0
        if len(pos_all) > 1:
            gap_real = (sum(abs(math.log(max(pos_all[i], 1e-8))
                                - math.log(max(pos_all[i + 1], 1e-8)))
                            for i in range(len(pos_all) - 1))
                        / (len(pos_all) - 1))

        def _pct(arr, p):
            if not arr:
                return 0.0
            a = sorted(arr)
            return a[min(len(a) - 1, int(p * len(a)))]

        med_jump = _pct(jumps, 0.5)
        med_curv = _pct(curvs, 0.5)
        early = jumps[: max(1, len(jumps) // 3)]
        early_frac = sum(early) / (sum(jumps) + 1e-12)
        conv_step = len(jumps)
        thr = 0.05 * (max(jumps) if jumps else 1.0)
        for i, jv in enumerate(jumps):
            if jv < thr:
                conv_step = i
                break
        # Distilled = the x0 trajectory stops moving within the first few
        # steps AND most of its movement happens in that front window.  We do
        # NOT flag "last jump tiny relative to the max" as distilled: any
        # smooth fast-converging model (jump ~ sigma^p with p > 1) has a
        # small final jump once the grid reaches low sigma, so that clause
        # turned every cfg > 1 flow model into a false distilled positive.
        # The cfg override in _resolve_distilled_override already catches
        # true cfg <= 1 distilled models.
        is_distilled_eff = bool(early_frac > 0.75 and
                                conv_step <= max(2, len(jumps) // 3))
        probe_cfg = _probe_effective_cfg(model, extra_args, gap_ctx)
        if probe_cfg is None:
            try:
                probe_cfg = float(cfg.get("cond_probe_cfg"))
            except (TypeError, ValueError):
                probe_cfg = None
        is_distilled_eff = _resolve_distilled_override(
            cfg.get("cond_probe_is_distilled", "auto"),
            probe_cfg,
            is_distilled_eff)
        gap = None
        if bool(cfg.get("cond_probe_gap", True)):
            if probe_cfg is not None and probe_cfg <= 1.0 + 1e-6:
                gap = None
            else:
                gap = _measure_guidance_gap(model, extra_args, xs, x0s,
                                            grid, ones, gap_ctx=gap_ctx)
        endgame = None
        if bool(cfg.get("probe_endgame", True)):
            _progress(callback, "cond_probe", "progress", len(grid), len(grid) + 4,
                      "A-FloPS cond probe: endgame")
            endgame = _probe_endgame(model, extra_args, x0s[-1], grid[-1],
                                     pos_all, ones, space, _probe_gen(x))
        profile = {
            "version": 7,
            "cond_sig": cfg.get("cond_probe_sig"),
            "space": space,
            "n_probe_steps": m,
            "sigma_curve": [round(v, 6) for v in grid],
            "jump_curve": [round(v, 6) for v in jumps],
            "gap_curve": [round(v, 6) for v in gap_curve],
            "real_gap_curve": [round(v, 6) for v in real_gap_curve],
            "local_err_curve": [round(v, 6) for v in local_errs],
            "slope_curve": [round(j / max(gv, 1e-9), 8)
                            for j, gv in zip(jumps, gap_curve)],
            "med_local_err_real": (_pct(local_errs_real, 0.5)
                                   if local_errs_real else 0.0),
            "curv_curve": [round(v, 6) for v in curvs],
            "med_jump": med_jump,
            "med_curv": med_curv,
            "med_jump_rel": med_jump_rel,
            "med_detail_abs": med_detail_abs,
            "early_frac": early_frac,
            "conv_step": conv_step,
            "conv_sigma": (round(float(grid[int(conv_step)]), 6)
                           if 0 <= conv_step < len(grid) else None),
            "osc_sigma": (round(float(osc_sigma), 6)
                          if osc_sigma is not None else None),
            "is_distilled_eff": is_distilled_eff,
            "med_local_err": (_pct(local_errs, 0.5) if local_errs else 0.0),
            "guard_sig_med": (_pct(guard_sigs, 0.5) if guard_sigs else 0.0),
            "guard_sig_p90": (_pct(guard_sigs, 0.9) if guard_sigs else 0.0),
            "log_gap_probe": gap_probe,
            "log_gap_real": gap_real,
            "gap": gap,
            "probe_cfg": probe_cfg,
            "n_jump": len(jumps),
            "n_err": len(local_errs),
            "n_guard": len(guard_sigs),
            "jump_q": jump_q,
            "err_q": err_q,
            "fragility": _fragility_map(frag_acc),
            "ojf_curve": ojf_curve,
            "band_curve": band_curve,
            "dir_cons": dir_cons,
            "dir_cons_valid": dir_cons_valid,
            "endgame": endgame,
            "whitepoint": _whitepoint_from_x0s(x0s),
        }
        profile["trajectory_state"] = _compute_trajectory_state(profile)
        profile["calib"] = _derive_cond_calibrations(profile)
        return profile
    except Exception as e:
        logging.warning("[A-FloPS-cond-probe] probe failed: %s", e)
        return None

def _apply_cond_probe_profile(cfg, cond_profile, tune, weight, auto_tunable):
    if not cond_profile:
        return
    cond_calib = cond_profile.get("calib", {})
    w = max(0.0, min(float(weight), 1.0))
    if w <= 0.0:
        return

    def _blend_set(key):
        if key not in auto_tunable:
            return
        cond_val = cond_calib.get(key)
        if cond_val is None:
            return
        try:
            cur = float(cfg.get(key))
            cond_f = float(cond_val)
            blended = (1.0 - w) * cur + w * cond_f
            if cur > 0 and blended > 0 and key != "warmup":
                blended = max(0.25 * cur, min(4.0 * cur, blended))
            cfg[key] = blended
        except (TypeError, ValueError):
            pass

    if tune.get("guard", True):
        _blend_set("guard_floor")
    if tune.get("tol", True):
        _blend_set("rtol")
    if tune.get("wmax", True):
        _blend_set("wmax_base")
    if tune.get("eta", True):
        cond_se = cond_calib.get("safe_eta")
        if cond_se is not None and cfg.get("eta") is not None:
            if "eta" in auto_tunable:
                cfg["eta"] = min(float(cfg["eta"]), float(cond_se))
    if cond_profile.get("is_distilled_eff"):
        _blend_set("warmup")
    if tune.get("anomaly", True):
        for _ak in ("an_from", "an_spatial", "an_novelty", "an_value"):
            _blend_set(_ak)



def _build_curvature_schedule(profile, steps, denoise=1.0, p=2.0, shift=1.0,
                              base_sigmas=None):
    """Fine-tune the user's sigma list with the probe's curvature.

    Base math (rig-fitted on CFG5log/CFG1log): the aflops integrator steps in
    u = log(sigma) (its du), so its consistency error is second order in du:
    err ~ C * curv * du^2  (rig: curvature exponent a ~= 0.9 pooled with
    p = 2 fixed by the integrator order).  Equal-error placement therefore
    uses density ~ curv^(1/p), p = 2 -> sqrt(curv), biased by the user's
    shift:  (1 + (shift-1)*sigma).

    The density lives on the external list's INDEX axis: flat curvature (or
    a degenerate profile) reproduces the user's list exactly -- the probe
    only re-allocates steps WITHIN the schedule the user plugged in.  The
    native-time and log-sigma builds were removed with the Scheduler node
    (git history keeps them); base_sigmas is the only path.

    Anchor semantics follow the probe source: curv_curve[k] is the curvature
    at sigma_curve[k+1] (not at the interval midpoint).

    Returns a descending sigma tensor [start ... 0.0], or None."""
    try:
        curv = profile.get("curv_curve")
        sigc = profile.get("sigma_curve")
        if not curv or not sigc or len(curv) < 3 or len(sigc) < 4:
            return None
        steps = int(steps)
        if steps < 4 or float(denoise) <= 0.0:
            return None
        shift = max(float(shift), 1e-3)
        n = min(len(curv), len(sigc) - 1)
        anchors = [(float(sigc[k + 1]), max(float(curv[k]), 1e-9))
                   for k in range(n)]
        if max(c for _, c in anchors) <= 1e-9:
            return None
        # Degeneracy guard: the probe's own bar (_probe_degenerate) says a
        # profile with med_curv < 1e-4 carries no usable dynamics evidence.
        # A numerically-flat trajectory (distilled model at cfg 1, measured
        # curv ~1e-6 with zeros clamped to 1e-9) still passes the max>1e-9
        # test above, and its 1e-6-vs-1e-9 noise contrast drives a 1000x
        # density swing: measured log (CFG1) -> one starved 0.39 sigma gap
        # (2.3x the input schedule's max gap) and a 109x corrector overshoot.
        # Reject such profiles: the schedule then passes through untouched.
        _med_c = sorted(c for _, c in anchors)[len(anchors) // 2]
        if _med_c < 1e-4:
            return None

        def curv_at(s):
            if s >= anchors[0][0]:
                return anchors[0][1]
            if s <= anchors[-1][0]:
                return anchors[-1][1]
            for (s1, c1), (s2, c2) in zip(anchors, anchors[1:]):
                if s2 <= s <= s1:
                    l1, l2 = math.log(s1), math.log(s2)
                    l = math.log(s)
                    t = (l - l1) / (l2 - l1) if l2 != l1 else 0.0
                    return math.exp(math.log(c1)
                                    + t * (math.log(c2) - math.log(c1)))
            return anchors[-1][1]

        try:
            _bs = [float(v) for v in base_sigmas]
        except Exception:
            _bs = None
        if not _bs or len(_bs) < 5:
            return None
        _bn = len(_bs) - 1
        # steps is derived from the external list; denoise is the user's
        # list as-is (they slice it with their own nodes).
        _map = _bs[:-1]          # N boundary sigmas, descending
        _tail = _bs[-1]          # the user's own endpoint (usually 0)
        if _map[0] <= _map[-1] or _map[0] <= 0.0:
            return None

        def sig_of_u(u):
            f = min(max(float(u), 0.0), float(_bn - 1))
            i = int(f)
            t = f - i
            if i >= _bn - 1:
                return _map[-1]
            return _map[i] + t * (_map[i + 1] - _map[i])

        M = 800
        us = [_bn * (i / M) for i in range(M + 1)]
        ws = [((curv_at(sig_of_u(u)) ** (1.0 / p))
               * (1.0 + (shift - 1.0) * sig_of_u(u))) for u in us]
        cum = [0.0]
        for i in range(1, M + 1):
            cum.append(cum[-1] + 0.5 * (ws[i - 1] + ws[i])
                       * (us[i] - us[i - 1]))
        tot_w = cum[-1]
        if tot_w <= 1e-12:
            return None
        tgt = [tot_w * (j / _bn) for j in range(_bn)]
        out = []
        j = 0
        for i in range(1, M + 1):
            while j < len(tgt) and cum[i] >= tgt[j]:
                den = cum[i] - cum[i - 1]
                frac = (tgt[j] - cum[i - 1]) / den if den > 1e-12 \
                    else 0.0
                out.append(sig_of_u(us[i - 1]
                                    + frac * (us[i] - us[i - 1])))
                j += 1
        if len(out) < _bn:
            return None
        out = [round(float(v), 6) for v in out] + [round(_tail, 6)]
        out[0] = round(_map[0], 6)
        grid = torch.tensor(out, dtype=torch.float32)
        if grid.numel() != _bn + 1:
            return None
        return grid
    except Exception:
        return None


_DWELL_OV_GATE = 0.25     # below this measured oversteer the dwell never arms
_DWELL_OV_REF = 0.55      # oversteer at which the dwell reaches its cap
_DWELL_CAP = 0.20         # strength ceiling (fraction of a step withheld)
_DWELL_R_HI = 2.0         # flux ratio (vs image median) at full dwell weight
_DWELL_EMA = 0.7          # decay of the per-pixel flux envelope
_DWELL_CURV_FLOOR = 1e-4  # max curvature below this = degenerate probe (no dwell)


def _dwell_env(kn, ramp_end, fade_start, fade_end):
    """Phase envelope of the per-pixel dwell: ramp in over the first steps,
    hold through the measured active zone, fade out before the endgame."""
    t_in = _sstep((float(kn) - 0.04) / max(float(ramp_end) - 0.04, 1e-6))
    t_out = 1.0 - _sstep((float(kn) - float(fade_start)) /
                         max(float(fade_end) - float(fade_start), 1e-6))
    return t_in * t_out


def _recommend_dwell_from_profile(prof_m, prof_c, sigmas, lf_evid,
                                  fragile_ranges):
    """L1 per-pixel dwell: derive (strength, phase window) from probe evidence.

    Same evidence philosophy as the shift-schedule recommender and the local
    field's auto-gains, and it costs zero extra model calls: the probe
    decides HOW MUCH (the measured oversteer evidence scales the strength
    ceiling -- a model that does not overstep structure decisions never
    arms the dwell) and WHEN (the curvature curve's active zone places the
    fade-out; the endgame fragile band is a hard stop, so no new lag is
    ever introduced where the probe measured the model misreads structured
    state).  WHERE is decided at runtime by the per-pixel churn map --
    no probe can know which areas of THIS image will lag.  Everything here
    is derived from profile evidence; there are no architecture branches.

    Returns None (keep the exact classic full-step behavior) or
    {"dwell_max", "ramp_end", "fade_end", "ov"}.
    """
    try:
        if not isinstance(lf_evid, dict) or lf_evid.get("source") in (None, "none"):
            return None
        ov = min(max(float(lf_evid.get("ov", 0.0) or 0.0), 0.0), 1.0)
        if ov < _DWELL_OV_GATE:
            return None
        # finest curvature curve available drives the envelope (per-prompt
        # first, the same precedence as the shift schedule)
        prof = None
        for p in (prof_c, prof_m):
            if (isinstance(p, dict)
                    and len(p.get("curv_curve") or []) >= 3
                    and not _probe_degenerate(p)):
                prof = p
                break
        if prof is None:
            return None
        s = sigmas.detach().float()
        if s.numel() < 7:
            return None
        curv = [float(v) for v in (prof.get("curv_curve") or [])]
        sigc = prof.get("sigma_curve") or []
        n = min(len(curv), max(0, len(sigc) - 1))
        if n < 3:
            return None
        pts = sorted((math.log(max(float(sigc[i + 1]), 1e-8)), curv[i])
                     for i in range(n))
        cmax = max(c[1] for c in pts)
        if cmax <= _DWELL_CURV_FLOOR:
            return None
        # low-sigma edge of the measured active zone: scanning toward lower
        # sigma, structure decisions end where the curvature rises quiet ->
        # active; sig_q = the first (lowest) active sigma, so the dwell is
        # gone by the time the run passes the zone's edge (mapped onto this
        # run's grid as a fraction of steps)
        sv = [float(v) for v in s if float(v) > 1e-6]
        n_steps = len(sv)
        sig_q = None
        for i in range(1, len(pts)):
            if pts[i][1] >= 0.35 * cmax > pts[i - 1][1]:
                sig_q = math.exp(pts[i][0])
                break
        fade_end = 0.70
        if sig_q is not None and n_steps > 0:
            kn_q = next((i / float(n_steps) for i, v in enumerate(sv)
                         if v <= sig_q), 1.0)
            fade_end = kn_q
        fade_end = min(max(fade_end, 0.45), 0.75)
        # probe-measured endgame fragile band: hard stop before it
        try:
            rs, _rf = _rescue_window_from_fragile(s, fragile_ranges or [])
            if rs is not None:
                fade_end = min(fade_end, max(0.40, rs - 0.05))
        except Exception:
            pass
        strength = _DWELL_CAP * min(1.0, ov / _DWELL_OV_REF)
        return {"dwell_max": round(strength, 4),
                "ramp_end": round(min(0.30, 0.5 * fade_end), 4),
                "fade_end": round(fade_end, 4),
                "ov": round(ov, 4)}
    except Exception:
        return None


ENGINE_DEFAULTS = {
    "order": 2, "extrap_space": "auto",
    # Noise defaults carry the tuned values the removed Options nodes used to
    # set (rig: old-vs-new log diff, CFG5/CFG1).  The old "massive
    # improvement" runs had eta_mode=adaptive + s_tmin=1.0 -> eta_eff was 0.0
    # for EVERY step (escape detection computed, injection fully gated); the
    # redesigned defaults fell back to cosine + s_tmin=0.0 -> 0.1 eta noise on
    # every step -> washed-out structure.  Restoring the tuned values as the
    # DEFAULT keeps the autotuner's "auto = good" contract.
    "eta_mode": "adaptive", "falloff": 2.0, "eta_min": 0.0,
    "eta_escape_jump": -1.0,      # <0 = auto: derived from the probe's med_jump_rel
    "eta_escape_detail": -1.0,    # <0 = auto: derived from the probe's med_detail_abs
    "eta_fade_sigma": -1.0,       # <0 = auto: derived from the probe's conv_sigma
    "eta_fade_floor": 0.25,       # fraction of the baseline eta retained after the late fade
    "wmax_base": 2.0,
    "wmax_adapt": True,
    "wmax_adapt_k": 0.5,
    "wmax_mid_floor": 1.55,
    "s_tmin": 1.0, "s_tmax": 0.0,
    "rtol": 0.01,
    "err_mode": "rel_l2",
    "grad_strength": 5.0,
    "aflops_lam_min": -3.0,
    "aflops_lam_max": 0.0,
    "x0m_env": 1.3,
    "outlier_guard": True, "guard_sensitivity": 2.5, "guard_hold": 2,
    "guard_window": 8, "guard_floor": 0.10, "guard_ratio": 3.0,
    "guard_quantile": 0.99,
    "guard_mad_floor": 0.1,
    "guard_phase_relax": 1.0,
    "an_enable": True, "an_from": 0.15, "an_win": 9,
    "an_spatial": 5.0, "an_novelty": 4.0, "an_value": 2.5,
    "an_phase_relax": 0.5,
    "an_region_z": 8.0,
    "lf_enable": True,
    "lf_guard_feed": True,
    "lf_closed_loop": True,
    "lf_runtime_gate": True,
    "lf_env_fade": True,
    "lf_spatial_health": True,
    "lf_band_refresh": True,
    "lf_dir_vol": True,
    "lf_semantic_prior": True,
    "lf_detail_rescue": True,
    "lf_dir_real_feedback": False,
    "lf_drift_budget": 0.0,
    "rescue_cap": 0.35,      # rescue envelope = cap * clamp (<= ~5%/step)
    "rescue_start": 0.55,    # kn where detection arms
    "rescue_full": 0.72,     # kn where rescue authority reaches full
    "rescue_max_cov": 0.35,  # coverage fraction where the gate shuts off
    "rescue_floor": 0.02,    # min block delta magnitude, fraction of sigma
    "dwell_mode": "auto",   # per-pixel dwell: off | auto | manual (auto arms only with probe evidence)
    "dwell_max": 0.20,       # manual: max fraction of a step a pixel withholds
    "dwell_until": 0.65,     # manual: run fraction where the dwell window ends
    "probe_distributions": True,
    "probe_endgame": True,
    "probe_dir_cons": True,
    "eta": None, "s_noise": 1.0, "noise_space": "auto",
    "noise_color": "auto", "noise_beta": 1.0,
    "color_schedule": "constant",
    "color_beta_max": 1.5, "color_beta_min": -0.5,
    "color_temporal": False, "color_temporal_beta": 0.0,
    "color_decay_floor": 1e-3, "color_dc_zero": True,
    "color_strength": 1.0,
    "adaptive_tau_eta": 0.15,
    "adaptive_tau_err": 0.5,
    "adaptive_tau_snr": 2.0,
    "adaptive_cons_gamma": 1.5,
    "color_conc_gate": True,
    "color_conc_strength": 1.0,
    "_model_info": None,
    "probe_enabled": False,
    "probe_steps": 10,
    "probe_resolution": 48,
    "probe_force": False,
    "probe_tune_guard": True,
    "probe_tune_tol": True,
    "probe_tune_wmax": True,
    "probe_tune_anomaly": True,
    "probe_tune_eta": True,
    "probe_tune_lf": True,
    "cond_probe_enabled": False,
    "cond_probe_steps": 4,
    "cond_probe_resolution": 16,
    "cond_probe_gap": True,
    "cond_probe_weight": 0.5,
    "cond_probe_tune_guard": True,
    "cond_probe_tune_tol": True,
    "cond_probe_tune_wmax": True,
    "cond_probe_tune_anomaly": True,
    "cond_probe_tune_eta": True,
    "cond_probe_tune_lf": True,
    "cond_probe_force": False,
    "cond_probe_cfg": 7.0,
    "cond_probe_focused": True,
    "history_consistency_gate": True,
    "history_consistency_threshold": 0.0,
    "order_pixel_blend": True,
    "order_pixel_blend_sharpness": 1.0,
    "order_ladder_auto": True,
    "order_pixel_adaptive": True,
    "order_max": 0,
    "order_elev_gain": 0.10,
    "order_demote_gain": 0.25,
    "order_novelty_k": 4.0,
    "order_reversal_cos": 0.0,
    "order_reversal_mag": 1.0,
    "order_jema_decay": 0.7,
    "order_reset_on_anomaly": True,
    "probe_is_distilled": "auto",
    "cond_probe_is_distilled": "auto",
}
def _progress(callback, ptype, phase, i=0, n=1, msg="", extra=None):
    if callback is None:
        return
    try:
        evt = {"type": ptype, "phase": phase, "i": int(i), "n": int(n),
               "msg": msg}
        if extra:
            evt.update(extra)
        callback(evt)
    except Exception:
        pass


def aflops_engine(model, x, sigmas, extra_args=None, callback=None, disable=None,
                  cfg=None, log_errors=False):
    cfg = {**ENGINE_DEFAULTS, **(cfg or {})}
    global _LAST_CFG
    extra_args = {} if extra_args is None else extra_args
    _env_fp = os.environ.get("AFLOPS_FOCUSED_PROBE")
    if _env_fp == "0":
        cfg["cond_probe_focused"] = False
    elif _env_fp == "1":
        cfg["cond_probe_focused"] = True
    auto_tunable = {k for k in ENGINE_DEFAULTS
                    if cfg.get(k) == ENGINE_DEFAULTS[k]}
    lf_prof_m = None
    lf_prof_c = None
    lf_prof_w = float(cfg.get("cond_probe_weight", 0.5))
    _gap_ctx = None
    try:
        _gap_ctx = _build_gap_ctx(model, extra_args)
    except Exception:
        _gap_ctx = None
    try:
        cfg["_guider_live_cfg"] = _probe_effective_cfg(model, extra_args,
                                                       _gap_ctx)
    except Exception:
        cfg["_guider_live_cfg"] = None

    def _dist_resolved(prof, key):
        if not isinstance(prof, dict):
            return prof
        want = _resolve_distilled_override(
            cfg.get(key, "auto"), cfg.get("_guider_live_cfg"),
            prof.get("is_distilled_eff", False))
        if want == bool(prof.get("is_distilled_eff", False)):
            return prof
        prof = dict(prof)
        prof["is_distilled_eff"] = want
        try:
            prof["trajectory_state"] = _compute_trajectory_state(prof)
        except Exception:
            pass
        return prof

    if bool(cfg.get("probe_enabled", False)):
        mk = _model_cache_key(model)
        pkey = mk + "|s:" + _schedule_sig(x, sigmas, cfg)
        profile = _PROBE_PROFILES.get(pkey)
        if profile is not None and int(profile.get("version", 1) or 1) < 7:
            profile = None
        # Fall back to the most recent profile for this model when the
        # schedule-qualified key misses -- e.g. a profile pre-computed
        # node-side by the Probe Options node BEFORE the KSampler started
        # (measured on a provisional schedule; its curves are resampled onto
        # the real sigmas by the consumers, exactly like the node-side cond
        # probe).  Without this fallback the engine would re-probe here at
        # sampling time even though the values are already cached.
        probe_from_node = False
        if profile is None and not bool(cfg.get("probe_force", False)):
            profile = _latest_profile(model)
            if profile is not None and int(profile.get("version", 1) or 1) < 7:
                profile = None
            probe_from_node = profile is not None
            if probe_from_node and log_errors:
                logging.info("[A-FloPS-probe] using pre-computed profile "
                             "(node-side probe ran before the KSampler; "
                             "skip sampling-time probe)")
        if profile is None or bool(cfg.get("probe_force", False)):
            if log_errors:
                logging.info("[A-FloPS-probe] running pre-profile pass "
                             "(%d steps @ %dx%d latent)",
                             int(cfg.get("probe_steps", 10)),
                             int(cfg.get("probe_resolution", 48)),
                             int(cfg.get("probe_resolution", 48)))
            _progress(callback, "probe", "start", 0,
                      int(cfg.get("probe_steps", 10)) + 4,
                      "A-FloPS model probe")
            _probe_model_cache_guard(model, True)
            try:
                profile = _run_probe(model, x, sigmas, extra_args, cfg,
                                     callback=callback)
            finally:
                _probe_model_cache_guard(model, False)
            _progress(callback, "probe", "end",
                      int(cfg.get("probe_steps", 10)) + 4,
                      int(cfg.get("probe_steps", 10)) + 4,
                      "A-FloPS model probe done")
            if profile is not None:
                _carry_run_feedback(profile, _PROBE_PROFILES.get(pkey))
                _PROBE_PROFILES[pkey] = profile
                _PROFILE_BY_MODEL[mk] = profile
                _prune_probe_cache(mk, keep=8)
                if log_errors:
                    logging.info("[A-FloPS-probe] profile cached: distilled=%s "
                                 "safe_fresh_frac=%.3f med_jump=%.4f med_curv=%.4f",
                                 profile.get("is_distilled_eff"),
                                 profile.get("safe_fresh_frac", -1.0),
                                 profile.get("med_jump", -1.0),
                                 profile.get("med_curv", -1.0))
        if profile is not None:
            if probe_from_node:
                # Promote the pre-computed (node-side) profile into the
                # current schedule-qualified cache slot so later runs and the
                # end-of-run direction feedback find it directly.
                try:
                    _PROBE_PROFILES[pkey] = profile
                    _prune_probe_cache(mk, keep=8)
                except Exception:
                    pass
            profile = _dist_resolved(profile, "probe_is_distilled")
            tune = {"guard": bool(cfg.get("probe_tune_guard", True)),
                    "tol": bool(cfg.get("probe_tune_tol", True)),
                    "wmax": bool(cfg.get("probe_tune_wmax", True)),
                    "anomaly": bool(cfg.get("probe_tune_anomaly", True)),
                    "eta": bool(cfg.get("probe_tune_eta", False))}
            _apply_probe_profile(cfg, profile, tune, auto_tunable)
            cfg["_probe_key"] = pkey
            lf_prof_m = profile
            if profile.get("is_distilled_eff"):
                cfg["_distilled_eff"] = True
    if bool(cfg.get("cond_probe_enabled", False)):
        mk = _model_cache_key(model)
        ckey = _cond_cache_key(cfg, extra_args, mk)
        if ckey is None and log_errors:
            logging.info("[A-FloPS-cond-probe] no conditioning available to "
                         "fingerprint; profile cannot be cached")
        skey = None
        if ckey is not None:
            try:
                skey = (ckey[0], ckey[1], _schedule_sig(x, sigmas, cfg))
            except Exception:
                skey = None
        cond_profile = None
        from_cache = False
        if skey is not None and not bool(cfg.get("cond_probe_force", False)):
            cond_profile = _COND_PROBE_PROFILES_SCHED.get(skey)
            from_cache = cond_profile is not None
            if (cond_profile is not None
                    and int(cond_profile.get("version", 1) or 1) < 7):
                cond_profile = None
                from_cache = False
        if (cond_profile is None and ckey is not None
                and not bool(cfg.get("cond_probe_force", False))):
            # Fall back to the latest per-prompt profile for this model+prompt
            # (pre-computed node-side by the autotuner on the incoming
            # schedule); its curves are resampled onto the real sigmas by the
            # consumers, exactly like the model probe's node-side fallback.
            cond_profile = _COND_PROBE_PROFILES.get(ckey)
            from_cache = cond_profile is not None
            if (cond_profile is not None
                    and int(cond_profile.get("version", 1) or 1) < 7):
                cond_profile = None
                from_cache = False
        if cond_profile is not None:
            if log_errors:
                logging.info("[A-FloPS-cond-probe] using cached per-prompt "
                             "profile (model+prompt+schedule unchanged; "
                             "skip probe)")
        else:
            if log_errors:
                logging.info("[A-FloPS-cond-probe] running per-prompt probe "
                             "(%d steps @ %dx%d latent, weight=%.2f)",
                             int(cfg.get("cond_probe_steps", 4)),
                             int(cfg.get("cond_probe_resolution", 16)),
                             int(cfg.get("cond_probe_resolution", 16)),
                             float(cfg.get("cond_probe_weight", 0.5)))
            _cond_K = int(cfg.get("cond_probe_steps", 4))
            _progress(callback, "cond_probe", "start", 0, _cond_K + 4,
                      "A-FloPS cond probe")
            _probe_model_cache_guard(model, True)
            try:
                cond_profile = _run_cond_probe(
                    model, x, sigmas, extra_args, cfg,
                    callback=callback,
                    gap_ctx=_gap_ctx)
            finally:
                _probe_model_cache_guard(model, False)
            _progress(callback, "cond_probe", "end", _cond_K + 4, _cond_K + 4,
                      "A-FloPS cond probe done")
            if cond_profile is not None and cond_profile.get("gap") is None \
                    and ckey is not None:
                _prev_gap = (_COND_PROBE_PROFILES.get(ckey) or {}).get("gap")
                if isinstance(_prev_gap, dict) and _prev_gap:
                    cond_profile["gap"] = _prev_gap
        if cond_profile is not None:
            cond_profile = _dist_resolved(cond_profile,
                                          "cond_probe_is_distilled")
            if ckey is not None:
                if not from_cache:
                    if skey is not None:
                        _carry_run_feedback(
                            cond_profile,
                            _COND_PROBE_PROFILES_SCHED.get(skey))
                    _COND_PROBE_PROFILES[ckey] = cond_profile
                    if skey is not None:
                        _COND_PROBE_PROFILES_SCHED[skey] = cond_profile
                cfg["_cond_probe_key"] = "%s|%s" % (ckey[0], ckey[1])
            cond_tune = {"guard": bool(cfg.get("cond_probe_tune_guard", True)),
                         "tol": bool(cfg.get("cond_probe_tune_tol", True)),
                         "wmax": bool(cfg.get("cond_probe_tune_wmax", True)),
                         "anomaly": bool(cfg.get("cond_probe_tune_anomaly", True)),
                         "eta": bool(cfg.get("cond_probe_tune_eta", True))}
            cond_weight = float(cfg.get("cond_probe_weight", 0.5))
            _apply_cond_probe_profile(cfg, cond_profile, cond_tune,
                                      cond_weight, auto_tunable)
            lf_prof_c = cond_profile
            if cond_profile.get("is_distilled_eff"):
                cfg["_distilled_eff"] = True
            if log_errors:
                gap = cond_profile.get("gap")
                ts = cond_profile.get("trajectory_state", {})
                logging.info("[A-FloPS-cond-probe] profile ready: "
                             "distilled=%s med_jump=%.4f med_curv=%.4f "
                             "med_err=%.4f gap=%s "
                             "oversteer=%.3f oversmooth=%.3f "
                             "weight=%.2f",
                             cond_profile.get("is_distilled_eff"),
                             cond_profile.get("med_jump", -1.0),
                             cond_profile.get("med_curv", -1.0),
                             cond_profile.get("med_local_err", -1.0),
                             "n/a" if gap is None else
                             "s=%.2f/o=%.2f" % (gap.get("strength", 0),
                                                gap.get("overshoot", 0)),
                             ts.get("oversteer", 0.0),
                             ts.get("oversmooth", 0.0),
                             cond_weight)
    # Derive the escape thresholds from the probe's measured per-pixel jump /
    # detail levels -- the SAME quantities the runtime detector uses -- unless
    # the user set them explicitly (a negative value means "auto").
    try:
        def _prof_scalar(profs, key):
            for _p in profs:
                if isinstance(_p, dict) and _p.get(key) is not None:
                    return float(_p[key])
            return None
        if float(cfg.get("eta_escape_jump", -1.0)) < 0:
            _mj = _prof_scalar((lf_prof_c, lf_prof_m), "med_jump_rel")
            if _mj is not None and _mj > 0:
                # stalled = the per-pixel jump collapsed to a quarter of its
                # measured normal level.
                cfg["eta_escape_jump"] = round(max(0.25 * _mj, 1e-4), 6)
        if float(cfg.get("eta_escape_detail", -1.0)) < 0:
            _md = _prof_scalar((lf_prof_c, lf_prof_m), "med_detail_abs")
            if _md is not None and _md > 0:
                # flat = the high-frequency energy dropped to half of the
                # measured normal level.
                cfg["eta_escape_detail"] = round(max(0.5 * _md, 0.02), 6)
        if float(cfg.get("eta_fade_sigma", -1.0)) < 0:
            # Anchor the late noise fade at the oscillation onset (where the
            # probe measured the x0 trajectory first reversing direction);
            # fall back to the jump-collapse sigma when the trajectory stays
            # monotone (no reversal -> fade where the model converged).
            _cs = (_prof_scalar((lf_prof_c, lf_prof_m), "osc_sigma")
                   or _prof_scalar((lf_prof_c, lf_prof_m), "conv_sigma"))
            if _cs is not None and _cs > 0:
                cfg["eta_fade_sigma"] = round(float(_cs), 6)
    except Exception:
        pass
    _fragile = []
    if bool(cfg.get("probe_endgame", True)) and (lf_prof_m is not None
                                                 or lf_prof_c is not None):
        try:
            _fragile = _detect_fragile_ranges(lf_prof_m, lf_prof_c)
        except Exception:
            _fragile = []
    _fp_on = bool(cfg.get("cond_probe_focused", True)) and bool(cfg.get("lf_enable", True))
    if (_fp_on
            and lf_prof_m is not None and lf_prof_c is not None
            and bool(cfg.get("cond_probe_enabled", False))
            and bool(cfg.get("probe_endgame", True))):
        try:
            if _fragile:
                _focused_key = None
                if skey is not None:
                    _focused_key = (skey[0], skey[1], skey[2], "focused")
                _focused_prof = None
                if _focused_key is not None and not bool(
                        cfg.get("cond_probe_force", False)):
                    _focused_prof = _COND_PROBE_PROFILES_SCHED.get(_focused_key)
                if _focused_prof is None:
                    _focused_grid = _build_focused_grid(
                        sigmas, _fragile,
                        K_base=max(3, int(cfg.get("cond_probe_steps", 4))),
                        K_max=max(6, int(cfg.get("cond_probe_steps", 4)) + 4),
                        distilled_mode=_probe_distilled_mode(cfg))
                    if _focused_grid and len(_focused_grid) >= 4:
                        if log_errors:
                            logging.info("[A-FloPS-focused] running focused "
                                         "cond probe (%d points, %d fragile "
                                         "ranges: %s)",
                                         len(_focused_grid), len(_fragile),
                                         [(round(sl, 4), round(sh, 4), round(f, 2))
                                          for sl, sh, f in _fragile])
                        _fp_total = len(_focused_grid)
                        _progress(callback, "cond_probe", "start", 0,
                                  _fp_total,
                                  "A-FloPS focused cond probe")
                        _probe_model_cache_guard(model, True)
                        try:
                            _focused_prof = _run_focused_cond_probe(
                                model, x, sigmas, extra_args, cfg,
                                callback=callback,
                                gap_ctx=_gap_ctx,
                                fragile_ranges=_fragile,
                                focused_grid=_focused_grid,
                                base_profile=lf_prof_c)
                        finally:
                            _probe_model_cache_guard(model, False)
                        _progress(callback, "cond_probe", "end",
                                  _fp_total, _fp_total,
                                  "A-FloPS focused cond probe done")
                        if _focused_prof is not None and _focused_key is not None:
                            _carry_run_feedback(
                                _focused_prof,
                                _COND_PROBE_PROFILES_SCHED.get(_focused_key))
                            _COND_PROBE_PROFILES_SCHED[_focused_key] = _focused_prof
                if _focused_prof is not None:
                    _focused_prof = _dist_resolved(
                        _focused_prof, "cond_probe_is_distilled")
                    _focused_prof = _inherit_base_guard_stats(
                        _focused_prof, lf_prof_c)
                    lf_prof_c = _focused_prof
                    if log_errors:
                        ts = _focused_prof.get("trajectory_state", {})
                        logging.info("[A-FloPS-focused] profile ready: "
                                     "oversteer=%.3f oversmooth=%.3f "
                                     "med_err=%.4f n_steps=%d",
                                     ts.get("oversteer", 0.0),
                                     ts.get("oversmooth", 0.0),
                                     _focused_prof.get("med_local_err", -1.0),
                                     _focused_prof.get("n_probe_steps", 0))
        except Exception as e:
            if log_errors:
                logging.warning("[A-FloPS-focused] probe skipped: %s", e)
    eta_base = 0.0 if cfg["eta"] is None else float(cfg["eta"])
    space_mode = cfg["extrap_space"]
    adaptive_on = bool(cfg.get("order_pixel_adaptive", True))
    ladder_auto = adaptive_on and bool(cfg.get("order_ladder_auto", True))
    if adaptive_on:
        _om = int(cfg.get("order_max", 0) or 0)
        order_cap = max(2, min(16, _om if _om > 0 else 10))
    else:
        order_cap = max(1, min(4, int(cfg["order"])))
    order = order_cap
    hist_cap = max(8, order_cap) if adaptive_on else 8
    corrector = "aflops"
    err_mode = cfg["err_mode"]
    grad_strength = float(cfg.get("grad_strength", 5.0))
    noise_space = cfg["noise_space"]
    s_noise = float(cfg["s_noise"])
    guard = (_OutlierGuard(window=cfg["guard_window"], z=cfg["guard_sensitivity"],
                           floor=cfg["guard_floor"],
                           ratio=cfg.get("guard_ratio", 3.0),
                           hold=cfg["guard_hold"],
                           warmup=min(4, max(1, (len(sigmas) - 1) // 4)),
                           mad_floor=cfg.get("guard_mad_floor", 0.1))
             if cfg["outlier_guard"] else None)
    gquant = float(cfg.get("guard_quantile", 0.99))
    wmax_base = float(cfg.get("wmax_base", 2.0))
    wmax_adapt = bool(cfg.get("wmax_adapt", True))
    wmax_mid_floor = float(cfg.get("wmax_mid_floor", 1.55))
    wmax_adapt_k = float(cfg.get("wmax_adapt_k", 0.5))
    wmax_limit = wmax_base
    an_enable = bool(cfg["an_enable"])
    an_from = float(cfg["an_from"])
    an_win = int(cfg["an_win"])
    if an_win % 2 == 0:
        an_win += 1
    an_spatial = float(cfg["an_spatial"])
    an_novelty = float(cfg["an_novelty"])
    an_value = float(cfg["an_value"])
    an_phase_relax = float(cfg.get("an_phase_relax", 0.5))
    an_region_z = float(cfg.get("an_region_z", 12.0))
    an_sustain = float(cfg.get("an_sustain_mult", 3.0))
    lf_feed = bool(cfg.get("lf_guard_feed", True))
    n_cand = len(sigmas) - 1
    lf_auto_g, lf_evid = _lf_auto_gains(lf_prof_m, lf_prof_c, lf_prof_w,
                                        n_cand, sigmas=sigmas, cfg=cfg)
    lf_gains = None
    _lf_clamp = 0.0
    _lf_drift = float(cfg.get("lf_drift_budget", 0.0) or 0.0)
    if _lf_drift <= 0.0:
        _lf_drift = float((lf_auto_g or {}).get("drift") or 0.10)
    if lf_auto_g is not None:
        _lf_clamp = min(max(float(lf_auto_g["clamp"]), 0.05), 1.0)
        # fragility gentle-step authority scales with the model's measured
        # (gap-corrected) local error: a more nonlinear velocity field needs
        # stronger gentle-steering in its fragile regions.
        _frag_gain = _LF_FRAG_GAIN
        try:
            _me = None
            for _p in (lf_prof_c, lf_prof_m):
                if isinstance(_p, dict):
                    _ts = _p.get("trajectory_state") or {}
                    _v = _ts.get("med_local_err_real")
                    if _v is None:
                        _v = _p.get("med_local_err")
                    if _v is not None:
                        _me = float(_v)
                        break
            if _me is not None and _me > 0:
                _frag_gain = min(max(_me / _FRAG_ERR_REF, _FRAG_GAIN_MIN),
                                 _FRAG_GAIN_MAX)
        except Exception:
            pass
        lf_gains = {"k": float(lf_auto_g["k"]),
                    "v": float(lf_auto_g["v"]),
                    "d": float(lf_auto_g["d"]),
                    "s": float(lf_auto_g["s"]),
                    "amp": 0.15 * _lf_clamp,
                    "ramp": 0.35 * _lf_clamp,
                    "drift": _lf_drift,
                    "v_sched": lf_auto_g.get("v_sched"),
                    "hi_ref": lf_auto_g.get("hi_ref"),
                    "frag_gain": round(_frag_gain, 3),
                    "env_fade": bool(cfg.get("lf_env_fade", True))}
    lf_on = (bool(cfg.get("lf_enable", True))
             and lf_gains is not None and lf_gains["k"] > 1e-3)
    lf_vol = None
    lf_medref = None
    lf_damp = None
    lf_cum = None
    lf_cum_next = None
    lf_dir_acc = [0.0, 0]
    lf_closed_loop = lf_on and bool(cfg.get("lf_closed_loop", True))
    lf_trust = None
    lf_emap_prev = None
    lf_m_prev = None
    lf_amp_prev = 0.0
    lf_gate = None
    lf_rho_prev = None
    lf_applied_prev = False
    lf_suppress = None
    lf_spatial_health = lf_on and bool(cfg.get("lf_spatial_health", True))
    lf_prior = None
    if lf_on and bool(cfg.get("lf_semantic_prior", True)):
        try:
            _gap = (lf_prof_c or {}).get("gap") or {}
            _pr = _gap.get("prior")
            if isinstance(_pr, dict) and _pr.get("v"):
                _pg = torch.tensor([float(v) for v in _pr["v"]],
                                   device=x.device,
                                   dtype=torch.float32).reshape(
                                       1, 1, int(_pr["h"]), int(_pr["w"]))
                _pg = F.interpolate(_pg, size=x.shape[-2:], mode="bilinear",
                                    align_corners=False).clamp(0.0, 1.0)
                if x.dim() == 5:
                    _pg = _pg.unsqueeze(2)  # broadcast over frames
                lf_prior = _pg.to(x.dtype)
        except Exception:
            lf_prior = None
    lf_evid["prior"] = lf_prior is not None
    lf_frag = None
    if lf_on:
        # spatial fragility prior: per-prompt probe first, model probe fallback
        try:
            for _fp in (lf_prof_c, lf_prof_m):
                _fr = (_fp or {}).get("fragility") if isinstance(_fp, dict) else None
                if isinstance(_fr, dict) and _fr.get("v"):
                    _fg = torch.tensor([float(v) for v in _fr["v"]],
                                       device=x.device,
                                       dtype=torch.float32).reshape(
                                           1, 1, int(_fr["h"]), int(_fr["w"]))
                    _fg = F.interpolate(_fg, size=x.shape[-2:],
                                        mode="bilinear",
                                        align_corners=False).clamp(0.0, 1.0)
                    if x.dim() == 5:
                        _fg = _fg.unsqueeze(2)
                    lf_frag = _fg.to(x.dtype)
                    break
        except Exception:
            lf_frag = None
    lf_evid["fragility"] = lf_frag is not None
    lf_gdir = None
    if lf_on:
        try:
            _gap = (lf_prof_c or {}).get("gap") or {}
            _pd = _gap.get("prior_dir")
            if isinstance(_pd, dict) and _pd.get("v"):
                _c = int(_pd.get("c", x.shape[1]))
                if _c == int(x.shape[1]):
                    _tg = torch.tensor([float(v) for v in _pd["v"]],
                                       device=x.device,
                                       dtype=torch.float32)
                    _tg = _tg.reshape(1, _c, int(_pd["h"]), int(_pd["w"]))
                    _tg = F.interpolate(_tg, size=x.shape[-2:],
                                        mode="bilinear",
                                        align_corners=False)
                    if x.dim() == 5:
                        _tg = _tg.unsqueeze(2)  # broadcast over frames
                    lf_gdir = _tg.to(x.dtype)
        except Exception:
            lf_gdir = None
    lf_evid["prior_dir"] = lf_gdir is not None
    lf_rescue_on = lf_on and bool(cfg.get("lf_detail_rescue", True))
    _lf_rescue = None
    if lf_rescue_on:
        _lf_rescue = {"on": True, "block": 4,
                      "start": float(cfg.get("rescue_start", 0.55)),
                      "full": float(cfg.get("rescue_full", 0.72)),
                      "amp": (float(cfg.get("rescue_cap", 0.35)) * _lf_clamp),
                      "ramp": 0.5 * lf_gains["ramp"],
                      "max_cov": float(cfg.get("rescue_max_cov", 0.35)),
                      "floor": float(cfg.get("rescue_floor", 0.02)),
                      "fast": bool(n_cand <= 9
                                   or bool(cfg.get("_distilled_eff", False))),
                      "s": 0.0}
        # Endgame evidence -> rescue window placement: when the probe
        # measured WHERE structured-state misinterpretation lives (fragile
        # sigma band), ramp the rescue authority over that band instead of
        # the fixed 0.55/0.72 of the run.  Only when the user left
        # rescue_start/rescue_full untouched and either probe_tune_lf toggle
        # is on.
        if (_fragile and (bool(cfg.get("probe_tune_lf", True))
                          or bool(cfg.get("cond_probe_tune_lf", True)))
                and "rescue_start" in auto_tunable
                and "rescue_full" in auto_tunable):
            _rs, _rf = _rescue_window_from_fragile(sigmas, _fragile)
            if _rs is not None:
                if log_errors:
                    logging.info("[A-FloPS-lf] rescue window placed from "
                                 "endgame evidence: start=%.3f full=%.3f",
                                 _rs, _rf)
                _lf_rescue["start"] = _rs
                _lf_rescue["full"] = _rf
    lf_resc_state = None
    jhist = []
    prev_elev = None
    model_info = cfg.get("_model_info")
    dist_eff = bool(cfg.get("_distilled_eff", False))
    dist_hint = bool((model_info or {}).get("is_distilled")) or dist_eff
    sigmas = sigmas.detach().float().to(x.device)
    # ---- L1: per-pixel dwell fraction ------------------------------------
    # Pixels whose x0 prediction keeps churning (per-pixel jump envelope
    # above the image median) take a PARTIAL step toward the next sigma:
    # their effective noise level lags the grid, which is the per-pixel
    # version of the high global shift -- applied only where the artifact
    # forms instead of warping the whole schedule.  The probe decides HOW
    # MUCH (oversteer evidence) and WHEN (curvature active zone + endgame
    # fragile band); the run decides WHERE.  Off by default; auto arms only
    # on qualifying probe evidence (otherwise the exact classic behavior).
    _dwell_mode = str(cfg.get("dwell_mode", "off") or "off").strip().lower()
    if _dwell_mode not in ("auto", "manual"):
        _dwell_mode = "off"
    dwell_max = 0.0
    dwell_fade = 0.65
    if _dwell_mode == "manual":
        dwell_max = min(max(float(cfg.get("dwell_max", 0.20) or 0.20),
                            0.0), 0.45)
        dwell_fade = min(max(float(cfg.get("dwell_until", 0.65) or 0.65),
                             0.2), 0.85)
    elif _dwell_mode == "auto":
        try:
            _dw = _recommend_dwell_from_profile(
                lf_prof_m, lf_prof_c, sigmas, lf_evid, _fragile)
            if _dw is not None:
                dwell_max = float(_dw["dwell_max"])
                dwell_fade = float(_dw["fade_end"])
                if log_errors:
                    logging.info(
                        "[A-FloPS-dwell] adopted: max=%.3f ramp_end=%.2f "
                        "fade_end=%.2f (oversteer=%.3f, source=%s)",
                        dwell_max, float(_dw["ramp_end"]), dwell_fade,
                        float(_dw["ov"]), lf_evid.get("source", "?"))
            elif log_errors:
                logging.info("[A-FloPS-dwell] off: no qualifying probe "
                             "evidence (oversteer below gate or no usable "
                             "curvature curve)")
        except Exception as e:
            if log_errors:
                logging.warning("[A-FloPS-dwell] skipped: %s", e)
    dwell_on = dwell_max > 1e-3 and n_cand >= 6
    dwell_ramp = 0.30
    dwell_fade_start = 1.0
    if dwell_on:
        dwell_ramp = min(0.30, 0.5 * dwell_fade)
        dwell_fade_start = max(dwell_fade - 0.15, dwell_ramp + 0.02)
        cfg["_dwell"] = {"mode": _dwell_mode, "adopted": True,
                         "dwell_max": round(dwell_max, 4),
                         "ramp_end": round(dwell_ramp, 4),
                         "fade_end": round(dwell_fade, 4)}
        if log_errors and _dwell_mode == "manual":
            logging.info("[A-FloPS-dwell] manual: max=%.3f fade_end=%.2f",
                         dwell_max, dwell_fade)
    elif _dwell_mode != "off":
        cfg["_dwell"] = {"mode": _dwell_mode, "adopted": False}
    dw_ema = None
    dw_w = None
    sigma_data = 1.0
    if model_info is not None:
        if model_info.get("is_flow") is not None and noise_space == "auto":
            space_override = "flow" if model_info["is_flow"] else "ve_edm"
        else:
            space_override = None
        sigma_data = model_info.get("sigma_data", 1.0) or 1.0
        if model_info.get("is_video") and not cfg.get("color_temporal"):
            cfg["color_temporal"] = True
    else:
        space_override = None
    if space_override is not None:
        space = space_override
    else:
        space = _detect_space(sigmas, noise_space)
    # The sampler is needed whenever the effective eta can be > 0.  With the
    # v15 escape scheme eta can come from eta_min even when eta_base
    # (stochasticity) is 0, so gate on either.
    _eta_floor_cfg = float(cfg.get("eta_min", 0.0) or 0.0)
    noise_sampler = (_make_noise_sampler(x, extra_args, cfg)
                     if (eta_base > 0 or _eta_floor_cfg > 0) else None)
    _lf_seed_raw = extra_args.get("seed", None)
    if _lf_seed_raw is None:
        _lf_seed_raw = 0
    _lf_seed_bytes = hashlib.sha256(
        ("lf_noise:" + str(_lf_seed_raw)).encode()).digest()
    _lf_noise_seed = int.from_bytes(_lf_seed_bytes[:8], 'little')
    lf_noise_gen = torch.Generator(device=x.device)
    lf_noise_gen.manual_seed(_lf_noise_seed)
    is_adaptive_noise = (str(cfg.get("noise_color", "white")) == "auto" and
                         noise_sampler is not None)
    ones = x.new_ones([x.shape[0]])
    pos = sigmas[sigmas > 1e-8]
    s_max_n = float(sigmas[0])
    s_min_n = float(pos[-1]) if len(pos) > 0 else 1e-8
    calls = 0
    tol = max(float(cfg.get("rtol", 0.05)), 1e-3)
    space_sum = {"log_sigma": 0.0, "sigma": 0.0}
    space_locked = None if space_mode == "auto" else space_mode
    hist = []
    ord_px = None
    j_ema = None
    Iema = None
    prev_err = None
    prev_cons = None
    _LAST_ERRORS.clear()
    _LAST_CFG = {"mode": "aflops",
                 "order": order, "order_cap": order_cap,
                 "adaptive_order": adaptive_on,
                 "ladder_auto": ladder_auto,
                 "space": space_locked or "auto",
                 "corrector": corrector,
                 "eta_base": eta_base, "s_noise": s_noise,
                 "err_mode": err_mode,
                 "noise_color": cfg.get("noise_color", "white"),
                 "color_schedule": cfg.get("color_schedule", "constant"),
                 "color_beta_max": cfg.get("color_beta_max", 1.5),
                 "color_beta_min": cfg.get("color_beta_min", -0.5),
                 "model_info": (None if model_info is None else dict(model_info)),
                 "steps": n_cand, "distilled_hint": dist_hint,
                 "sigma_data": sigma_data,
                 "probe_key": cfg.get("_probe_key"),
                 "cond_probe_key": cfg.get("_cond_probe_key"),
                 "lf": {"enabled": lf_on,
                        "strength": (lf_gains["k"] if lf_gains else 0.0),
                        "detail": (lf_gains["d"] if lf_gains else 0.0),
                        "volatility": (lf_gains["v"] if lf_gains else 0.0),
                        "resharpen": (lf_gains["s"] if lf_gains else 0.0),
                        "clamp": _lf_clamp,
                        "drift": (lf_gains["drift"] if lf_gains else 0.0),
                        "guard_feed": lf_feed,
                        "runtime_gate": (lf_on and lf_closed_loop
                                         and bool(cfg.get("lf_runtime_gate", True))),
                        "v3": {"closed_loop": lf_closed_loop,
                               "env_fade": bool(lf_gains and lf_gains.get("env_fade")),
                               "spatial_health": lf_spatial_health,
                               "band_refresh": (lf_on and bool(cfg.get("lf_band_refresh", True))
                                                and bool(lf_gains and lf_gains.get("hi_ref"))),
                               "dir_vol": (lf_on and bool(cfg.get("lf_dir_vol", True))),
                               "semantic_prior": lf_prior is not None,
                               "v_sched": bool(lf_gains and lf_gains.get("v_sched")),
                               "rescue": lf_rescue_on},
                        "evidence": lf_evid},
                 "cfg": {k: v for k, v in cfg.items() if k != "_model_info"}}
    if log_errors and lf_on:
        logging.info(
            "[A-FloPS] local schedule field auto-tuned from probe "
            "discovery (%s: oversteer=%.2f oversmooth=%.2f err=%.4f "
            "spike=%.2f distilled=%s -> strength=%.2f volatility=%.2f "
            "detail=%.2f resharpen=%.2f clamp=%.2f drift=%.2f | v3: n=%s "
            "conf=%s eg=%s dir=%s closed_loop=%s env_fade=%s spatial_health=%s "
            "v_sched=%s band_ref=%s prior=%s rescue=%s)",
            lf_evid.get("source", "?"), lf_evid.get("ov", 0.0),
            lf_evid.get("osm", 0.0), lf_evid.get("err", 0.0),
            lf_evid.get("spike", 1.0), lf_evid.get("dist", False),
            lf_gains["k"], lf_gains["v"], lf_gains["d"],
            lf_gains["s"], _lf_clamp, lf_gains["drift"],
            lf_evid.get("n"), lf_evid.get("conf"), lf_evid.get("eg"),
            lf_evid.get("dir"), lf_closed_loop,
            lf_gains.get("env_fade"), lf_spatial_health,
            bool(lf_gains.get("v_sched")), bool(lf_gains.get("hi_ref")),
            lf_prior is not None, lf_rescue_on)
    elif (log_errors and not lf_on and bool(cfg.get("lf_enable", True))
          and lf_evid is not None and lf_evid.get("source") == "none"):
        logging.info(
            "[A-FloPS] local schedule field OFF: no probe evidence "
            "(the field is automatic-only; enable the Model Probe or "
            "Cond Probe so its gains can be derived -- it never runs on "
            "guessed defaults)")
    # The engine's own sampling call.  If a model-internal cache (prefix/KV
    # caches keyed by full-resolution sequence shapes) blows up mid-run --
    # e.g. slots left behind by an older ComfyUI revision, an interrupted
    # earlier run, or any other source outside our probe isolation -- retry
    # once with the model's caches disabled and stay on that path for the
    # rest of the run.  Normal runs keep the model's own caching behaviour
    # completely untouched.
    _run_caches_off = [False]

    def _run_model_call(xx, ss):
        if _run_caches_off[0]:
            ea = dict(extra_args)
            _mopt = _probe_model_options(ea)
            if _mopt is not None:
                ea["model_options"] = _mopt
            return model(xx, ss, **ea)
        try:
            return model(xx, ss, **extra_args)
        except Exception as e:
            ea2 = dict(extra_args)
            _mopt = _probe_model_options(ea2)
            _fixed = _mopt is not None
            if not _fixed:
                # Last resort for models whose own per-run cache carries the
                # poisoned state without any options-level switch: flip the
                # instance switch off for the rest of this run.  A disabled
                # cache is transparent (it just recomputes), so this can
                # never change the run's results.
                _reset = _find_prefix_reset(model)
                if callable(_reset):
                    try:
                        _reset(False)
                        _fixed = True
                    except Exception:
                        _fixed = False
            if not _fixed:
                raise
            if _mopt is not None:
                ea2["model_options"] = _mopt
            logging.warning("[A-FloPS] model call failed (%s: %s); retrying "
                            "with the model's internal caches disabled for "
                            "the rest of this run", type(e).__name__, e)
            _run_caches_off[0] = True
            return model(xx, ss, **ea2)
    for i in range(n_cand):
        s = float(sigmas[i])
        sn = float(sigmas[i + 1])
        k = i + 1
        x_input = x
        x0 = _run_model_call(x, sigmas[i] * ones)
        calls += 1
        u_log = math.log(max(s, 1e-8))
        u_sig = s
        frac = min(max((math.log(max(s_max_n, 1e-8)) - math.log(max(s, 1e-8))) /
                       max(math.log(max(s_max_n, 1e-8)) - math.log(max(s_min_n, 1e-8)), 1e-8), 0.0), 1.0)
        jmap = (x0 - hist[-1][2]).pow(2).mean(dim=1).sqrt() if len(hist) > 0 else None
        eta_stuck = eta_jump_rel = eta_detail_abs = None
        if cfg["eta_mode"] == "adaptive":
            eta_jump_rel, eta_detail_abs = _stuck_components(x0, jmap)
            _dir_cos = None
            if jmap is not None and len(hist) >= 2:
                _d1 = (x0.detach() - hist[-1][2]).float()
                _d2 = (hist[-1][2] - hist[-2][2]).float()
                _den = _d1.pow(2).sum().sqrt() * _d2.pow(2).sum().sqrt()
                if float(_den) > 1e-8:
                    _dir_cos = float((_d1 * _d2).sum() / _den)
            eta_stuck = (_global_stuck_signal(x0, jmap, cfg,
                                              dir_cos=_dir_cos, s=float(s))
                         if jmap is not None else 0.0)
            eta_eff = _adaptive_eta(prev_err, prev_cons, s, sn, s_max_n, s_min_n,
                                    space, sigma_data, s_noise, model_info,
                                    eta_base, cfg,
                                    few_step=(n_cand <= 8 or dist_eff),
                                    stuck=eta_stuck)
        else:
            eta_eff = _eta_schedule(eta_base, cfg["eta_mode"], frac, s, s_max_n,
                                    cfg["falloff"], cfg["eta_min"])
        if not (s >= float(cfg["s_tmin"]) and (float(cfg["s_tmax"]) <= 0.0 or s <= float(cfg["s_tmax"]))):
            eta_eff = 0.0
        if dist_eff and i == n_cand - 2:
            eta_eff = 0.0
        outlier_sig = None
        guard_trip = guard_active = False
        if guard is not None and len(hist) > 0 and sn > 1e-8:
            outlier_sig = _jump_signal(x0, hist[-1][2], x, q=gquant)
            z_relax = float(cfg.get("guard_phase_relax", 1.0))
            z_eff = float(cfg["guard_sensitivity"]) * (1.0 + z_relax * (1.0 - frac))
            guard_trip, guard_active = guard.step(outlier_sig, z=z_eff)
        if eta_eff < 1e-6:
            eta_eff = 0.0
        s2, s_up = _ancestral_split(s, sn, eta_eff, space)
        amask = None
        an_detected = False
        step_cons = None
        if jmap is not None and s2 > 1e-8:
            _xn = float(x0.float().norm().clamp_min(1e-8))
            _jn = float(jmap.float().norm().clamp_min(1e-8))
            _m0 = _consistency_step(s, s2, space)
            step_cons = _compute_consistency(_jn, _xn, _m0)
        j_prev_max = torch.stack(jhist, 0).max(dim=0).values if jhist else None
        elev = None
        if an_enable and jmap is not None:
            elev = _elevation_mask(jmap, an_win, an_value, x0, mult=an_sustain)
        if (an_enable and k >= int(an_from * n_cand)
                and len(jhist) >= (1 if n_cand <= 6 else 2)
                and j_prev_max is not None and s2 > 1e-8 and jmap is not None):
            an_phase_mult = 1.0 + an_phase_relax * (1.0 - frac)
            amask = _anomaly_mask(x0, hist[-1][2], j_prev_max,
                                  an_win,
                                  an_spatial * an_phase_mult,
                                  an_novelty * an_phase_mult,
                                  an_value,
                                  jmap=jmap,
                                  region_z=an_region_z)
            if prev_elev is not None and elev is not None:
                sust = elev & prev_elev
                if len(jhist) > 0:
                    sust = sust & (jmap > jhist[-1])
                sust = _clean_mask(sust, erode=1, dilate=2)
                if bool(sust.any()):
                    amask = amask | sust
            if bool(amask.any()):
                an_detected = True
        if jmap is not None:
            jhist.append(jmap)
            if len(jhist) > 3:
                jhist.pop(0)
        prev_elev = elev
        if dwell_on and jmap is not None:
            # per-pixel churn envelope: max-decay EMA of the jump map,
            # normalized by the image median (>= median: keeping pace, no
            # dwell; ~2x median: full dwell weight)
            _jdw = jmap.float().unsqueeze(1)
            dw_ema = (_jdw if dw_ema is None
                      else torch.maximum(_jdw, _DWELL_EMA * dw_ema))
            _b_dw = dw_ema.shape[0]
            _dw_med = dw_ema.reshape(_b_dw, -1).median(dim=1).values\
                .reshape(-1, 1, 1, 1).clamp_min(1e-12)
            dw_w = (((dw_ema / _dw_med) - 1.0) / (_DWELL_R_HI - 1.0))\
                .clamp(0.0, 1.0)
        lf_m = lf_rho = None
        lf_diag = None
        lf_cum_next = None
        if lf_on:
            _kn = k / n_cand
            if lf_damp is not None:
                lf_damp = lf_damp * 0.65
                lf_damp = torch.where(lf_damp < 0.02,
                                      torch.zeros_like(lf_damp), lf_damp)
            if lf_feed:
                new_damp = None
                if guard_trip and jmap is not None:
                    flat_j = jmap.reshape(jmap.shape[0], -1).float()
                    thr = torch.quantile(
                        flat_j, min(max(gquant, 0.0), 0.999),
                        dim=1).view(-1, *([1] * (jmap.dim() - 1)))
                    new_damp = (jmap.float() >= thr).float().unsqueeze(1) * 0.9
                if amask is not None and bool(amask.any()):
                    msk = amask.float().unsqueeze(1)
                    new_damp = msk if new_damp is None else torch.maximum(new_damp, msk)
                if new_damp is not None:
                    lf_damp = (new_damp if lf_damp is None
                               else torch.maximum(lf_damp, new_damp))
            lf_wdir = None
            if bool(cfg.get("lf_dir_vol", True)) and len(hist) >= 2:
                _d1 = (x0.detach() - hist[-1][2]).float()
                _d2 = (hist[-1][2] - hist[-2][2]).float()
                _num = (_d1 * _d2).sum(dim=1)
                _den = (_d1.pow(2).sum(dim=1).sqrt()
                        * _d2.pow(2).sum(dim=1).sqrt())
                _cos = torch.where(_den > 1e-8, _num / _den.clamp_min(1e-8),
                                   torch.zeros_like(_num))
                if _kn >= 1.0 / 3.0:
                    _cos_mean = _cos.mean()
                    lf_dir_acc[0] += float(_cos_mean)
                    lf_dir_acc[1] += 1
                lf_wdir = _lf_dir_weight(
                    _cos, 1.0 / math.sqrt(max(1, x.shape[1]))).unsqueeze(1)
            g_step = lf_gains
            _vs = lf_gains.get("v_sched")
            if _vs and i < len(_vs) and float(_vs[i]) != 1.0:
                g_step = dict(lf_gains)
                g_step["v"] = _lf_v_step(lf_gains["v"], float(_vs[i]))
            if _lf_rescue is not None:
                _lf_rescue["s"] = float(s)
            lf_m, lf_rho, vol_new, med_new, lf_resc, lf_cum_next = _lf_step_field(
                x0, jmap, lf_vol, lf_medref, lf_damp,
                _lf_window(_kn), _lf_vol_window(_kn), g_step, kn=_kn,
                trust=(lf_trust if lf_closed_loop else None),
                gate=(lf_gate if lf_closed_loop else None),
                yield_map=lf_suppress,
                w_dir=lf_wdir, prior=lf_prior, fragility=lf_frag,
                spatial_health=lf_spatial_health,
                rescue=_lf_rescue, resc_state=lf_resc_state,
                eg_n=float(lf_evid.get("eg_n") or 0.0),
                step_d=(x_input - x0), gdir=lf_gdir,
                cum=lf_cum)
            lf_resc_rw = None
            if lf_resc is not None:
                lf_resc_state = lf_resc.get("state")
                lf_resc_rw = lf_resc.get("rw")
            if vol_new is not None:
                lf_vol = vol_new
            lf_medref = med_new
            # Per-step local-field telemetry is computed for the report
            # regardless of log_errors (the report collects per-step data
            # whenever the run is < 64 steps, so gating this on log_errors
            # silently blanked the field's diagnostics for the JSON report).
            lf_diag = {"m_min": round(float((1.0 + lf_m).min()), 4),
                       "m_max": round(float((1.0 + lf_m).max()), 4),
                       "m_abs": round(float(lf_m.abs().mean()), 4),
                       "rho_max": round(float(lf_rho.max()), 4),
                       "damp": (round(float((lf_damp > 0.05).float().mean()), 4)
                                if lf_damp is not None else 0.0),
                       "env": round(_lf_env_fade(_kn, float(lf_evid.get("eg_n") or 0.0))
                                    if lf_gains.get("env_fade") else 1.0, 3),
                       "vmul": (round(float(g_step["v"])
                                      / max(lf_gains["v"], 1e-9), 3)
                                if (_vs and i < len(_vs)) else None),
                       "trust": (round(float(lf_trust.mean()), 3)
                                 if lf_trust is not None else None),
                       "tmin": (round(float(lf_trust.min()), 3)
                                if lf_trust is not None else None),
                       "gate": (round(float(lf_gate), 3)
                                if lf_gate is not None else None),
                       "yield": (round(float(lf_suppress.mean()), 3)
                                 if lf_suppress is not None else None),
                       "resc": ({"cov": round(float(lf_resc.get("cov", 0.0)), 4),
                                 "hit": lf_resc.get("hit", 0.0),
                                 "lf_share": lf_resc.get("lf_share", 0.0)}
                                if lf_resc is not None else 0.0),
                       "drift_p90": (round(float(lf_cum_next.abs()
                                                  .quantile(0.9)), 4)
                                     if lf_cum_next is not None else 0.0),
                       "drift": round(float(lf_gains.get("drift") or 0.0), 3)}
        x1 = exact_step(x, x0, s, s2)
        use_space = space_locked or "log_sigma"
        # Order-k correction needs >= k prior velocity samples.  The old
        # len(hist)+1 let an order-2 midpoint + exponential-integrator slope
        # run on the FIRST correction step (1 history entry): a 2-point
        # interpolation across the sigma~1 commit, where x0 flips
        # discontinuously.  Log rig (CFG5log/CFG1log): the only >1x-tol step
        # errors at that point are exactly these first order-2 corrections
        # (2.79x / 6.29x tol; every order-1 step nearby <= 0.41x tol).
        eff_cap = max(1, min(order_cap, len(hist)))
        if guard_active:
            eff_cap = min(eff_cap, 1)
        hist_cos_val = None
        if (bool(cfg.get("history_consistency_gate", True))
                and len(hist) >= 2 and eff_cap >= 2 and s2 > 1e-8):
            _hd1 = (x0.float() - hist[-1][2].float())
            _hd2 = (hist[-1][2].float() - hist[-2][2].float())
            _hn1 = _hd1.norm().clamp_min(1e-8)
            _hn2 = _hd2.norm().clamp_min(1e-8)
            hist_cos_val = float(((_hd1 * _hd2).sum() / (_hn1 * _hn2))
                                 .clamp(-1.0, 1.0))
            _thr = float(cfg.get("history_consistency_threshold", 0.0))
            if hist_cos_val < -_thr:
                eff_cap = 1
            elif (hist_cos_val < _thr and eff_cap > 2
                  and not (adaptive_on
                           or bool(cfg.get("order_pixel_blend", True)))):
                eff_cap = 2
        if wmax_adapt and prev_err is not None:
            smooth = max(0.0, min(float(cfg["rtol"]) / max(prev_err, 1e-8), 3.0))
            wmax_limit = wmax_base * (1.0 + wmax_adapt_k * smooth)
        else:
            wmax_limit = wmax_base
        _eg_n = float(lf_evid.get("eg_n") or 0.0)
        if _eg_n > 0.3:
            _wmax_max = wmax_base * (1.0 + wmax_adapt_k * 3.0)
            wmax_limit = max(wmax_limit, _wmax_max)
        eff = 1
        wmax = 0.0
        x0m = x0
        if eff_cap >= 2 and s2 > 1e-8 and len(hist) >= 1:
            u2_log = math.log(max(s2, 1e-8))
            u2_sig = s2
            if space_locked is None:
                pts_log = [(h[0], h[2]) for h in hist] + [(u_log, x0)]
                pts_sig = [(h[1], h[2]) for h in hist] + [(u_sig, x0)]
                _lim2 = max(wmax_limit, wmax_mid_floor)
                for ce in range(eff_cap, 1, -1):
                    c_log = _mid_x0(pts_log, 0.5 * (u_log + u2_log), ce)
                    if c_log[1] > (_lim2 if ce == 2 else wmax_limit):
                        continue
                    if k >= 2:
                        c_sig = _mid_x0(pts_sig, 0.5 * (s + u2_sig), ce)
                        space_sum["log_sigma"] += _err_norm(
                            exact_step(x, c_log[0], s, s2), x1, err_mode,
                            x0_ref=x0, strength=grad_strength)
                        space_sum["sigma"] += _err_norm(
                            exact_step(x, c_sig[0], s, s2), x1, err_mode,
                            x0_ref=x0, strength=grad_strength)
                    x0m, wmax, eff = c_log[0], c_log[1], ce
                    break
                if eff >= 2 and k >= 5:
                    space_locked = min(space_sum, key=space_sum.get)
                    use_space = space_locked
            else:
                u_i = u_log if use_space == "log_sigma" else u_sig
                u_n = u2_log if use_space == "log_sigma" else s2
                pts = [((h[0] if use_space == "log_sigma" else h[1]), h[2])
                       for h in hist] + [(u_i, x0)]
                _lim2 = max(wmax_limit, wmax_mid_floor)
                for ce in range(eff_cap, 1, -1):
                    cand = _mid_x0(pts, 0.5 * (u_i + u_n), ce)
                    if cand[1] <= (_lim2 if ce == 2 else wmax_limit):
                        x0m, wmax, eff = cand[0], cand[1], ce
                        break
        blend_w_mean = None
        ord_tel = None
        dwell_tel = None
        ladder_ran = False
        lf_contam = None
        if lf_applied_prev and lf_m_prev is not None:
            _cm = (lf_m_prev.abs() > (_LF_ACT_M * lf_amp_prev)).unsqueeze(1)
            if lf_rho_prev is not None:
                _cm = _cm | (lf_rho_prev > _LF_ACT_RHO)
            lf_contam = _cm.to(torch.bool)
        lf_applied_prev = False
        if (bool(cfg.get("order_pixel_adaptive", True))
                and eff_cap >= 2 and s2 > 1e-8 and len(hist) >= 1):
            bcos = None
            if len(hist) >= 2:
                _bd1 = (x0.float() - hist[-1][2].float())
                _bd2 = (hist[-1][2].float() - hist[-2][2].float())
                _bnum = (_bd1 * _bd2).sum(dim=1, keepdim=True)
                _bden = (_bd1.pow(2).sum(dim=1, keepdim=True).sqrt()
                         * _bd2.pow(2).sum(dim=1, keepdim=True).sqrt())
                bcos = torch.where(_bden > 1e-8,
                                   _bnum / _bden.clamp_min(1e-8),
                                   torch.zeros_like(_bnum))
            u_w = u_log if use_space == "log_sigma" else u_sig
            u2_w = u2_log if use_space == "log_sigma" else s2
            u_mid_w = 0.5 * (u_w + u2_w)
            pts_w = [((h[0] if use_space == "log_sigma" else h[1]), h[2])
                     for h in hist] + [(u_w, x0)]
            _lim2 = max(wmax_limit, wmax_mid_floor)
            cands = [x0]
            allowed = [True]
            wlevs = [0.0]
            for ce in range(2, eff_cap + 1):
                c, wv = _mid_x0(pts_w, u_mid_w, ce)
                cands.append(c)
                wlevs.append(wv)
                allowed.append(wv <= (_lim2 if ce == 2 else wmax_limit))
            _max_ord = eff_cap
            while _max_ord > 1 and not allowed[_max_ord - 1]:
                _max_ord -= 1
            wmax = max(wlevs[1:_max_ord]) if _max_ord >= 2 else 0.0
            P = min(len(hist), order_cap)
            I_list = _pixel_innovations(pts_w[:-1], u_w, x0, P)
            if I_list and I_list[0] is None and jmap is not None:
                I_list[0] = jmap.float().unsqueeze(1)
            _shape = next((e.shape for e in I_list if e is not None),
                          (x0.shape[0], 1, *x0.shape[2:]))
            dev = x0.device
            _fill = torch.full(_shape, _ORD_BIG, device=dev,
                               dtype=torch.float32)
            Im = torch.stack([_fill if e is None else e.float()
                              for e in I_list], 0)
            reset_m = None
            j = jmap.float().unsqueeze(1) if jmap is not None else None
            if j is not None:
                if j_ema is None:
                    j_ema = j
                else:
                    if ladder_auto:
                        _r = (j / j_ema.clamp_min(1e-12)).clamp(1e-6, 1e6)
                        _lr = _r.log()
                        _lb = _lr.shape[0]
                        _flat = _lr.reshape(_lb, -1)
                        _med = _flat.median(dim=1).values.unsqueeze(1)
                        _mad = (_flat - _med).abs().median(dim=1).values
                        _mz = (0.6745 * (_flat - _med)
                               / _mad.clamp_min(1e-3).unsqueeze(1))
                        reset_m = ((_mz > _ORD_Z).reshape(_lr.shape)
                                   & (_r > _ORD_R_FLOOR))
                    else:
                        reset_m = (j > (float(cfg.get("order_novelty_k", 4.0))
                                        * j_ema))
                    _rc = (0.0 if ladder_auto
                           else float(cfg.get("order_reversal_cos", 0.0)))
                    _rm = float(cfg.get("order_reversal_mag", 1.0))
                    if bcos is not None:
                        reset_m = reset_m | ((bcos < _rc)
                                             & (j > (_rm * j_ema)))
            if reset_m is None:
                reset_m = torch.zeros(_shape, dtype=torch.bool, device=dev)
            if (amask is not None
                    and bool(cfg.get("order_reset_on_anomaly", True))):
                reset_m = reset_m | amask.to(dev).bool().unsqueeze(1)
            o = ord_px
            if o is None:
                o = torch.full(_shape, 2.0, device=dev, dtype=torch.float32)
            o = o.clamp(1.0, float(max(P, 2)))
            if P >= 2:
                idxT = torch.arange(P, device=dev, dtype=o.dtype).view(
                    P, *([1] * o.dim()))
                if ladder_auto:
                    meas = Im < _ORD_BIG
                    if Iema is None:
                        Iema = Im.clone()
                    else:
                        if Iema.shape[0] < P:
                            _fills = torch.full(
                                (P - Iema.shape[0],) + _shape, _ORD_BIG,
                                device=dev, dtype=torch.float32)
                            Iema = torch.cat([Iema, _fills], 0)
                        Iema = Iema[:P]
                    Iema = torch.where(
                        meas & (Iema < _ORD_BIG),
                        _ORD_EMA * Iema + (1.0 - _ORD_EMA) * Im,
                        torch.where(meas, Im, Iema))
                    _bias = (1e-4 * idxT)
                    p_star = torch.where(
                        (idxT < float(_max_ord)) & meas,
                        Iema + _bias,
                        torch.full_like(Iema, _ORD_BIG)).argmin(0).float() + 1.0
                    move = (p_star - o).clamp(-1.0, 1.0)
                    if lf_contam is not None:
                        move = torch.where(lf_contam & (move > 0.0),
                                           torch.zeros_like(move), move)
                    o_new = torch.where(
                        reset_m, torch.ones_like(o),
                        (o + move).clamp(1.0, float(max(P, 2))))
                else:
                    I_o = (Im * (idxT == (o - 1.0).unsqueeze(0))).sum(0)
                    I_lo = (Im * (idxT == (o - 2.0).clamp_min(0.0)
                                  .unsqueeze(0))).sum(0)
                    I_hi = (Im * (idxT == o.unsqueeze(0))).sum(0)
                    demote_m = I_o > ((1.0 + float(cfg.get("order_demote_gain",
                                                           0.25))) * I_lo)
                    elev_m = ((o < float(P))
                              & (I_hi < ((1.0 - float(cfg.get("order_elev_gain",
                                                              0.10))) * I_o)))
                    o_new = torch.where(reset_m, torch.ones_like(o),
                                        torch.where(demote_m,
                                                    (o - 1.0).clamp_min(1.0),
                                                    torch.where(elev_m,
                                                                o + 1.0, o)))
            else:
                o_new = torch.where(reset_m, torch.ones_like(o), o)
            ord_px = o_new
            ladder_ran = True
            arm = []
            _run = 1
            for ce in range(1, eff_cap + 1):
                if allowed[ce - 1]:
                    _run = ce
                arm.append(_run)
            arm_t = torch.tensor(arm, device=dev, dtype=torch.long)
            o_l = o_new.clamp(1.0, float(eff_cap)).long()
            o_map = arm_t[(o_l - 1)].float()
            C = torch.stack([c.float() for c in cands], 0)
            idxC = torch.arange(eff_cap, device=dev,
                                dtype=torch.long).view(
                eff_cap, *([1] * o_map.dim()))
            _sel_hi = (idxC == (o_map.long() - 1).unsqueeze(0)).to(C.dtype)
            _sel_lo = (idxC == (o_map.long() - 2).clamp_min(0)
                       .unsqueeze(0)).to(C.dtype)
            A_hi = (C * _sel_hi).sum(0)
            A_lo = (C * _sel_lo).sum(0)
            _sharp = float(cfg.get("order_pixel_blend_sharpness", 6.0))
            if bcos is not None:
                _w = torch.sigmoid(bcos * _sharp)
            else:
                _w = torch.ones_like(A_hi[:, :1])
            x0m = _w * A_hi + (1.0 - _w) * A_lo
            eff = 2 if _max_ord >= 2 else 1
            if len(_LAST_ERRORS) < 64 or log_errors:
                blend_w_mean = float(_w.mean())
                ord_tel = (float(o_map.mean()), int(o_map.max()),
                           float(reset_m.float().mean()))
            if ladder_auto and P >= 2:
                idxY = torch.arange(P, device=dev, dtype=torch.long).view(
                    P, *([1] * o_map.dim()))
                I1m = Im[0]
                I_ap = (Im * (idxY == (o_map.long() - 1)
                              .unsqueeze(0))).sum(0)
                _yv = (1.0 - I_ap / (I1m + 1e-6)).clamp(0.0, 1.0)
                _yv = torch.where((I1m < _ORD_BIG) & (I_ap < _ORD_BIG),
                                  _yv, torch.zeros_like(_yv))
                lf_suppress = (_yv if lf_suppress is None
                               else _ORD_EMA * lf_suppress
                               + (1.0 - _ORD_EMA) * _yv)
            if j is not None:
                j_ema = torch.maximum(
                    j, (_ORD_ENV_DECAY if ladder_auto
                        else float(cfg.get("order_jema_decay", 0.7)))
                    * j_ema)
        elif (bool(cfg.get("order_pixel_blend", True))
                and eff >= 3 and s2 > 1e-8 and len(hist) >= 2):
            if space_locked is None:
                if use_space == "log_sigma":
                    _bpts = [(h[0], h[2]) for h in hist] + [(u_log, x0)]
                    _bmid = 0.5 * (u_log + u2_log)
                else:
                    _bpts = [(h[1], h[2]) for h in hist] + [(u_sig, x0)]
                    _bmid = 0.5 * (s + u2_sig)
            else:
                _bu_i = u_log if use_space == "log_sigma" else u_sig
                _bu_n = u2_log if use_space == "log_sigma" else s2
                _bpts = [((h[0] if use_space == "log_sigma" else h[1]), h[2])
                         for h in hist] + [(_bu_i, x0)]
                _bmid = 0.5 * (_bu_i + _bu_n)
            _c_o2 = _mid_x0(_bpts, _bmid, 2)
            if _c_o2[1] <= max(wmax_limit, wmax_mid_floor):
                _x0m_o2 = _c_o2[0]
                _bd1 = (x0.float() - hist[-1][2].float())
                _bd2 = (hist[-1][2].float() - hist[-2][2].float())
                _bnum = (_bd1 * _bd2).sum(dim=1, keepdim=True)
                _bden = (_bd1.pow(2).sum(dim=1, keepdim=True).sqrt()
                         * _bd2.pow(2).sum(dim=1, keepdim=True).sqrt())
                _bcos = torch.where(_bden > 1e-8,
                                    _bnum / _bden.clamp_min(1e-8),
                                    torch.zeros_like(_bnum))
                _sharp = float(cfg.get("order_pixel_blend_sharpness", 6.0))
                _w = torch.sigmoid(_bcos * _sharp)
                x0m = _w * x0m + (1.0 - _w) * _x0m_o2
                blend_w_mean = float(_w.mean())
        if guard_active and adaptive_on and ord_px is not None:
            ord_px = torch.ones_like(ord_px)
        if not ladder_ran and lf_suppress is not None:
            lf_suppress = _ORD_EMA * lf_suppress
        if eff >= 2:
            _x0m_env = (_AFLOPS_X0M_ENV
                        if cfg.get("x0m_env") is None
                        else float(cfg["x0m_env"]))
            _env_list = [h[2] for h in hist[-(eff + 1):]] + [x0]
            _env = _x0m_env * torch.stack(
                [t.float().norm() for t in _env_list]).max()
            _over = x0m.float().norm() > _env
            x0m = torch.where(_over, x0.float(), x0m.float())
        x_pred = exact_step(x, x0m, s, s2) if eff >= 2 else x1
        err = _err_norm(x_pred, x1, err_mode, x0_ref=x0,
                        strength=grad_strength) if (eff >= 2 and s2 > 1e-8) else None
        emap = None
        if lf_on and err is not None:
            emap = (x_pred - x1).float().pow(2).mean(dim=1).sqrt()
            if lf_closed_loop:
                lf_trust, lf_emap_prev, _eff, _nst = _lf_trust_update(
                    emap, lf_emap_prev, lf_m_prev, lf_amp_prev, lf_trust)
                if _eff is not None and bool(cfg.get("lf_runtime_gate", True)):
                    _base = (torch.ones_like(_eff) if lf_gate is None
                             else lf_gate)
                    _cand = (_base + _GATE_ETA * (_eff - _GATE_TARGET))\
                        .clamp(_GATE_FLOOR, 1.0)
                    lf_gate = torch.where(_nst > 0, _cand, _base)
        corr_used = None
        aflops_lam = None
        if s2 > 1e-8:
            # The residual integrator consumes the order-ladder midpoint (x0m),
            # not the raw x0 -- this is what feeds the per-pixel adaptive order
            # into the actual step.  When the ladder is gated (eff_cap == 1) x0m
            # is x0, so the plain 1st-order path is unchanged.
            aflops_out, aflops_lam = _aflops_step(
                x, x0m, hist, s, s2, order=eff_cap,
                lam_min=cfg.get("aflops_lam_min"),
                lam_max=cfg.get("aflops_lam_max"))
            if aflops_out is not None:
                # the integrator's deviation is measured against the *midpoint*
                # step (x_pred), which is the 1st-order version of the same
                # clean estimate the integrator uses.
                err = _err_norm(aflops_out, x_pred, err_mode, x0_ref=x0m,
                                strength=grad_strength)
                # The 2nd-order exponential integrator is only trusted while its
                # deviation from the midpoint step stays inside rtol.  At high
                # latent resolution (and low sigma) the velocity field is more
                # nonlinear, so the correction can overshoot -- which reads as
                # burned / glitchy patches.  Blend back toward the midpoint step
                # as the error grows past rtol (err == the already-computed step
                # error, so this is zero extra model calls).
                if err is not None and err > tol:
                    _w = max(tol / max(err, 1e-12), 0.0)
                    x = x_pred + _w * (aflops_out - x_pred)
                else:
                    x = aflops_out
                corr_used = "aflops"
        if corr_used is None:
            x = x_pred
            corr_used = "mid"
        cerr = None
        if s2 > 1e-8 and eff < 2:
            if err is None:
                cerr = _err_norm(x, x1, err_mode, x0_ref=x0,
                                 strength=grad_strength)
            if lf_on and emap is None and corr_used not in (None, "mid"):
                emap = (x - x1).float().pow(2).mean(dim=1).sqrt()
                if lf_closed_loop:
                    lf_trust, lf_emap_prev, _eff, _nst = _lf_trust_update(
                        emap, lf_emap_prev, lf_m_prev, lf_amp_prev,
                        lf_trust)
                    if (_eff is not None
                            and bool(cfg.get("lf_runtime_gate", True))):
                        _base = (torch.ones_like(_eff) if lf_gate is None
                                 else lf_gate)
                        _cand = (_base + _GATE_ETA * (_eff - _GATE_TARGET))\
                            .clamp(_GATE_FLOOR, 1.0)
                        lf_gate = torch.where(_nst > 0, _cand, _base)
        if lf_on and lf_m is not None and s2 > 1e-8:
            x = x0 + (1.0 + lf_m) * (s2 / max(s, 1e-8)) * (x_input - x0) + (x - x1)
            _surgical = False
            _rho_band = None
            if (bool(cfg.get("lf_band_refresh", True))
                    and lf_gains.get("hi_ref") and jmap is not None
                    and i < len(lf_gains["hi_ref"])):
                try:
                    _hi = _lf_band_fractions(jmap)[0]
                    _ref = float(lf_gains["hi_ref"][i])
                    _surgical = _hi > (1.25 * _ref + 0.03)
                    if _surgical:
                        if _ref > 1e-9:
                            # refresh magnitude scales with the measured excess
                            # of the hi-band over the model's own probe-measured
                            # baseline (0 at the detection threshold, max at 2x).
                            _excess = (_hi - _ref) / _ref
                            _rho_band = _BAND_RHO_MAX * min(max(
                                (_excess - _BAND_RHO_REF)
                                / (1.0 - _BAND_RHO_REF), 0.0), 1.0)
                        else:
                            _rho_band = _LF_BAND_RHO
                except Exception:
                    _surgical = False
            _rho_eff = lf_rho
            if _surgical and (_rho_eff is None or float(_rho_eff.max()) <= 1e-4):
                # Checkerboard/banding signature detected, but the refresh
                # magnitude is gated on the stagnation+low-detail channel,
                # which stays silent when a fast few-step model converges
                # globally (all jumps collapse together -> median-relative
                # stagnation reads 0).  Inject a band-surgical refresh scaled
                # by the measured hi-band excess so the late-stage background
                # artifact can still be broken up.
                _shp = (lf_rho.shape if lf_rho is not None
                        else (x.shape[0], 1, *x.shape[2:]))
                _rho_val = _rho_band if _rho_band is not None else _LF_BAND_RHO
                _rho_eff = torch.full(_shp, _rho_val, device=x.device,
                                      dtype=x.dtype)
            if _rho_eff is not None and float(_rho_eff.max()) > 1e-4:
                x = x0 + _lf_refresh(x - x0, _rho_eff, surgical=_surgical,
                                     ramp_cap=lf_gains["ramp"],
                                     hi_boost=lf_resc_rw, gen=lf_noise_gen)
                if lf_diag is not None:
                    lf_diag["band"] = ("surgical" if _surgical else "flat")
            lf_m_prev = lf_m.detach().squeeze(1)
            lf_amp_prev = (lf_gains["amp"]
                           * (_lf_env_fade(k / n_cand,
                                           float(lf_evid.get("eg_n") or 0.0))
                              if lf_gains.get("env_fade") else 1.0))
            lf_rho_prev = lf_rho.detach()
            lf_applied_prev = True
            if lf_cum_next is not None:
                lf_cum = lf_cum_next
        if (dwell_on and dw_w is not None and s2 > 1e-8 and sn > 1e-8
                and not guard_active):
            # L1 per-pixel dwell: partial deterministic step for churning
            # pixels (x <- x_input + alpha * (x - x_input)); the final step
            # and the guard's protective pass-through always run at full
            # grid speed.
            _dw_env = _dwell_env(k / n_cand, dwell_ramp, dwell_fade_start,
                                 dwell_fade)
            _dw_w = dw_w
            if lf_on and lf_m is not None:
                # one controller per pixel: where the local field already
                # steers at a significant fraction of its amplitude, the
                # dwell stands down so the combined displacement deviation
                # stays inside the small-dose budget
                try:
                    _la = (lf_m.detach().float().abs()
                           / max(float(lf_gains["amp"]), 1e-6)).clamp(0.0, 1.0)
                    _dw_w = _dw_w * (1.0 - _la).to(_dw_w.dtype)
                except Exception:
                    pass
            _dw_alpha = (1.0 - dwell_max * _dw_env * _dw_w).clamp(0.35, 1.0)
            if float(_dw_alpha.min()) < 0.9999:
                try:
                    x = x_input + _dw_alpha.to(x.dtype) * (x - x_input)
                    if len(_LAST_ERRORS) < 64 or log_errors:
                        dwell_tel = {"w": round(float(_dw_w.mean()), 4),
                                     "wmax": round(float(_dw_w.max()), 4),
                                     "amin": round(float(_dw_alpha.min()), 4),
                                     "env": round(_dw_env, 4)}
                except Exception as e:
                    # never let the dwell break a run: one failure disables
                    # it for the rest of this sampling run
                    dwell_on = False
                    dw_w = None
                    if log_errors:
                        logging.warning("[A-FloPS-dwell] disabled for this "
                                        "run: %s", e)
        if s_up > 0:
            override_beta = None
            if is_adaptive_noise:
                if jmap is not None:
                    fresh_frac = (s_up * s_noise) ** 2 / max(sn * sn, 1e-12)
                    x0_norm = float(x0.float().norm().clamp_min(1e-8))
                    jmap_norm = float(jmap.float().norm().clamp_min(1e-8))
                    m0_step = _consistency_step(s, s2, space)
                    consistency = _compute_consistency(jmap_norm, x0_norm, m0_step)
                    override_beta = _predict_adaptive_beta(
                        jmap, err, eta_eff, s, space, sigma_data,
                        consistency, cfg, corr_used=corr_used,
                        fresh_frac=fresh_frac)
                if override_beta is None and cfg.get("_noise_beta_prior") is not None:
                    # No jump map yet (the very first fresh-noise draw) or a
                    # degenerate centroid: fall back to the probe's spectral
                    # whitepoint prior instead of plain white noise.
                    override_beta = float(cfg["_noise_beta_prior"])
            if is_adaptive_noise:
                fresh = noise_sampler(sigmas[i], sigmas[i + 1],
                                      override_beta=override_beta) * (s_up * s_noise)
            else:
                fresh = noise_sampler(sigmas[i], sigmas[i + 1]) * (s_up * s_noise)
            if cfg.get("color_conc_gate", True) and hasattr(noise_sampler, "last_beta"):
                _cb = float(noise_sampler.last_beta)
                if abs(_cb) > 1e-6:
                    _conc = _spectral_concentration(_cb)
                    _g = 1.0 / math.sqrt(max(_conc, 1.0))
                    _g = 1.0 + (_g - 1.0) * float(cfg.get("color_conc_strength", 1.0))
                    fresh = fresh * _g
            if space == "flow":
                x = x * ((1.0 - sn) / max(1.0 - s2, 1e-6)) + fresh
            else:
                x = x + fresh
        hist.append((u_log, u_sig, x0.detach(), x_input.detach()))
        if len(hist) > hist_cap:
            hist.pop(0)
        prev_err = err
        prev_cons = step_cons
        if len(_LAST_ERRORS) < 64 or log_errors:
            _LAST_ERRORS.append({"mode": "engine", "step": k,
                                 "sigma": round(s, 6), "sigma_next": round(sn, 6),
                                 "err": None if err is None or math.isinf(err) else round(err, 6),
                                 "cerr": (None if cerr is None
                                          else round(cerr, 6)),
                                 "tol": round(tol, 6), "order": eff,
                                 "eff_cap": eff_cap,
                                 "hist_cos": (round(hist_cos_val, 4)
                                              if hist_cos_val is not None else None),
                                 "blend_w": (round(blend_w_mean, 4)
                                             if blend_w_mean is not None else None),
                                 "ord_mean": (round(ord_tel[0], 3)
                                              if ord_tel is not None else None),
                                 "ord_max": (ord_tel[1]
                                             if ord_tel is not None else None),
                                 "ord_reset": (round(ord_tel[2], 4)
                                               if ord_tel is not None else None),
                                 "wmax": round(wmax, 4),
                                 "wmax_limit": round(wmax_limit, 4),
                                 "outlier": None if outlier_sig is None else round(outlier_sig, 6),
                                 "guard_sev": round(getattr(guard, "last_sev", 1.0), 4) if guard is not None else None,
                                 "space": use_space if len(hist) >= 1 else "log_sigma",
                                 "guard": guard_trip,
                                 "an_detected": an_detected,
                                 "corr": corr_used,
                                 "eta_eff": round(eta_eff, 6),
                                 "eta_stuck": (round(eta_stuck, 4)
                                               if eta_stuck is not None else None),
                                 "eta_jump_rel": (round(eta_jump_rel, 6)
                                                  if eta_jump_rel is not None else None),
                                 "eta_detail_abs": (round(eta_detail_abs, 4)
                                                    if eta_detail_abs is not None else None),
                                 "noise_beta": (round(noise_sampler.last_beta, 4)
                                                if noise_sampler is not None and
                                                hasattr(noise_sampler, "last_beta") else None),
                                 "aflops_lam": (round(aflops_lam, 4)
                                                if aflops_lam is not None else None),
                                 "lf": lf_diag,
                                 "dwell": dwell_tel,
                                 "calls": calls})
            if len(_LAST_ERRORS) > 64:
                del _LAST_ERRORS[:-64]
        if log_errors:
            _nb = (noise_sampler.last_beta if noise_sampler is not None and
                   hasattr(noise_sampler, "last_beta") else None)
            logging.info("[A-FloPS] k=%d sigma=%.4f err=%s tol=%.4g ord=%d "
                         "guard=%s an=%s corr=%s eta=%.3f nb=%s lam=%s calls=%d"
                         " hcos=%s ec=%d bw=%s",
                         k, s, "n/a" if err is None else f"{err:.4g}", tol, eff,
                         guard_trip, an_detected,
                         corr_used, eta_eff,
                         "n/a" if _nb is None else f"{_nb:.2f}",
                         "n/a" if aflops_lam is None else f"{aflops_lam:.3f}",
                         calls,
                         "n/a" if hist_cos_val is None else f"{hist_cos_val:.3f}",
                         eff_cap,
                         "n/a" if blend_w_mean is None else f"{blend_w_mean:.3f}")
        if callback is not None and not disable:
            callback({"x": x, "denoised": x0, "i": k, "sigma": sigmas[i], "sigma_hat": sigmas[i]})
            _progress(callback, "step", "progress", k, n_cand,
                      "A-FloPS sampling: step %d/%d" % (k, n_cand),
                      extra={"value": k, "total": n_cand})
    try:
        if (lf_dir_acc[1] > 0
                and bool(cfg.get("lf_dir_real_feedback", False))):
            _dir_val = round(lf_dir_acc[0] / lf_dir_acc[1], 4)
            _profs = []
            if bool(cfg.get("cond_probe_enabled", False)) and skey is not None:
                _p = _COND_PROBE_PROFILES_SCHED.get(skey)
                if _p is not None:
                    _profs.append(_p)
            if bool(cfg.get("probe_enabled", False)) and pkey is not None:
                _p = _PROBE_PROFILES.get(pkey)
                if _p is not None:
                    _profs.append(_p)
            for _p in _profs:
                _lst = _p.setdefault("dir_real", [])
                _lst.append(_dir_val)
                del _lst[:-5]
            if log_errors and _profs:
                logging.info("[A-FloPS-lf] run direction feedback: "
                             "dir_real=%.3f (%d steps measured)",
                             _dir_val, lf_dir_acc[1])
    except Exception:
        pass
    return x


def _subdivide_tail(sig, tail_steps, factor):
    if factor <= 1 or len(sig) < 4:
        return sig
    tail_steps = max(1, min(int(tail_steps), len(sig) - 2))
    split = len(sig) - tail_steps - 1
    parts = [sig[:split + 1]]
    tail = sig[split:]
    for a, b in zip(tail[:-1], tail[1:]):
        lo, hi = float(a), float(b)
        if hi <= 0.0:
            grid = torch.exp(torch.linspace(math.log(max(lo, 1e-8)), math.log(1e-8),
                                            int(factor) + 1, device=sig.device, dtype=sig.dtype))
            grid[-1] = 0.0
        else:
            grid = torch.exp(torch.linspace(math.log(max(lo, 1e-8)), math.log(hi),
                                            int(factor) + 1, device=sig.device, dtype=sig.dtype))
        parts.append(grid[1:])
    return torch.cat(parts)

_SCHEDULER_MAP = {
    "log_uniform": "exponential",
    "sigma_uniform": "simple",
    "timestep_uniform": "normal",
    "karras": "karras",
    "beta": "beta",
}

def aflops_scheduler_sigmas(model_sampling, steps, spacing="log_uniform",
                            shift=1.0, denoise=1.0, tail_subdivide=1, tail_steps=4):
    steps = int(steps)
    if steps < 1 or float(denoise) <= 0.0:
        return torch.FloatTensor([])
    total = steps if float(denoise) >= 1.0 else int(steps / float(denoise))
    sig = comfy.samplers.calculate_sigmas(model_sampling, _SCHEDULER_MAP[spacing], total)
    sig = sig[-(steps + 1):].detach().float()
    shift = float(shift)
    if shift != 1.0:
        s0 = float(sig[0])
        if s0 > 1.5:
            u = sig / max(s0, 1e-8)
            sig = s0 * (shift * u / (1.0 + (shift - 1.0) * u))
        else:
            sig = shift * sig / (1.0 + (shift - 1.0) * sig)
    sig[-1] = 0.0
    return _subdivide_tail(sig, tail_steps, tail_subdivide)
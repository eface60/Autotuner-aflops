from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


# ---- PD-AOS AS A FIRST-CLASS SAMPLER NAME (KSamplerSelect / KSampler dropdowns).
# The registration is in memory only: it appends the name to ComfyUI's own lists and patches the one attribute
# `comfy.samplers.ksampler()` resolves.  Guarded so that a ComfyUI version whose internals moved cannot break the
# whole pack -- the failure is logged, and `pd_aos_comfy.status()` reports exactly which end is missing.
try:
    from .research_samplers.pd_aos_comfy import register as _pd_aos_register
    _pd_aos_register()
except Exception as _pd_aos_exc:  # noqa: BLE001
    import logging as _pd_aos_logging
    _pd_aos_logging.warning("[PD-AOS] registration not applied: %s", _pd_aos_exc)

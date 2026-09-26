import torch

def exact_step(x, x0, s, s2):
    r = float(s2) / max(float(s), 1e-12)
    return r * x + (1.0 - r) * x0

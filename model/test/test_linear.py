import torch.nn as nn
import torch.nn.functional as F


# ── 定数 ────────────────────────────────────────────────────────────────
_EPS = 1e-5
_ACT_BITS = 8
_Qb = (1 << (_ACT_BITS - 1)) - 1  # 127  (int8 互換)


# ── ユーティリティ ──────────────────────────────────────────────────────
def _round_clip(x, min_val, max_val):
    """RoundClip(x, a, b) = max(a, min(b, round(x)))"""
    return x.round().clamp_(min_val, max_val)


# ── 活性化量子化 (absmax, per-token, 8-bit) ─────────────────────────────
def activation_quant_float(x):
    gamma = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=_EPS)
    scale = _Qb / gamma
    x_q = _round_clip(x * scale, -_Qb, _Qb) / scale
    return x_q, scale


# ── 重み量子化 (absmean, ternary {-1, 0, +1}) ──────────────────────────
def weight_quant_float(w):

    alpha = w.abs().mean()
    w_q = _round_clip(w / (alpha + _EPS), -1, 1) * alpha
    return w_q, alpha


# ── TerLinearCUDA ───────────────────────────────────────────────────────
class TerLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=None):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight
        # activation
        x_q, _ = activation_quant_float(x)
        x_quant = x + (x_q - x).detach()
        # weight
        w_q, _ = weight_quant_float(w)
        w_quant = w + (w_q - w).detach()
        # return
        return F.linear(x_quant, w_quant, self.bias)

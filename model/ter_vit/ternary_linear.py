"""
BitLinear layer for ViT-1.58b (ternary quantization with STE).

Reference
---------
"ViT-1.58b: Mobile Vision Transformers in the 1-bit Era"
 Zhengqing Yuan et al., 2024  —  https://arxiv.org/abs/2406.18051

Weight quantization  (absmean → ternary {-1, 0, +1})
    α = mean(|W|)
    W̃ = RoundClip(W / (α + ε), -1, 1)

Activation quantization  (absmax → 8-bit, per-token)
    γ = max(|x|)   per token
    x̃ = RoundClip(x × Qb / γ, -Qb, Qb)
    Qb = 2^(b-1) - 1 = 127   (b = 8)

BitLinear flow
    y = W̃ · Quant(LN(x))
    y_dequant = y × α γ / Qb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TerLinear"]

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
    """8-bit absmax 量子化 → 逆量子化 (float, STE 学習用).

    Returns
    -------
    x_q   : 量子化→逆量子化済みテンソル (float, 入力と同 shape)
    scale : Qb / γ  — per-token スケール係数
    """
    gamma = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=_EPS)
    scale = _Qb / gamma
    x_q = _round_clip(x * scale, -_Qb, _Qb) / scale
    return x_q, scale


# ── 重み量子化 (absmean, ternary {-1, 0, +1}) ──────────────────────────
def weight_quant_float(w):
    """Absmean 3 値量子化 → 逆量子化 (float, STE 学習用).

    Returns
    -------
    w_q   : 量子化→逆量子化済みテンソル (float, 入力と同 shape)
    alpha : mean(|W|)  — スケーリング係数 (= β in paper)
    """
    alpha = w.abs().mean()
    w_q = _round_clip(w / (alpha + _EPS), -1, 1) * alpha
    return w_q, alpha


# ── TerLinear (STE ベース) ──────────────────────────────────────────────
class TerLinear(nn.Linear):
    """3 値量子化線形層 — BitLinear (STE 学習版).

    nn.Linear の drop-in replacement として ViT の Transformer encoder
    内で使用する。学習時は STE (Straight-Through Estimator) により
    非微分可能な量子化関数を通して勾配を伝播する。

    Flow
    ----
    x → LayerNorm → activation_quant (STE) ─┐
                                             ├─ F.linear → y
    weight → weight_quant (STE) ────────────┘
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        w = self.weight
        x_norm = self.norm(x)

        # STE: forward で量子化→逆量子化、backward で恒等勾配
        x_q, _ = activation_quant_float(x_norm)
        x_quant = x_norm + (x_q - x_norm).detach()

        w_q, _ = weight_quant_float(w)
        w_quant = w + (w_q - w).detach()

        return F.linear(x_quant, w_quant, self.bias)

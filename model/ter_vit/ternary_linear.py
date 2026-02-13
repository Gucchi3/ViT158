import os
import json
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["TerLinearCUDA"]

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


# ── int8 量子化 (CUDA kernel 用) ────────────────────────────────────────
def activation_quant_int8(x):
    """8-bit absmax 量子化 (int8)."""
    gamma = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=_EPS)
    scale = _Qb / gamma
    x_q = _round_clip(x * scale, -_Qb, _Qb).to(torch.int8)
    return x_q, scale


def weight_quant_int8(w):
    """Absmean 3 値量子化 (int8)."""
    alpha = w.abs().mean()
    w_q = _round_clip(w / (alpha + _EPS), -1, 1).to(torch.int8)
    return w_q, alpha


# ── 逆量子化関数 ────────────────────────────────────────────────────────
def dequant_float(mult_result, scale_x, scale_w):
    return mult_result.float() * (scale_w / scale_x)


# ── TerLinearCUDA ───────────────────────────────────────────────────────
class TerLinearCUDA(nn.Linear):
    """3 値量子化線形層 — BitLinear (CUDA kernel 版).

    CUDA デバイス上かつ kernel がビルド済みなら int8 GEMM を使用。
    それ以外は STE ベースの PyTorch フォールバックへ自動切り替え。
    """

    def __init__(self, in_features, out_features, bias=None):
        super().__init__(in_features, out_features, bias)
        self.norm = nn.LayerNorm(in_features)
   

    def forward(self, x):
        w = self.weight
        x_norm = self.norm(x)
        x_q, _ = activation_quant_float(x_norm)
        x_quant = x_norm + (x_q - x_norm).detach()

        w_q, _ = weight_quant_float(w)
        w_quant = w + (w_q - w).detach()

        return F.linear(x_quant, w_quant, self.bias)

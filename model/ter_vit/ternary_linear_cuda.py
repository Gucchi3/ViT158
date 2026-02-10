"""
CUDA-accelerated BitLinear layer for ViT-1.58b.

Uses a custom CUDA kernel (ternary_qkv_gemm) for int8 × ternary GEMM
when available, with automatic fallback to the STE-based PyTorch path.

Reference
---------
"ViT-1.58b: Mobile Vision Transformers in the 1-bit Era"
 Zhengqing Yuan et al., 2024  —  https://arxiv.org/abs/2406.18051
"""

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
 
 
# ── CUDA kernel ロード ──────────────────────────────────────────────────
_qkv_kernel = None


# ── CUDA compiler loader ───────────────────────────────────────────────
def _try_load_kernel():
    """config.json の USE_CUDA_KERNEL=1 なら JIT コンパイルを試みる."""
    global _qkv_kernel

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            if json.load(f).get("USE_CUDA_KERNEL", 0) != 1:
                return
    except Exception:
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, but USE_CUDA_KERNEL=1 is set.")

    try:
        from torch.utils.cpp_extension import load

        kernel_dir = os.path.join(os.path.dirname(__file__), "kernel")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _qkv_kernel = load(
                name="qkv_kernel",
                sources=[
                    os.path.join(kernel_dir, "qkv_kernel.cpp"),
                    os.path.join(kernel_dir, "qkv_kernel.cu"),
                ],
                verbose=False,
            )
    except Exception as exc:
        _qkv_kernel = None
        raise SystemExit(f"Failed to build CUDA kernel: {exc}")


# ── CUDA compile ───────────────────────────────────────────────────────
_try_load_kernel()


# ── autograd.Function (CUDA forward + STE backward) ────────────────────
class _TernaryGemmSTE(torch.autograd.Function):
    """Forward: CUDA int8 kernel,  Backward: STE (恒等勾配)."""

    @staticmethod
    def forward(ctx, x_norm, weight, bias):
        ctx.save_for_backward(x_norm, weight, bias)

        # 量子化
        x_int8, x_scale = activation_quant_int8(x_norm)   # (*, D) → int8
        w_int8, alpha = weight_quant_int8(weight)        # (N, D) → int8

        # GEMM:  (M, K) × (N, K)^T → (M, N)
        x_2d = x_int8.reshape(-1, x_int8.shape[-1])
        y = _qkv_kernel.ternary_qkv_gemm(x_2d, w_int8)

        # 逆量子化:  y_float = y_int × α / x_scale
        #   α = mean(|W|),  x_scale = Qb / γ
        #   → y_float = y_int × α γ / Qb   (論文 式の通り)
        y = y * (alpha / x_scale.reshape(-1, 1))
        y = y.reshape(*x_norm.shape[:-1], -1)

        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_norm, weight, bias = ctx.saved_tensors
        # STE: 量子化を無視し、通常の線形層として勾配を伝播
        grad_x = grad_output @ weight
        grad_w = (
            grad_output.reshape(-1, grad_output.shape[-1]).t()
            @ x_norm.reshape(-1, x_norm.shape[-1])
        )
        grad_b = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if bias is not None
            else None
        )
        return grad_x, grad_w, grad_b


# ── TerLinearCUDA ───────────────────────────────────────────────────────
class TerLinearCUDA(nn.Linear):
    """3 値量子化線形層 — BitLinear (CUDA kernel 版).

    CUDA デバイス上かつ kernel がビルド済みなら int8 GEMM を使用。
    それ以外は STE ベースの PyTorch フォールバックへ自動切り替え。
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.norm = nn.LayerNorm(in_features)
        self.test = True

    def forward(self, x):
        w = self.weight
        x_norm = self.norm(x)
        
        #　現在実装テスト中
        if self.test == True:
            # 入力xと重みwを量子化
            x_q, x_scale = activation_quant_int8(x_norm)
            w_q, w_scale = weight_quant_int8(w)
            # 勾配計算のためのもの
            x_int = x_norm * x_scale
            w_int = w / w_scale
            # 勾配伝達をカット (x_q_d -> xのquantizeとdetachされたものの頭文字)
            x_q_d = x_int + (x_q.float() - x_int).detach()
            w_q_d = w_int + (w_q.float() - w_int).detach()
            # floatにキャスト（F.linearはint8の入力に対応して泣いため、いったんfloatにキャストしてからF.linearに入力する）
            # x * w を実行 biasはfalseが規定
            mul_res = F.linear(x_q_d, w_q_d, self.bias)
            result = dequant_float(mul_res, x_scale, w_scale)
            return result

        # CUDA kernel パス
        if _qkv_kernel is not None and x_norm.is_cuda:
            return _TernaryGemmSTE.apply(x_norm, w, self.bias)

        # フォールバック: STE + F.linear  (CPU / kernel 未ビルド時)
        x_q, _ = activation_quant_float(x_norm)
        x_quant = x_norm + (x_q - x_norm).detach()

        w_q, _ = weight_quant_float(w)
        w_quant = w + (w_q - w).detach()

        return F.linear(x_quant, w_quant, self.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F



# ── 量子化器 ────────────────────────────────────────────────────────────────
class Quantizer:
    # ── 定数 ────────────────────────────────────────────────────────────────
    def __init__(self):
        self.EPS = 1e-5
    
    
    # ── ユーティリティ ───────────────────────────────────────────────────────
    def _round_clip(self, x, min_val, max_val):
        return x.round().clamp(min_val, max_val)
    
    
    # ── 量子化 (absmax, n-bit, per_dim) ──────────────────────────────────────
    def to_bit_per_dim(self, x, bit=8, as_float=False, unsigned=False):
        gamma = x.abs().max(dim=-2, keepdim=True).values.clamp(min=self.EPS)

        if unsigned:
            x_max = x.max(dim=-2, keepdim=True).values.clamp(min=self.EPS)
            scale = (2**(bit) - 1) / x_max
        else:
            scale = ((2**(bit - 1) - 1)) / gamma

        if as_float:
            if unsigned:
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1) / scale
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1)) / scale
        else:
            if unsigned:
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1).to(x.dtype)
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1)).to(x.dtype)

        return x_q, scale
    
    
    # ── 量子化 (absmax, n-bit, per_toeken─────────────────────────────────────
    def to_bit_per_token(self, x, bit=8, as_float=False, unsigned=False):
        gamma = x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.EPS)

        if unsigned:
            x_max = x.max(dim=-1, keepdim=True).values.clamp(min=self.EPS)
            scale = (2**(bit) - 1) / x_max
        else:
            scale = ((2**(bit - 1) - 1)) / gamma

        if as_float:
            if unsigned:
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1) / scale
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1)) / scale
        else:
            if unsigned:
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1).to(x.dtype)
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1)).to(x.dtype)

        return x_q, scale
    
    
    # ── 量子化 (absmax, n-bit, per_tensor) ────────────────────────────────────
    def to_bit_per_tensor(self, x, bit=8, as_float=False, unsigned=False):
        gamma = x.abs().max().clamp(min=self.EPS)

        if unsigned:
            x_max = x.max().clamp(min=self.EPS)
            scale = (2**(bit) - 1) / x_max
        else:
            scale = ((2**(bit - 1) - 1)) / gamma

        if as_float:
            if unsigned:
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1) / scale
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1)) / scale
        else:
            if unsigned:
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1).to(x.dtype)
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1)).to(x.dtype)

        return x_q, scale
    
    
    
    # ── ３値量子化 (absmean, ternary {-1, 0, +1}, per_dim) ───────────────────────
    def to_ter_per_dim(self, w, as_float=False):
        alpha = w.abs().mean(dim=-2, keepdim=True)
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1).to(w.dtype)
        return w_q, alpha

    # ── ３値量子化 (absmean, ternary {-1, 0, +1}, per_token) ─────────────────────
    def to_ter_per_token(self, w, as_float=False):
        alpha = w.abs().mean(dim=-1, keepdim=True)
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1).to(w.dtype)
        return w_q, alpha

    # ── ３値量子化 (absmean, ternary {-1, 0, +1}, per_tensor) ────────────────────
    def to_ter_per_tensor(self, w, as_float=False):
        alpha = w.abs().mean()
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1).to(w.dtype)
        return w_q, alpha


# ── NEQ (per-token n-bit quantizer) ───────────────────────────────────────
class NEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.EPS = 1e-8

    def clip_round(self, x, min_val, max_val):
        return x.round().clamp(min_val, max_val)

    def forward(self, x, bit=8, unsigned=False, as_float=False):
        if unsigned:
            qmax = 2**bit - 1
            x_max = x.max(dim=-1, keepdim=True).values.clamp(min=self.EPS)
            scale = x_max / qmax
            if as_float:
                x_q = self.clip_round(x.clamp(min=0) / (scale + self.EPS), 0, qmax) * scale
            else:
                x_q = self.clip_round(x.clamp(min=0) / (scale + self.EPS), 0, qmax).to(x.dtype)
        else:
            qmax = 2**(bit - 1) - 1
            x_max = x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.EPS)
            scale = x_max / qmax
            if as_float:
                x_q = self.clip_round(x / (scale + self.EPS), -qmax, qmax) * scale
            else:
                x_q = self.clip_round(x / (scale + self.EPS), -qmax, qmax).to(x.dtype)

        if as_float:
            x_quant = x + (x_q - x).detach()
            return x_quant, scale

        return x_q, scale


# ── スケール変換器 ───────────────────────────────────────────────────────────
class ScaleConverter:
    @staticmethod
    def convert(x, scale_old, scale_new, bit=None, unsigned=False, eps=1e-8):
        x_new = torch.round(x * (scale_old / (scale_new + eps)))
        if bit is None:
            return x_new.to(x.dtype)
        if unsigned:
            qmin, qmax = 0, 2**bit - 1
        else:
            qmin, qmax = -(2**(bit - 1) - 1), (2**(bit - 1) - 1)
        return x_new.clamp(qmin, qmax).to(x.dtype)


# ── 逆量子化器 ───────────────────────────────────────────────────────────────
class  DeQuantizer:
    #def __init__(self):
    
    def to_float(self, x, scale_a, scale_b, scale_c=None):
        if scale_c is None:
            return x * scale_a * scale_b
        else:
            return x * scale_a * scale_b * scale_c
        




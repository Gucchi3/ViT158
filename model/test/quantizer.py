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
    
    
    # ── 量子化 (absmax, n-bit) ────────────────────────────────────────
    def to_bit_per_dim(self, x, bit=8, as_float=True, unsigned=False):
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
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1)
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1))

        return x_q, scale


    def to_bit_per_token(self, x, bit=8, as_float=True, unsigned=False):
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
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1)
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1))

        return x_q, scale


    def to_bit_per_tensor(self, x, bit=8, as_float=True, unsigned=False):
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
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(bit) - 1)
            else:
                x_q = self._round_clip(x * scale, -(2**(bit - 1) - 1), (2**(bit - 1) - 1))

        return x_q, scale

    
    
    # ── ３値量子化 (absmean, ternary {-1, 0, +1}) ───────────────────────────
    def to_ter_per_dim(self, w, as_float=True):
        alpha = w.abs().mean(dim=-2, keepdim=True)
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1)
        return w_q, alpha
    
    
    # ── ３値量子化 (absmean, ternary {-1, 0, +1}) ───────────────────────────
    def to_ter_per_token(self, w, as_float=True):
        alpha = w.abs().mean(dim=-1, keepdim=True)
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1)
        return w_q, alpha


    def to_ter_per_tensor(self, w, as_float=True):
        alpha = w.abs().mean()
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1)
        return w_q, alpha



# ── 逆量子化器 ───────────────────────────────────────────────────────────────
class  DeQuantizer:
    #def __init__(self):
    
    def to_float(self, x, scale_a, scale_b):
        return x * scale_a * scale_b


# ── Affine変換 ───────────────────────────────────────────────────────────────
class Affine(nn.Module):
    #def __init__(self):
        
    def forward(self):
        return
    


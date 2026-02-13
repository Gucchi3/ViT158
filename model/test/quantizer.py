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
    
    
    # ── 量子化 (absmax, n-bit, per_toeken─────────────────────────────────────
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
    
    
    # ── 量子化 (absmax, n-bit, per_tensor) ────────────────────────────────────
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
    
    
    
    # ── ３値量子化 (absmean, ternary {-1, 0, +1}, per_dim) ───────────────────────
    def to_ter_per_dim(self, w, as_float=True):
        alpha = w.abs().mean(dim=-2, keepdim=True)
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1)
        return w_q, alpha
    
    
    # ── ３値量子化 (absmean, ternary {-1, 0, +1}, per_token) ─────────────────────
    def to_ter_per_token(self, w, as_float=True):
        alpha = w.abs().mean(dim=-1, keepdim=True)
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1)
        return w_q, alpha
    
    # ── ３値量子化 (absmean, ternary {-1, 0, +1}, per_tensor) ────────────────────
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
    def __init__(self, D, bit=8, unsigned=False, init_gamma=1.0, init_beta=0.0):
        super().__init__()
        self.quantizer = Quantizer()
        self.bit = bit
        self.unsigned = unsigned
        self.D = D
        self.gamma = nn.Parameter(torch.full((D,), float(init_gamma)))
        self.beta = nn.Parameter(torch.full((D,), float(init_beta)))

    def forward(self, x):
        if x.shape[-1] != self.D:
            raise ValueError(f"Expected input embedding dim D={self.D}, but got {x.shape[-1]}")

        gamma_q, scale_gamma = self.quantizer.to_bit_per_tensor(self.gamma, bit=self.bit, as_float=True, unsigned=self.unsigned)

        gamma_ste = self.gamma + (gamma_q - self.gamma).detach()

        if self.unsigned:
            x_max = x.max().clamp(min=self.quantizer.EPS)
            scale_x = (2**(self.bit) - 1) / x_max
        else:
            x_gamma = x.abs().max().clamp(min=self.quantizer.EPS)
            scale_x = (2**(self.bit - 1) - 1) / x_gamma

        scale_beta = scale_x * scale_gamma
        beta_q = torch.round(self.beta * scale_beta) / (scale_beta + self.quantizer.EPS)
        beta_ste = self.beta + (beta_q - self.beta).detach()

        shape = [1] * x.ndim
        shape[-1] = self.D
        gamma_ste = gamma_ste.view(*shape)
        beta_ste = beta_ste.view(*shape)

        return x * gamma_ste + beta_ste
    


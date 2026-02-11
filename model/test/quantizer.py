import torch
import torch.nn as nn
import torch.nn.functional as F



# ── 量子化器 ────────────────────────────────────────────────────────────────
class Quantizer:
    # ── 定数 ────────────────────────────────────────────────────────────────
    def __init__(self):
        self.EPS = 1e-5
        self.ACT_BITS = 8
        self.QB = (1 << (self.ACT_BITS - 1)) - 1  # 127
    
    
    # ── ユーティリティ ───────────────────────────────────────────────────────
    def _round_clip(self, x, min_val, max_val):
        return x.round().clamp_(min_val, max_val)
    
    
    # ── 活性化量子化 (absmax, n-bit) ────────────────────────────────────────
    def activation_quant(self, x, conv=True, as_float=True, unsigned=False):
        """
        activation_quant 
        
        :param x: 入力x
        :param conv: Conv or ViT -> (すべてからの最大値か、per_token and per_headか)<->(dim=-1)
        :param as_float: Fake Quantization 
        :param unsigned: -127 ~ 127 or 0 ~ 255
        """
        if conv:
            gamma = x.abs().max().clamp_(min=self.EPS)
        else:
            gamma = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.EPS)

        if unsigned:
            if conv:
                x_max = x.max().clamp_(min=self.EPS)
            else:
                x_max = x.max(dim=-1, keepdim=True).values.clamp_(min=self.EPS)

            scale = (2**(self.ACT_BITS) - 1) / x_max
        else:
            scale = self.QB / gamma

        if as_float:
            if unsigned:
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(self.ACT_BITS) - 1) / scale
            else:
                x_q = self._round_clip(x * scale, -self.QB, self.QB) / scale
        else:
            if unsigned:
                x_q = self._round_clip(x.clamp(min=0) * scale, 0, 2**(self.ACT_BITS) - 1)
            else:
                x_q = self._round_clip(x * scale, -self.QB, self.QB)

        return x_q, scale
    
    
    # ── 重み量子化 (absmean, ternary {-1, 0, +1}) ───────────────────────────
    def weight_quant(self, w, as_float=True):
        alpha = w.abs().mean()
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1)
        return w_q, alpha



# # ── 逆量子化器 ───────────────────────────────────────────────────────────────
class  DeQuantizer:
    #def __init__(self):
    
    def dequantize(self, x, scale_a, scale_b):
        return x * scale_a * scale_b




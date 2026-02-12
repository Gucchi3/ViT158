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
        return x.round().clamp_(min_val, max_val)
    
    
    # ── 量子化 (absmax, n-bit) ────────────────────────────────────────
    def to_bit(self, x, bit=8, conv=False, as_float=True, unsigned=False):
        """
        to_bit 
        
        :param x       : 入力
        :param bit     : 量子化bit数
        :param conv    : Conv or ViT -> (すべてからの最大値か、per_token and per_headか)<->(dim=-1)
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
    def to_ter(self, w, as_float=True):
        """
        to_ter
        
        :param w       : 入力
        :param as_float: Fake Quantization 
        """
        alpha = w.abs().mean()
        if as_float:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1) * alpha
        else:
            w_q = self._round_clip(w / (alpha + self.EPS), -1, 1)
        return w_q, alpha



# # ── 逆量子化器 ───────────────────────────────────────────────────────────────
class  DeQuantizer:
    #def __init__(self):
    
    def to_float(self, x, scale_a, scale_b):
        return x * scale_a * scale_b



# quantizer.to_bit()
# quantizer.to_ter()
# dequantizer.to_float()
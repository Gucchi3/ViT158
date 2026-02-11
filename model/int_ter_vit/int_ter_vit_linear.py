import os, sys
import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .quantizer import Quantizer_float, Quantizer_int, DeQuantizer_float, activation_quant_float, activation_quant_int8, weight_quant_float, weight_quant_int8, dequant_float
except ImportError:
    parent_dir = Path(__file__).resolve().parents[2]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from .quantizer import Quantizer_float, Quantizer_int, DeQuantizer_float, activation_quant_float, activation_quant_int8, weight_quant_float, weight_quant_int8, dequant_float


__all__ = ["TerLinearCUDA"]


# ── TerLinearCUDA ───────────────────────────────────────────────────────
class IntTerLinear(nn.Linear):
    """3 値量子化線形層 — BitLinear (CUDA kernel 版).

    CUDA デバイス上かつ kernel がビルド済みなら int8 GEMM を使用。
    それ以外は STE ベースの PyTorch フォールバックへ自動切り替え。
    """

    def __init__(self, in_features, out_features, bias=None):
        super().__init__(in_features, out_features, bias)
        self.norm = nn.LayerNorm(in_features)
        self.test = False #? test運用フラッグ

    def forward(self, x):
        w = self.weight
        x_norm = self.norm(x)
        
        #? 正常動作確認済み
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



        # フォールバック: STE + F.linear  (CPU / kernel 未ビルド時)
        x_q, _ = activation_quant_float(x_norm)
        x_quant = x_norm + (x_q - x_norm).detach()

        w_q, _ = weight_quant_float(w)
        w_quant = w + (w_q - w).detach()

        return F.linear(x_quant, w_quant, self.bias)

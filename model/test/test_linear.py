import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


# ── import ──────────────────────────────────────────────────────────────
try:
    from .quantizer import Quantizer, Affine
except ImportError:
    parent_dir = Path(__file__).resolve().parents[2]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from .quantizer import Quantizer, Affine



# ── TerLinear ───────────────────────────────────────────────────────────
class TerLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_layer_norm=True):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.use_layer_norm = use_layer_norm
        self.norm = Affine(in_features) if use_layer_norm else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        w = self.weight
        b = self.bias
        # activation
        x_q, scale_x = self.quantizer.to_bit_per_tensor(x, bit=8, as_float=True, unsigned=False)
        x_quant = x + (x_q - x).detach()
        # weight
        w_q, alpha_w = self.quantizer.to_ter_per_tensor(w, as_float=True)
        w_quant = w + (w_q - w).detach()
        # bias (s_b = s_x * s_w, s_w = 1 / alpha_w)
        if b is not None:
            scale_w = 1.0 / (alpha_w + self.quantizer.EPS)
            scale_b = scale_x * scale_w
            b_q = torch.round(b * scale_b) / (scale_b + self.quantizer.EPS)
            b_quant = b + (b_q - b).detach()
        else:
            b_quant = None
        # return
        return F.linear(x_quant, w_quant, b_quant)


# ── Q_Linear ───────────────────────────────────────────────────────────
class Q_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_layer_norm=True):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.use_layer_norm = use_layer_norm
        self.norm = Affine(in_features) if use_layer_norm else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        w = self.weight
        b = self.bias
        # activation
        x_q, scale_x = self.quantizer.to_bit_per_tensor(x, bit=8, as_float=True, unsigned=False)
        x_quant = x + (x_q - x).detach()
        # weight
        w_q, scale_w = self.quantizer.to_bit_per_tensor(w, bit=8, as_float=True, unsigned=False)
        w_quant = w + (w_q - w).detach()
        # bias (s_b = s_x * s_w)
        if b is not None:
            scale_b = scale_x * scale_w
            b_q = torch.round(b * scale_b) / (scale_b + self.quantizer.EPS)
            b_quant = b + (b_q - b).detach()
        else:
            b_quant = None
        # return
        return F.linear(x_quant, w_quant, b_quant)

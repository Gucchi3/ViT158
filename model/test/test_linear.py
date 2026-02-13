import os, sys
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


# ── import ──────────────────────────────────────────────────────────────
try:
    from .quantizer import Quantizer
except ImportError:
    parent_dir = Path(__file__).resolve().parents[2]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from .quantizer import Quantizer



# ── TerLinear ───────────────────────────────────────────────────────────
class TerLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_layer_norm=True):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.use_layer_norm = use_layer_norm
        self.norm = nn.LayerNorm(in_features) if use_layer_norm else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        w = self.weight
        # activation
        x_q, _ = self.quantizer.to_bit_per_tensor(x, bit=8, as_float=True, unsigned=False)
        x_quant = x + (x_q - x).detach()
        # weight
        w_q, _ = self.quantizer.to_ter_per_tensor(w, as_float=True)
        w_quant = w + (w_q - w).detach()
        # return
        return F.linear(x_quant, w_quant, self.bias)


# ── Q_Linear ───────────────────────────────────────────────────────────
class Q_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_layer_norm=True):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.use_layer_norm = use_layer_norm
        self.norm = nn.LayerNorm(in_features) if use_layer_norm else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        w = self.weight
        # activation
        x_q, _ = self.quantizer.to_bit_per_tensor(x, bit=8, as_float=True, unsigned=False)
        x_quant = x + (x_q - x).detach()
        # weight
        w_q, _ = self.quantizer.to_bit_per_tensor(w, bit=8, as_float=True, unsigned=False)
        w_quant = w + (w_q - w).detach()
        # return
        return F.linear(x_quant, w_quant, self.bias)

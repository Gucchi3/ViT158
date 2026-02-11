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



# ── TerLinearCUDA ───────────────────────────────────────────────────────
class TerLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=None):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = self.norm(x)
        w = self.weight
        # activation
        x_q, _ = self.quantizer.activation_quant(x)
        x_quant = x + (x_q - x).detach()
        # weight
        w_q, _ = self.quantizer.weight_quant(w)
        w_quant = w + (w_q - w).detach()
        # return
        return F.linear(x_quant, w_quant, self.bias)

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


# ── import ──────────────────────────────────────────────────────────────
try:
    from .quantizer import Quantizer, NEQ
except ImportError:
    parent_dir = Path(__file__).resolve().parents[2]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from .quantizer import Quantizer, NEQ



# ── TerLinear ───────────────────────────────────────────────────────────
class TerLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_layer_norm=True, bit=8, as_float=False, unsigned=False):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.use_layer_norm = use_layer_norm
        self.bit = bit
        self.as_float = as_float
        self.unsigned = unsigned
        self.norm = Affine(in_features, bit=bit, unsigned=unsigned, as_float=as_float) if use_layer_norm else nn.Identity()

    def forward(self, x, input_scale=None, return_scale=False):
        x = self.norm(x)
        w = self.weight
        b = self.bias

        # activation
        x_q, scale_x = self.quantizer.to_bit_per_tensor(
            x,
            bit=self.bit,
            as_float=self.as_float,
            unsigned=self.unsigned,
        )
        x_quant = x + (x_q - x).detach()
        # weight
        w_q, alpha_w = self.quantizer.to_ter_per_tensor(w, as_float=self.as_float)
        w_quant = w + (w_q - w).detach()
        # bias (s_b = s_x * s_w, s_w = 1 / alpha_w)
        scale_w = 1.0 / (alpha_w + self.quantizer.EPS)
        scale_b = scale_x * scale_w
        if b is not None:
            b_q = torch.round(b * scale_b)
            b_quant = b + (b_q - b).detach()
        else:
            b_quant = None
        self.last_input_scale = scale_x
        self.last_weight_scale = scale_w
        self.last_output_scale = scale_b
        # return
        out = F.linear(x_quant, w_quant, b_quant)
        if return_scale:
            return out, scale_b
        return out


# ── Q_Linear ────────────────────────────────────────────────────────────
class Q_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_layer_norm=True, bit=8, as_float=False, unsigned=False):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.use_layer_norm = use_layer_norm
        self.bit = bit
        self.as_float = as_float
        self.unsigned = unsigned
        self.norm = Affine(in_features, bit=bit, unsigned=unsigned, as_float=as_float) if use_layer_norm else nn.Identity()

    def forward(self, x, input_scale=None, return_scale=False):
        x = self.norm(x)
        w = self.weight
        b = self.bias

        # activation
        if input_scale is None:
            x_q, scale_x = self.quantizer.to_bit_per_tensor(
                x,
                bit=self.bit,
                as_float=self.as_float,
                unsigned=self.unsigned,
            )
            x_quant = x + (x_q - x).detach()
        else:
            scale_x = input_scale
            x_quant = x
        # weight
        w_q, scale_w = self.quantizer.to_bit_per_tensor(
            w,
            bit=self.bit,
            as_float=self.as_float,
            unsigned=self.unsigned,
        )
        w_quant = w + (w_q - w).detach()
        # bias (s_b = s_x * s_w)
        scale_b = scale_x * scale_w
        if b is not None:
            b_q = torch.round(b * scale_b)
            b_quant = b + (b_q - b).detach()
        else:
            b_quant = None
        self.last_input_scale = scale_x
        self.last_weight_scale = scale_w
        self.last_output_scale = scale_b
        # return
        out = F.linear(x_quant, w_quant, b_quant)
        if return_scale:
            return out, scale_b
        return out


# ── Q_Conv2d ────────────────────────────────────────────────────────────
class Q_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bit=8, as_float=False, unsigned=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.quantizer = Quantizer()
        self.bit = bit
        self.as_float = as_float
        self.unsigned = unsigned

    def forward(self, x, input_scale=None, return_scale=False):
        w = self.weight
        b = self.bias

        if input_scale is None:
            x_q, scale_x = self.quantizer.to_bit_per_tensor(
                x,
                bit=self.bit,
                as_float=self.as_float,
                unsigned=self.unsigned,
            )
            x_quant = x + (x_q - x).detach()
        else:
            scale_x = input_scale
            x_quant = x

        w_q, scale_w = self.quantizer.to_bit_per_tensor(
            w,
            bit=self.bit,
            as_float=self.as_float,
            unsigned=self.unsigned,
        )
        w_quant = w + (w_q - w).detach()

        scale_b = scale_x * scale_w
        if b is not None:
            b_q = torch.round(b * scale_b)
            b_quant = b + (b_q - b).detach()
        else:
            b_quant = None

        self.last_input_scale = scale_x
        self.last_weight_scale = scale_w
        self.last_output_scale = scale_b

        out = F.conv2d(x_quant, w_quant, b_quant, stride=self.stride, padding=self.padding)
        if return_scale:
            return out, scale_b
        return out


# ── Affine変換 ───────────────────────────────────────────────────────────────
class Affine(nn.Module):
    def __init__(self, D, bit=8, unsigned=False, as_float=False, init_gamma=1.0, init_beta=0.0):
        super().__init__()
        self.quantizer = Quantizer()
        self.bit = bit
        self.unsigned = unsigned
        self.as_float = as_float
        self.D = D
        self.gamma = nn.Parameter(torch.full((D,), float(init_gamma)))
        self.beta = nn.Parameter(torch.full((D,), float(init_beta)))

    def forward(self, x, input_scale=None, return_scale=False):
        if x.shape[-1] != self.D:
            raise ValueError(f"Expected input embedding dim D={self.D}, but got {x.shape[-1]}")

        gamma_q, scale_gamma = self.quantizer.to_bit_per_tensor(self.gamma, bit=self.bit, as_float=False, unsigned=self.unsigned)

        gamma_ste = self.gamma + (gamma_q - self.gamma).detach()

        if self.unsigned:
            x_max = x.max().clamp(min=self.quantizer.EPS)
            scale_x = (2**(self.bit) - 1) / x_max
        else:
            x_gamma = x.abs().max().clamp(min=self.quantizer.EPS)
            scale_x = (2**(self.bit - 1) - 1) / x_gamma

        scale_beta = scale_x * scale_gamma
        beta_q = torch.round(self.beta * scale_beta)
        beta_ste = self.beta + (beta_q - self.beta).detach()
        self.last_input_scale = scale_x
        self.last_gamma_scale = scale_gamma
        self.last_output_scale = scale_beta

        shape = [1] * x.ndim
        shape[-1] = self.D
        gamma_ste = gamma_ste.view(*shape)
        beta_ste = beta_ste.view(*shape)

        return x * gamma_ste + beta_ste
    


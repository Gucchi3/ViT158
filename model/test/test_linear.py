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
    def __init__(self, in_features, out_features, bias=True, use_norm=False, bit=8, as_float=False, unsigned=False):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.use_norm = use_norm
        self.bit = bit
        self.as_float = as_float
        self.unsigned = unsigned
        self.norm = Affine(in_features, bit=bit, unsigned=unsigned, as_float=as_float) if use_norm else nn.Identity()

    def forward(self, x, input_scale=None, return_scale=False):
        if self.use_norm:
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
        # return
        out = F.linear(x_quant, w_quant, b_quant)
        if return_scale:
            return out, scale_b
        return out


# ── Q_Linear ────────────────────────────────────────────────────────────
class Q_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_norm=False, bit=8, as_float=False, unsigned=False):
        super().__init__(in_features, out_features, bias)
        self.quantizer = Quantizer()
        self.use_norm = use_norm
        self.bit = bit
        self.as_float = as_float
        self.unsigned = unsigned
        self.norm = Affine(in_features, bit=bit, unsigned=unsigned, as_float=as_float) if use_norm else nn.Identity()

    def forward(self, x, input_scale=None, return_scale=False):
        if self.use_norm:
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
        # return
        out = F.linear(x_quant, w_quant, b_quant)
        if return_scale:
            return out, scale_b
        return out


# ── Q_Conv2d ────────────────────────────────────────────────────────────
class Q_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_norm=False, bit=8, as_float=False, unsigned=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.quantizer = Quantizer()
        self.use_norm = use_norm
        self.norm = nn.BatchNorm2d(in_channels) if use_norm else nn.Identity()
        self.bit = bit
        self.as_float = as_float
        self.unsigned = unsigned

    def forward(self, x, input_scale=None, return_scale=False):
        if self.use_norm:
            x = self.norm(x)
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


        shape = [1] * x.ndim
        shape[-1] = self.D
        gamma_ste = gamma_ste.view(*shape)
        beta_ste = beta_ste.view(*shape)

        return x * gamma_ste + beta_ste


# ── LST_Act_Quant (LSQ+風 活性化量子化) ─────────────────────────────────────
class LST_Act_Quant(nn.Module):
    """LSQ+ベースの可読性重視版アクティベーション量子化器。

    疑似量子化入力（FP）`x` と `scale_x` を受け取る。
    本実装では `x` をFP前提で扱い、`scale_x` はインターフェース互換のために受け取る。
    """

    def __init__(self, num_bits=4, init_steps=100, init_lr=0.01):
        super().__init__()
        self.quantizer = Quantizer()
        self.num_bits = num_bits
        self.qmin = 0
        self.qmax = 2**num_bits - 1

        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))

        self.init_steps = init_steps
        self.init_lr = init_lr
        self.initialized = False

    def _fake_quant_with_ste(self, x_fp):
        scaled = (x_fp - self.offset) / (self.scale + self.quantizer.EPS)
        clipped = scaled.clamp(self.qmin, self.qmax)
        rounded = self.quantizer._round_clip(clipped, self.qmin, self.qmax)
        q_ste = clipped + (rounded - clipped).detach()
        return q_ste * self.scale + self.offset

    def _initialize_by_mse(self, x_fp):
        if self.initialized:
            return

        with torch.enable_grad():
            x_min = x_fp.min().detach().item()
            x_max = x_fp.max().detach().item()

            if x_max > x_min:
                init_scale = (x_max - x_min) / (self.qmax - self.qmin)
            else:
                init_scale = 1.0

            self.scale.data.fill_(init_scale)
            self.offset.data.fill_(x_min)

            optimizer = torch.optim.SGD([self.scale, self.offset], lr=self.init_lr)
            for _ in range(self.init_steps):
                optimizer.zero_grad()
                x_hat = self._fake_quant_with_ste(x_fp)
                mse = torch.mean((x_hat - x_fp) ** 2)
                mse.backward()
                optimizer.step()

                with torch.no_grad():
                    self.scale.data.clamp_(min=self.quantizer.EPS)

        self.initialized = True

    def forward(self, x, scale_x):

        if self.training and (not self.initialized):
            self._initialize_by_mse(x)

        if not self.initialized:
            return x

        return self._fake_quant_with_ste(x)



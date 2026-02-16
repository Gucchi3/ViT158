import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


# ── import ──────────────────────────────────────────────────────────────
try:
    from .quantizer import Quantizer, NEQ
    from .stats_quantizer import StatsQuantizer_CGA
except ImportError:
    parent_dir = Path(__file__).resolve().parents[2]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from .quantizer import Quantizer, NEQ
    from .stats_quantizer import StatsQuantizer_CGA



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


# ── StatsQ-CGA Linear/Conv ──────────────────────────────────────────────
class StatsQ_CGA_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=8, boundary_range=0.005, granularity="per_tensor", ternary=False):
        super().__init__(in_features, out_features, bias)
        self.x_quantizer = LSQ_Act_Quant(num_bits=bit)
        self.w_quantizer = StatsQuantizer_CGA(num_bits=bit, clip_learnable=False, boundaryRange=boundary_range, granularity=granularity, ternary=ternary)

    def forward(self, x):
        x_quant = self.x_quantizer(x, scale_x=None)
        w_quant = self.w_quantizer(self.weight)
        return F.linear(x_quant, w_quant, self.bias)


class StatsQ_CGA_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bit=8, boundary_range=0.005, granularity="per_tensor"):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.x_quantizer = LSQ_Act_Quant(num_bits=bit)
        self.w_quantizer = StatsQuantizer_CGA(
            num_bits=bit,
            clip_learnable=False,
            boundaryRange=boundary_range,
            granularity=granularity,
        )

    def forward(self, x):
        x_quant = self.x_quantizer(x, scale_x=None)

        w = self.weight
        w_flat = w.reshape(w.shape[0], -1)
        w_quant = self.w_quantizer(w_flat).reshape_as(w)

        return F.conv2d(x_quant, w_quant, self.bias, stride=self.stride, padding=self.padding)


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


# ── LSQ_Act_Quant (LSQ+ 活性化量子化) ─────────────────────────────────────
class LSQ_Act_Quant(nn.Module):
    """LSQ+ベースの可読性重視版アクティベーション量子化器。

    疑似量子化入力（FP）`x` と `scale_x` を受け取る。
    本実装では `x` をFP前提で扱い、`scale_x` はインターフェース互換のために受け取る。
    """

    def __init__(self, num_bits=8, init_steps=100, init_lr=0.01, mse_init_steps=16):
        super().__init__()
        self.quantizer = Quantizer()
        self.num_bits = num_bits
        self.qmin = 0
        self.qmax = 2**num_bits - 1

        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))

        self.init_steps = init_steps
        self.init_lr = init_lr
        self.mse_init_steps = mse_init_steps
        self.initialized = False
        self.mse_initialized = False

        # LSQ+ 論文式(6)(7)に合わせた初期化用バッファ
        # 収集した各バッチの x_min/x_max を平均し、
        # s_init = (x_max - x_min) / (p - n), beta_init = x_min - n*s_init を用いる
        self.register_buffer("_init_batch_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_init_xmin_sum", torch.tensor(0.0))
        self.register_buffer("_init_xmax_sum", torch.tensor(0.0))
        self.register_buffer("_mse_step_count", torch.tensor(0, dtype=torch.long))

        self._mse_optimizer = None


    def _fake_quant_with_ste(self, x_fp):
        scaled = (x_fp - self.offset) / (self.scale + self.quantizer.EPS)
        clipped = scaled.clamp(self.qmin, self.qmax)
        rounded = self.quantizer._round_clip(clipped, self.qmin, self.qmax)
        q_ste = clipped + (rounded - clipped).detach()
        return q_ste * self.scale + self.offset


    def _collect_init_stats(self, x_fp):
        x_detached = x_fp.detach()
        batch_xmin = x_detached.min()
        batch_xmax = x_detached.max()

        self._init_xmin_sum += batch_xmin
        self._init_xmax_sum += batch_xmax
        self._init_batch_count += 1


    def _initialize_from_stats(self):
        # model.eval() でもこの関数は実行されるが、trainを通してinitialize=Trueになっていれば、
        # 次の行ですぐにreturnされる。
        if self.initialized:
            return

        count = int(self._init_batch_count.item())
        if count <= 0:
            return

        x_min = self._init_xmin_sum / count
        x_max = self._init_xmax_sum / count

        init_scale = (x_max - x_min) / max(self.qmax - self.qmin, 1)
        init_scale = torch.clamp(init_scale, min=self.quantizer.EPS)
        init_offset = x_min - self.qmin * init_scale

        with torch.no_grad():
            self.scale.copy_(init_scale.to(self.scale.dtype))
            self.offset.copy_(init_offset.to(self.offset.dtype))
            self.initialized = True


    def _mse_optimize_step(self, x_fp):
        # ここで初期化済みならreturn
        if self.mse_initialized:
            return

        if int(self.mse_init_steps) <= 0:
            self.mse_initialized = True
            return

        if self._mse_optimizer is None:
            self._mse_optimizer = torch.optim.SGD([self.scale, self.offset], lr=self.init_lr)

        x_detached = x_fp.detach()

        self._mse_optimizer.zero_grad()
        x_hat = self._fake_quant_with_ste(x_detached)
        mse = torch.mean((x_hat - x_detached) ** 2)
        mse.backward()
        self._mse_optimizer.step()

        with torch.no_grad():
            self.scale.clamp_(min=self.quantizer.EPS)

        self._mse_step_count += 1
        if int(self._mse_step_count.item()) >= int(self.mse_init_steps):
            self.mse_initialized = True



    def forward(self, x, scale_x):
        if not self.initialized:
            self._collect_init_stats(x)

            # train時は複数バッチ分を集めてから初期化
            if self.training:
                if int(self._init_batch_count.item()) >= max(int(self.init_steps), 1):
                    self._initialize_from_stats()
            # eval時は初回バッチの統計で即初期化（フォールバック）
            else:
                self._initialize_from_stats()

        if not self.initialized:
            return x

        # LSQ+ Sec.3.2.2: 統計初期値の後に、式(10)のMSEを数バッチ最適化
        # 学習初期のキャリブレーションとしてscale/offsetのみ内部更新する。
        if self.training and (not self.mse_initialized):
            self._mse_optimize_step(x)
            return x

        return self._fake_quant_with_ste(x)


# ── LSQ_Weight_Quant (LSQ+ 重み量子化) ─────────────────────────────────────








import torch
import torch.nn as nn
import math
import numpy as np

## create 1D mask
def create_mask(s2, prob):
    raw = torch.zeros((s2,))
    raw[:int((1-prob) * s2)] = 1.0/(1.0-prob)
    ridx = torch.randperm(s2)
    return raw[ridx]

def round_pass(x):
    """Straight-Through Estimator for rounding"""
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def grad_scale(x, scale):
    """Gradient scaling"""
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def clip(x, eps):
    """Differentiable clipping"""
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()

def modify_grad(x, mask):
    """
    Gradient masking for CGA (Algorithm 1)
    mask: 1 for weights to update, 0 for weights to freeze
    """
    return x * mask

# ── ??? ───────────────────────────────────────────────────────────────
class TrackOscillation(nn.Module):
    """
    論文Sec.4の振動追跡 + Sec.5.2 CGAの高信頼度重みフリーズ
    修正版: 振動が「少ない」(低いema_oscillation)重みをフリーズ
    """

    def __init__(self, momentum=0.01, freeze_threshold=0, use_ema_x_int=True):
        super(TrackOscillation, self).__init__()
        self.momentum = momentum

        self.prev_x_int = None
        self.prev_switch_dir = None

        # Statistics to log
        self.ema_oscillation = None
        self.oscillated_sum = None
        self.total_oscillation = None
        self.iters_since_reset = 0

        # Extra variables for weight freezing
        self.freeze_threshold = freeze_threshold
        self.use_ema_x_int = use_ema_x_int
        self.frozen = None
        self.frozen_x_int = None
        self.ema_x_int = None

    def __call__(self, x_int, skip_tracking=False, *args, **kwargs):
       
        # Apply weight freezing
        if self.frozen is not None:
            x_int = ~self.frozen * x_int + self.frozen * self.frozen_x_int

        if skip_tracking:
            return x_int

        with torch.no_grad():
            # Check if everything is correctly initialized
            self.check_init(x_int)

            # Detect difference in x_int
            delta_x_int = torch.round(self.prev_x_int - x_int).detach()  # {-1, 0, 1}
            switch_dir = torch.sign(delta_x_int)
            switched = delta_x_int != 0

            # Detect oscillation
            oscillated = (self.prev_switch_dir * switch_dir) == -1
            self.ema_oscillation = (
                self.momentum * oscillated + (1 - self.momentum) * self.ema_oscillation
            )

            # Update prev_switch_dir for switched variables
            self.prev_switch_dir[switched] = switch_dir[switched]
            self.prev_x_int = x_int
            self.oscillated_sum = oscillated.sum()
            self.total_oscillation += oscillated
            self.iters_since_reset += 1

            # Update EMA of x_int (always, regardless of freezing)
            if self.use_ema_x_int:
                self.ema_x_int = self.momentum * x_int + (1 - self.momentum) * self.ema_x_int

            # Freeze high-confidence weights (LOW oscillation = high confidence)
            # 論文のCGA: BR外の高信頼度重みを先にフリーズ
            if self.freeze_threshold > 0:
                # 修正: 振動が少ない(低いema_oscillation)重みをフリーズ
                freeze_weights = self.ema_oscillation < self.freeze_threshold
                self.frozen[freeze_weights] = True
                if self.use_ema_x_int:
                    self.frozen_x_int[freeze_weights] = torch.round(self.ema_x_int[freeze_weights])
                else:
                    self.frozen_x_int[freeze_weights] = x_int[freeze_weights]

        return x_int

    def check_init(self, x_int):
        if self.prev_x_int is None:
            self.prev_switch_dir = torch.zeros_like(x_int)
            self.prev_x_int = x_int.detach()
            self.ema_oscillation = torch.zeros_like(x_int)
            self.oscillated_sum = 0
            self.total_oscillation = torch.zeros_like(x_int)
        else:
            assert (
                self.prev_x_int.shape == x_int.shape
            ), "Tracking shape does not match current tensor shape."

        # For weight freezing
        if self.frozen is None and self.freeze_threshold > 0:
            self.frozen = torch.zeros_like(x_int, dtype=torch.bool)
            self.frozen_x_int = torch.zeros_like(x_int)
            if self.use_ema_x_int:
                self.ema_x_int = x_int.detach().clone()


# ── StatsQuantizer ────────────────────────────────────────────────────
class StatsQuantizer(nn.Module):
    """
    StatsQ: Statistical Weight Quantization
    論文 Eq.(7) の厳密な実装
    
    s = 2 * E(|W|)  (row-wise or specified dimension)
    W_s = W / s
    W_c = Clip(W_s, -α/2, α/2 - ε)
    W_q = s * (⌊W_c * n - 0.5⌋ + 0.5) / n
    
    where n = 2^(b-1), b = num_bits
    """
    def __init__(self, num_bits, clip_learnable=False):
        super(StatsQuantizer, self).__init__()
        self.num_bits = num_bits
        init_act_clip_val = 2.0

        self.clip_val = nn.Parameter(
            torch.Tensor([init_act_clip_val]), 
            requires_grad=clip_learnable
        )
        self.s = None

    def forward(self, weight):
        real_weights = weight

        # 統計的スケーリングファクター: s = 2 * E(|W|)
        if len(weight.shape) == 2:
            # Fully-connected: row-wise (dim, 1)
            scaling_factor = 2 * torch.mean(abs(real_weights), dim=1, keepdim=True)
        elif len(weight.shape) == 3:
            # 3D tensor: (1, dim, 1)
            scaling_factor = 2 * torch.mean(
                torch.mean(abs(real_weights), dim=-1, keepdim=True), 
                dim=0, keepdim=True
            )
        else:
            raise ValueError(f"Unsupported weight shape: {weight.shape}")

        scaling_factor = scaling_factor.detach()
        self.s = scaling_factor.squeeze().detach()
        
        # スケーリングとクリッピング
        scaled_weights = real_weights / scaling_factor
        cliped_weights = torch.clamp(
            scaled_weights, 
            min=(-self.clip_val/2), 
            max=(self.clip_val/2) - 1e-6
        )
        
        # 量子化: n = 2^(b-1)
        n = float(2 ** (self.num_bits - 1))
        quan_weights_no_grad = scaling_factor * (
            (torch.round((cliped_weights) * n - 0.5) + 0.5) / n
        )
        
        # STE (Straight-Through Estimator)
        quan_weights = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        return quan_weights


# ── StatsQuantizer_CGA ────────────────────────────────────────────────
class StatsQuantizer_CGA(nn.Module):
    """
    StatsQ + CGA (Confidence-Guided Annealing)
    論文 Algorithm 1 に厳密に準拠
    
    CGA: BR外（高信頼度）の重みを先にフリーズし、BR内の重みのみ更新
    
    ternary=True の場合、1.58-bit 三値量子化 {-1, 0, +1} に切り替え:
      s = E(|W|)  (granularity に従う)
      W_q = s * RoundClip(W / s, -1, 1)
    CGA の BR は決定境界 ±0.5 の近傍で定義される
    """
    def __init__(self, num_bits, clip_learnable=False, boundaryRange=0.005, granularity="per_tensor", ternary=False):
        super(StatsQuantizer_CGA, self).__init__()

        self.num_bits = num_bits
        self.ternary = ternary
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(
            torch.Tensor([init_act_clip_val]), 
            requires_grad=clip_learnable
        )
        self.s = None
        self.boundaryRange = boundaryRange
        self.granularity = granularity

    def _compute_scaling_factor(self, real_weights):
        # ternary: s = E(|W|),  multi-bit: s = 2 * E(|W|)
        coeff = 1.0 if self.ternary else 2.0

        if self.granularity == "per_tensor":
            scaling_factor = coeff * torch.mean(abs(real_weights))
            if real_weights.ndim == 2:
                scaling_factor = scaling_factor.view(1, 1)
            elif real_weights.ndim == 3:
                scaling_factor = scaling_factor.view(1, 1, 1)
            else:
                raise ValueError(f"Unsupported weight shape: {real_weights.shape}")
        elif self.granularity == "per_channel":
            if real_weights.ndim == 2:
                scaling_factor = coeff * torch.mean(abs(real_weights), dim=1, keepdim=True)
            elif real_weights.ndim == 3:
                # (B, N, D) の D 方向で量子化スケールを共有
                scaling_factor = coeff * torch.mean(abs(real_weights), dim=2, keepdim=True)
            else:
                raise ValueError(f"Unsupported weight shape: {real_weights.shape}")
        elif self.granularity == "per_token":
            if real_weights.ndim == 2:
                scaling_factor = coeff * torch.mean(abs(real_weights), dim=1, keepdim=True)
            elif real_weights.ndim == 3:
                # 既存実装互換: token軸Nごとにスケール（batch共有）
                scaling_factor = coeff * torch.mean(
                    torch.mean(abs(real_weights), dim=-1, keepdim=True),
                    dim=0,
                    keepdim=True,
                )
            else:
                raise ValueError(f"Unsupported weight shape: {real_weights.shape}")
        else:
            raise ValueError(
                f"Unsupported granularity: {self.granularity}. "
                "Use one of ['per_tensor', 'per_channel', 'per_token']."
            )

        return scaling_factor
    
    
    def forward(self, weight):
        real_weights = weight

        # 統計的スケーリングファクター
        scaling_factor = self._compute_scaling_factor(real_weights)

        scaling_factor = scaling_factor.detach()
        self.s = scaling_factor.squeeze().detach()
        
        if self.ternary:
            # ── 三値量子化パス {-1, 0, +1} ──
            scaled_weights = real_weights / scaling_factor
            # Clip to [-1, 1]
            clipped = torch.clamp(scaled_weights, -1.0, 1.0)
            
            # CGA: 決定境界 ±0.5 の近傍が BR
            if self.training:
                # clipped の各要素について、最も近い決定境界 (±0.5) までの距離
                abs_val = torch.abs(clipped)
                dist_from_boundary = torch.abs(abs_val - 0.5)
                within_BR = (dist_from_boundary <= self.boundaryRange).float()
                # BR外は勾配を切る
                clipped = clipped.detach() * (1 - within_BR) + clipped * within_BR
            
            # Round to {-1, 0, +1}
            quan_weights_no_grad = scaling_factor * torch.round(clipped)
        else:
            # ── 既存マルチビット量子化パス ──
            scaled_weights = real_weights / scaling_factor
            cliped_weights = torch.clamp(
                scaled_weights, 
                min=(-self.clip_val/2), 
                max=(self.clip_val/2) - 1e-6
            )
            
            n = float(2 ** (self.num_bits - 1))
            b4_round = (cliped_weights) * n - 0.5
            
            # CGA: Algorithm 1の実装
            if self.training:
                # BR内かどうかの判定（ベクトル化実装）
                # b4_round が半整数 (i + 0.5) の近傍 ±boundaryRange にあるか判定
                # ⟺ (b4_round - 0.5) が最も近い整数から boundaryRange 以内にあるか
                shifted = b4_round - 0.5
                dist_from_int = torch.abs(shifted - torch.round(shifted))
                # within_BR = 1: BR内（更新する）, 0: BR外（フリーズ）
                within_BR = (dist_from_int <= self.boundaryRange).float()
                
                # BR外は勾配を切る（論文Algorithm 1のG_j = G_{j-1} * 1_{Wq ∈ BRx}）
                b4_round = b4_round.detach() * (1 - within_BR) + b4_round * within_BR
            
            quan_weights_no_grad = scaling_factor * ((torch.round(b4_round) + 0.5) / n)
        
        # STE
        quan_weights = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        return quan_weights


# ── StatsQUantizer_for_Attention ──────────────────────────────────────
class StatsQuantizer_4d(nn.Module):
    """
    4次元テンソル用のStatsQ (B, num_heads, N, in_features)
    論文 Sec.3.2: Attention機構での量子化
    最後の特徴次元でスケールを共有
    """
    def __init__(self, num_bits, clip_learnable=False):
        super(StatsQuantizer_4d, self).__init__()

        self.num_bits = num_bits
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(
            torch.Tensor([init_act_clip_val]), 
            requires_grad=clip_learnable
        )
        self.s = None
    
    def forward(self, weight):
        real_weights = weight

        # 4D: (B, num_heads, N, d) → 特徴次元dでスケール共有
        scaling_factor = 2 * torch.mean(
            torch.mean(
                torch.mean(abs(real_weights), dim=3, keepdim=True),
                dim=1, keepdim=True
            ),
            dim=0, keepdim=True
        )
        
        scaling_factor = scaling_factor.detach()
        self.s = scaling_factor.squeeze().cpu()
        
        scaled_weights = real_weights / scaling_factor
        cliped_weights = torch.clamp(
            scaled_weights, 
            min=(-self.clip_val/2), 
            max=(self.clip_val/2) - 1e-6
        )
        
        n = float(2 ** (self.num_bits - 1))
        quan_weights_no_grad = scaling_factor * (
            (torch.round((cliped_weights) * n - 0.5) + 0.5) / n
        )
        
        # STE
        quan_weights = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        return quan_weights

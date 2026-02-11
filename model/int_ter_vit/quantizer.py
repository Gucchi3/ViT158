import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 定数 ──────────────────────────────────────────────────────────────────
_EPS = 1e-5
_ACT_BITS = 8
_Qb = (1 << (_ACT_BITS - 1)) - 1  # 127  (int8 互換)


# ── 量子化_int8 ─────────────────────────────────────────────────────────
class Quantizer_int(nn.Module):
    
    def __init__(self, init_scale = 0.1):
        super().__init__()
        # 量子化スケールを学習可能パラメータとして定義
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        
    def forward(self, x, qmin, qmax):
        # self.scaleを正に固定
        scale = torch.clamp(self.scale, min=1e-8)
        # 量子化
        x_scaled = x / scale
        # 丸めこみ、クリップ
        x_q = torch.round(x_scaled).clamp(qmin, qmax)
        # STEのためにdetach
        x_q_d = x_scaled + (x_q - x_scaled).detach()        
        
        return x_q_d


# ── 量子化_float ────────────────────────────────────────────────────────
class Quantizer_float(nn.Module):
    
    def __init__(self, init_scale = 0.1):
        super().__init__()
        # 量子化スケールを学習可能パラメータとして定義
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        
    def forward(self, x, qmin, qmax):
        # self.scaleを正に固定
        scale = torch.clamp(self.scale, min=1e-8)
        # 量子化
        x_scaled = x / scale
        # 丸めこみ、クリップ
        x_q = torch.round(x_scaled).clamp(qmin, qmax)
        # STEのためにdetach
        x_q_d = x_scaled + (x_q - x_scaled).detach()        
        # 逆量子化
        x_result = x_q_d * scale
        
        return x_result

#! ######################################################################
#!               ───── 実装テスト中 LSQ初期化テスト ─────                   
#!#######################################################################
class Quantizer_float_leanable(nn.Module):
    
    def __init__(self, init_scale=0.1):
        super().__init__()
        # 量子化スケールを学習可能パラメータとして定義
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        # LSQ 初期化をしたかどうか
        self.register_buffer("initialized", torch.tensor(0))
    
    @torch.no_grad()
    def _init_from_activation_lsq(self, x, qmin, qmax):
        """
        LSQ 論文に基づく初期化:
        s_init = 2 * mean(|v|) / sqrt(Q_P)
        ここでは activations を unsigned 前提とし、Q_P = qmax とする。
        """
        # すでに初期化済みならパス
        if self.initialized.item() == 1:
            return
        # 入力をvとする。
        v = x
        # 絶対値の平均を取得 -> mean⟨|v|⟩
        mean_abs = v.abs().mean()
        # QPは量子化値がとる幅の最大値（if bit=8なら、QP=255or127）
        QP = float(qmax)
        # 論文内の初期化値生成用数式 -> s_init = 2⟨|v|⟩ / √Q_P 
        s_init = 2.0 * mean_abs / (QP ** 0.5)
        # 生成した初期値を格納
        with torch.no_grad():
            self.scale.copy_(s_init)
        # bufferに1を格納
        self.initialized.fill_(1)
    
    def forward(self, x, qmin, qmax):
        # 初回の forward で LSQ 初期化を 1 回だけ行う
        if self.initialized.item() == 0:
            self._init_from_activation_lsq(x, qmin, qmax)
        
        # self.scaleを正に固定
        scale = torch.clamp(self.scale, min=1e-8)
        # 量子化
        x_scaled = x / scale
        # 丸めこみ、クリップ
        x_q = torch.round(x_scaled).clamp(qmin, qmax)
        # STEのためにdetach
        x_q_d = x_scaled + (x_q - x_scaled).detach()
        # 逆量子化
        x_result = x_q_d * scale
        
        return x_result


# ── 逆量子化_float ───────────────────────────────────────────────────────
class DeQuantizer_float_init(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_int, x_scale, w_scale):
        # 逆量子化
        x_float = x_int * x_scale * w_scale
        return x_float


# ──　absmean 学習可能量子化 ───────────────────────────────────────────────
class Quantizer_with_absmean(nn.Module):
    def __init__(self, init_scale_a=0.1, init_scale_b=-0.1):
        super().__init__()
        # 学習可能な閾値
        self.scale_a = nn.Parameter(torch.tensor(float(init_scale_a)))
        self.scale_b = nn.Parameter(torch.tensor(float(init_scale_b)))

    def forward(self, w_input):
        scale = torch.mean(torch.abs(w_input))
        # absmean
        w_middle = w_input / ((scale) + _EPS)
        # 三値量子化: >scale_a -> 1, <scale_b -> -1, それ以外 -> 0
        w_output = torch.zeros_like(w_middle)
        w_output = torch.where(w_middle >= self.scale_a,  torch.ones_like(w_output), w_output)
        w_output = torch.where(w_middle <= self.scale_b, -torch.ones_like(w_output), w_output)

        return w_output, scale

# ──　absmean 学習可能量子化 ───────────────────────────────────────────────
class Quantizer_with_absmean(nn.Module):
    def __init__(self, init_scale_a=0.1, init_scale_b=-0.1):
        super().__init__()
        # 学習可能な閾値
        self.scale_a = nn.Parameter(torch.tensor(float(init_scale_a)))
        self.scale_b = nn.Parameter(torch.tensor(float(init_scale_b)))

    def forward(self, w_input):
        scale = torch.mean(torch.abs(w_input))
        # absmean
        w_middle = w_input / ((scale) + _EPS)
        # 三値量子化: >scale_a -> 1, <scale_b -> -1, それ以外 -> 0
        w_output = torch.zeros_like(w_middle)
        w_output = torch.where(w_middle >= self.scale_a,  torch.ones_like(w_output), w_output)
        w_output = torch.where(w_middle <= self.scale_b, -torch.ones_like(w_output), w_output)

        return w_output, scale


# ──　L1 norm ─────────────────────────────────────────────────────────────
class L1Norm(nn.Module):
    def __init__(self, dim=None, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, w_input):
        # L1ノルム（絶対値の総和）
        w_abs_total = w_input.abs().sum(dim=self.dim, keepdim=True)
        # ゼロ除算回避
        w_abs_total = torch.clamp(w_abs_total, min=self.eps)
        # L1 正規化
        w_output = w_input / w_abs_total
        return w_output


# ── ユーティリティ ─────────────────────────────────────────────────────────
def _round_clip(x, min_val, max_val):
    """RoundClip(x, a, b) = max(a, min(b, round(x)))"""
    return x.round().clamp_(min_val, max_val)


# ── 活性化量子化 (absmax, per-token, 8-bit) ────────────────────────────────
def activation_quant_float(x):
    """8-bit absmax 量子化 → 逆量子化 (float, STE 学習用).

    Returns
    -------
    x_q   : 量子化→逆量子化済みテンソル (float, 入力と同 shape)
    scale : Qb / γ  — per-token スケール係数
    """
    gamma = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=_EPS)
    scale = _Qb / gamma
    x_q = _round_clip(x * scale, -_Qb, _Qb) / scale
    return x_q, scale


# ── 重み量子化 (absmean, ternary {-1, 0, +1}) ──────────────────────────────
def weight_quant_float(w):
    """Absmean 3 値量子化 → 逆量子化 (float, STE 学習用).

    Returns
    -------
    w_q   : 量子化→逆量子化済みテンソル (float, 入力と同 shape)
    alpha : mean(|W|)  — スケーリング係数 (= β in paper)
    """
    alpha = w.abs().mean()
    w_q = _round_clip(w / (alpha + _EPS), -1, 1) * alpha
    return w_q, alpha


# ── int8 量子化 (CUDA kernel 用) ───────────────────────────────────────────
def activation_quant_int8(x):
    """8-bit absmax 量子化 (int8)."""
    gamma = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=_EPS)
    scale = _Qb / gamma
    x_q = _round_clip(x * scale, -_Qb, _Qb).to(torch.int8)
    return x_q, scale

# ──　重みint8量子化absmean ──────────────────────────────────────────────────
def weight_quant_int8(w):
    """Absmean 3 値量子化 (int8)."""
    alpha = w.abs().mean()
    w_q = _round_clip(w / (alpha + _EPS), -1, 1).to(torch.int8)
    return w_q, alpha


# ── 逆量子化関数 ────────────────────────────────────────────────────────
def dequant_float(mult_result, scale_x, scale_w):
    return mult_result.float() * (scale_w / scale_x)














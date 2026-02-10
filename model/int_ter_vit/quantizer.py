import torch
import torch.nn as nn
import torch.nn.functional as F

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
class Quantizer_float(nn.Module):
    
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
        ここでは activations を unsigned 前提とし、Q_P = qmax とする [web:29]。
        """
        if self.initialized.item() == 1:
            return
        
        v = x
        mean_abs = v.abs().mean()              # ⟨|v|⟩
        
        # unsigned activations を仮定: Q_P = 2^b - 1 ≒ qmax [web:29]
        # （qmin=0, qmax=2^b-1 で呼ぶ前提）
        QP = float(qmax)
        if QP <= 0:
            QP = 1.0  # ゼロ除算防止用のフォールバック
        
        s_init = 2.0 * mean_abs / (QP ** 0.5)  # s_init = 2⟨|v|⟩ / √Q_P [web:29]
        
        self.scale.data = s_init
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
class DeQuantizer_float(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_int, x_scale, w_scale):
        # 逆量子化
        x_float = x_int * x_scale * w_scale
        return x_float





# ── 定数 ────────────────────────────────────────────────────────────────
_EPS = 1e-5
_ACT_BITS = 8
_Qb = (1 << (_ACT_BITS - 1)) - 1  # 127  (int8 互換)

# ── ユーティリティ ──────────────────────────────────────────────────────
def _round_clip(x, min_val, max_val):
    """RoundClip(x, a, b) = max(a, min(b, round(x)))"""
    return x.round().clamp_(min_val, max_val)


# ── 活性化量子化 (absmax, per-token, 8-bit) ─────────────────────────────
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


# ── 重み量子化 (absmean, ternary {-1, 0, +1}) ──────────────────────────
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


# ── int8 量子化 (CUDA kernel 用) ────────────────────────────────────────
def activation_quant_int8(x):
    """8-bit absmax 量子化 (int8)."""
    gamma = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=_EPS)
    scale = _Qb / gamma
    x_q = _round_clip(x * scale, -_Qb, _Qb).to(torch.int8)
    return x_q, scale


def weight_quant_int8(w):
    """Absmean 3 値量子化 (int8)."""
    alpha = w.abs().mean()
    w_q = _round_clip(w / (alpha + _EPS), -1, 1).to(torch.int8)
    return w_q, alpha


# ── 逆量子化関数 ────────────────────────────────────────────────────────
def dequant_float(mult_result, scale_x, scale_w):
    return mult_result.float() * (scale_w / scale_x)














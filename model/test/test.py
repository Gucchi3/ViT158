import torch
from torch import nn
from torch.nn import Module, ModuleList

from pathlib import Path
import os, sys
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

try:
    from .test_linear import TerLinear, Q_Linear, Q_Conv2d, Affine, StatsQ_CGA_Linear, StatsQ_CGA_Conv2d, LSQ_Act_Quant
    from .quantizer import Quantizer, NEQ, ScaleConverter
    from .stats_quantizer import StatsQuantizer_CGA
except ImportError:
    parent_dir = Path(__file__).resolve().parents[2]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from .test_linear import TerLinear, Q_Linear, Q_Conv2d, Affine, StatsQ_CGA_Linear, StatsQ_CGA_Conv2d, LSQ_Act_Quant
    from .quantizer import Quantizer, NEQ, ScaleConverter
    from .stats_quantizer import StatsQuantizer_CGA
    


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Embeddings_CGA(Module):
    def __init__(self, image_size, patch_size, channels, dim, pool='cls', emb_dropout=0., bit=8, boundary_range=0.005, granularity="per_tensor"):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_cls_tokens = 1 if pool == 'cls' else 0
        # パッチ埋め込み
        self.proj = StatsQ_CGA_Conv2d(channels, dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width), bias=False, bit=bit, boundary_range=boundary_range, granularity=granularity)
        # クラストークン, 位置埋め込み
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if num_cls_tokens > 0 else None
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_cls_tokens, dim))
        # ドロップアウト
        self.dropout = nn.Dropout(emb_dropout)
        # 量子化器
        self.quantizer = Quantizer()
        # 再量子化器
        self.requantizer = LSQ_Act_Quant(num_bits=8)
        # 正規化
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 1. パッチ埋め込み 
        x = self.proj(x).flatten(2).transpose(1, 2)
        # 2. クラストークン結合 
        if self.cls_token is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = x.shape[0])
            x = torch.cat((cls_tokens, x), dim = 1)
        # 3. 位置埋め込みの量子化->逆量子化 
        pos_emb_q, _ = self.quantizer.to_bit_per_tensor(self.pos_embedding, bit=16, as_float=True)
        # 4. 加算 
        x = x + pos_emb_q
        # 5. 再量子化->逆量子化 
        x = self.requantizer(x, scale_x=None)
        # return
        x = self.dropout(x)
        return self.norm(x)

# classes
# ── Feed Forward ─────────────────────────────────────────────────────
class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.fc1 = Q_Linear(dim, hidden_dim, bias=False, use_norm=False, bit=8, as_float=True, unsigned=False)
        # self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc1 = StatsQ_CGA_Linear(dim, hidden_dim, bias=False, bit=8, boundary_range=0.005, granularity="per_tensor", ternary=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        # self.fc2 = Q_Linear(hidden_dim, dim, bias=False, use_norm=False, bit=8, as_float=True, unsigned=False)
        # self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.fc2 = StatsQ_CGA_Linear(hidden_dim, dim, bias=False, bit=8, boundary_range=0.005, granularity="per_tensor", ternary=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        # x, _ = self.fc1(x, return_scale=True)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        # x, _ = self.fc2(x, return_scale=True)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ── Attention ─────────────────────────────────────────────────────
class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = Q_Linear(
        #     dim,
        #     inner_dim * 3,
        #     bias=False,
        #     use_norm=False,
        #     bit=8,
        #     as_float=True,
        #     unsigned=False,
        # )
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv = StatsQ_CGA_Linear(dim, inner_dim * 3, bias=False, bit=8, boundary_range=0.005, granularity="per_tensor", ternary=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 正規化
        x = self.norm(x)
        # qkv
        # qkv, _ = self.to_qkv(x, return_scale=True)
        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # QとKの内積 * sqrt(dim)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # softmax
        attn = self.attend(dots)
        attn = self.dropout(attn)
        # attnをvに乗算
        out = torch.matmul(attn, v)
        # 次元結合
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ── Transformer ─────────────────────────────────────────────────────
class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


# ── ViT ─────────────────────────────────────────────────────
class TestViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., quant_bit = 8, quant_boundary_range = 0.005, quant_granularity = "per_tensor"):
        super().__init__()
        # image, patchのheight,widthを格納
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # もし画像サイズがパッチサイズで割り切れない場合にアサ―ト
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 総パッチ数計算
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # poolの内容確認アサ―ト
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # poolがclsなら、クラストークン付与
        num_cls_tokens = 1 if pool == 'cls' else 0
        # パッチ埋め込み（最初のConvにStatsQ-CGAを適用）
        self.to_patch_embedding = Embeddings_CGA(image_size=image_size, patch_size=patch_size, channels=channels, dim=dim, pool=pool, emb_dropout=emb_dropout, bit=quant_bit, boundary_range=quant_boundary_range, granularity=quant_granularity)
        # Transformer層定義
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # プーリングタイプをインスタンス化
        self.pool = pool
        # 恒等関数レイヤー
        self.to_latent = nn.Identity()
        # 分類ヘッド
        # 最終MLP headにStatsQ-CGAを適用
        self.mlp_head = StatsQ_CGA_Linear(dim, num_classes, bias=False, bit=quant_bit, boundary_range=quant_boundary_range, granularity=quant_granularity) if num_classes > 0 else None

    def forward(self, img):
        # バッチ次元数取得
        batch = img.shape[0]
        # パッチ埋め込み + CLSトークン + 位置埋め込み
        x = self.to_patch_embedding(img)
        # Transform層
        x = self.transformer(x)
        # headがないならrturn
        if self.mlp_head is None:
            return x
        # meanならシーケンス全体の平均、clsならCLSトークンの値を取得
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # 恒等関数
        x = self.to_latent(x)
        # headの出力をreturn
        return self.mlp_head(x)
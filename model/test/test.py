import torch
from torch import nn
from torch.nn import Module, ModuleList

from pathlib import Path
import os, sys
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

try:
    from .test_linear import TerLinear, Q_Linear
    from .quantizer import Quantizer
except ImportError:
    parent_dir = Path(__file__).resolve().parents[2]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from .test_linear import TerLinear, Q_Linear
    from .quantizer import Quantizer
    

# helpers
# ── pair ────────────────────────────────────────────────────────────
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

#
# ── Feed Forward ─────────────────────────────────────────────────────
class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            TerLinear(in_features = dim, out_features = hidden_dim, bias = True, use_layer_norm = True),
            #nn.GELU(),
            nn.ReLU6(),
            nn.Dropout(dropout),
            TerLinear(in_features = hidden_dim, out_features = dim, bias = True, use_layer_norm = True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


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
        # TerLinear ???
        self.to_qkv = TerLinear(in_features = dim, out_features = inner_dim * 3, bias = False, use_layer_norm = True)

        self.to_out = nn.Sequential(
            TerLinear(in_features = inner_dim, out_features = dim, bias = True, use_layer_norm = True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # qkv
        qkv = self.to_qkv(x).chunk(3, dim = -1)
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


# ── ViT ─────────────────────────────────────────────────────────────
class TestViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.quantizer = Quantizer()
        self.input_bit = 8
        self.embed_bit = 8
        # image, patchのheight,widthを格納
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # もし画像サイズがパッチサイズで割り切れない場合にアサ―ト
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 総パッチ数計算
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 各パッチの次元計算
        patch_dim = channels * patch_height * patch_width
        # poolの内容確認アサ―ト
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # poolがclsなら、クラストークン付与
        num_cls_tokens = 1 if pool == 'cls' else 0
        # パッチ埋め込みを行うシーケンシャルレイヤー
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim), # Dual PatchNorm (https://openreview.net/pdf?id=jgMqve6Qhw)
            Q_Linear(in_features = patch_dim, out_features = dim, bias = True, use_layer_norm = False),
            nn.LayerNorm(dim),
        )
        # クラストークン、位置埋め込み定義（学習可能）
        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))
        # Dropout層定義
        self.dropout = nn.Dropout(emb_dropout)
        # Transformer層定義
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # プーリングタイプをインスタンス化
        self.pool = pool
        # 恒等関数レイヤー (ここは任意の関数に変更可能)
        self.to_latent = nn.Identity()
        # 分類ヘッド
        self.mlp_head = Q_Linear(in_features = dim, out_features = num_classes, bias = True, use_layer_norm = False) if num_classes > 0 else None
    
    
    # ── スケール変換用関数 ─────────────────────────────────────────────────────
    def _rescale_int(self, x_int, from_scale, to_scale, bit):
        scale_ratio = to_scale / (from_scale + self.quantizer.EPS)
        int_max = (2**(bit - 1) - 1)
        return torch.round(x_int * scale_ratio).clamp(-int_max, int_max)
    
    
    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, img):
        """
        cls token、位置埋め込みは最初両方ともそれぞれのスケールで量子化される、
        その後、x_float = x_int / scale_cls をして逆量子化したのち x_int = x_float * scale_xをすることで、スケール変換する。
        """
        # 入力疑似量子化
        img_q, scale_img = self.quantizer.to_bit_per_tensor(img, bit=self.input_bit, as_float=True, unsigned=False)
        # STE
        img = img + (img_q - img).detach()
        # バッチ次元数取得
        batch = img.shape[0]
        # パッチ埋め込み
        x = self.to_patch_embedding(img)
        # パッチ埋め込みを整数化
        x_int, scale_x = self.quantizer.to_bit_per_tensor(x, bit=self.embed_bit, as_float=False, unsigned=False)
        # CLSトークン埋め込み
        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        if cls_tokens.numel() > 0:
            cls_int, scale_cls = self.quantizer.to_bit_per_tensor(cls_tokens, bit=self.embed_bit, as_float=False, unsigned=False)
            cls_int = self._rescale_int(cls_int, from_scale=scale_cls, to_scale=scale_x, bit=self.embed_bit)
            x_int = torch.cat((cls_int, x_int), dim = 1)
        # 連結後のシーケンス長取得
        seq = x_int.shape[1]
        # 位置埋め込みを整数化してスケールを合わせてから加算
        pos_int, scale_pos = self.quantizer.to_bit_per_tensor(self.pos_embedding[:seq], bit=self.embed_bit, as_float=False, unsigned=False)
        pos_int = self._rescale_int(pos_int, from_scale=scale_pos, to_scale=scale_x, bit=self.embed_bit)
        x_int = x_int + pos_int
        # 共通スケールで浮動小数へ復元
        x = x_int / (scale_x + self.quantizer.EPS)
        # Dropout
        x = self.dropout(x)
        # Transform層
        x = self.transformer(x)
        # headがないならrturn
        if self.mlp_head is None:
            return x
        # meanならシーケンス全体の平均、clsならCLSトークンの値を取得
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # 恒等関数 (任意の関数に変更可)
        x = self.to_latent(x)
        # headの出力をreturn
        return self.mlp_head(x)
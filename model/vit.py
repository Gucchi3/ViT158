import sys
from pathlib import Path

import math
from typing import Optional
from functools import partial
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接実行時と通常のインポート時の両方に対応
try:
    from .ter_linear_cuda import TerLinearCUDA as TerLinear
except ImportError: 
    #? ###############################################################################################################################################################################
    #? 上記のPathは相対パスであるため、ViT158フォルダ内から直接「Python model/vit.py」をたたいた時に、modelフォルダがパッケージ化されないため、相対パスではimportできない。（相対パスはパッケージ内専用）
    #? このため、sys.pathの0番ににViT158のパスを追加することで、ViT158フォルダ内から「model.ter_linear」でファイルにアクセスすることができる。  
    #? ###############################################################################################################################################################################
  # ViT158フォルダをsys.pathに追加
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from model.ter_linear_cuda import TerLinearCUDA as TerLinear 

#? __init__でimportを行うclassは必ず__all__に記述しないといけないというルールがある。
__all__ = ["VisionTransformer"]


# ── helpers ──────────────────────────────────────────────────────────────
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


# ── Patch Embedding ─────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


# ── DropPath ─────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype).add_(keep).floor_()
        return x / keep * mask


# ── MLP ──────────────────────────────────────────────────────────────────
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = TerLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = TerLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ── Attention ────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = TerLinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = TerLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ── Attention (Float32) ─────────────────────────────────────────────────
class AttentionFloat(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ── Transformer Block ───────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        #self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionFloat(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        #self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(x))
        x = x + self.drop_path2(self.mlp(x))
        return x


# ── Vision Transformer ──────────────────────────────────────────────────
class VisionTransformer(nn.Module):
    """
    ViT with 1.58-bit (ternary) quantized linear layers.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token & positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # class token

    def forward(self, x):
        return self.head(self.forward_features(x))


if __name__ == "__main__":
    import sys
    import json
    import os
    from pathlib import Path
    
    # 親ディレクトリをsys.pathに追加（ViT158フォルダから実行できるように）
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from torchinfo import summary
    from model import VisionTransformer

    # config.json読み込み
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # モデル初期化
    model = VisionTransformer(
        img_size=config["IMG_SIZE"],
        patch_size=config["PATCH_SIZE"],
        in_chans=config["IN_CHANS"],
        num_classes=config["NUM_CLASSES"],
        embed_dim=config["EMBED_DIM"],
        depth=config["DEPTH"],
        num_heads=config["NUM_HEADS"],
        mlp_ratio=config["MLP_RATIO"],
        qkv_bias=config["QKV_BIAS"],
    )

    # モデル情報表示
    print("=" * 80)
    print("Vision Transformer Model Information")
    print("=" * 80)
    summary(
        model,
        input_size=(1, config["IN_CHANS"], config["IMG_SIZE"], config["IMG_SIZE"]),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
        verbose=1,
    )

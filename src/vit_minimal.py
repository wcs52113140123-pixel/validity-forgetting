"""A ViT-B/16 forward pass in plain torch, loading timm-format weights directly.

Written because the target host has neither timm nor torchvision and no internet to
install them, and because the HF ViT key names differ between transformers 4.x and
5.x — a conversion validated against one version silently produces a randomly
initialised model on the other. Depending on nothing but torch removes that class of
failure entirely; correctness is checked against timm on the machine that has it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim=768, heads=12, mlp=3072):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn_qkv = nn.Linear(dim, dim * 3)
        self.attn_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, mlp)
        self.fc2 = nn.Linear(mlp, dim)
        self.heads = heads

    def forward(self, x):
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.attn_qkv(h).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, N, C)
        x = x + self.attn_proj(a)
        h = self.norm2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class ViTB16(nn.Module):
    def __init__(self, dim=768, depth=12, img=224, patch=16):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)[:, 0]


def load_timm(path, map_location="cpu"):
    """Load a timm vit_base_patch16_224 state dict into ViTB16.

    Asserts that every parameter is populated: a silent partial load is the failure
    this module exists to prevent.
    """
    sd = torch.load(path, map_location=map_location)
    m = ViTB16()
    new = {
        "patch.weight": sd["patch_embed.proj.weight"],
        "patch.bias": sd["patch_embed.proj.bias"],
        "cls": sd["cls_token"],
        "pos": sd["pos_embed"],
        "norm.weight": sd["norm.weight"],
        "norm.bias": sd["norm.bias"],
    }
    for i in range(12):
        p, q = f"blocks.{i}.", f"blocks.{i}."
        new[q + "attn_qkv.weight"] = sd[p + "attn.qkv.weight"]
        new[q + "attn_qkv.bias"] = sd[p + "attn.qkv.bias"]
        new[q + "attn_proj.weight"] = sd[p + "attn.proj.weight"]
        new[q + "attn_proj.bias"] = sd[p + "attn.proj.bias"]
        new[q + "fc1.weight"] = sd[p + "mlp.fc1.weight"]
        new[q + "fc1.bias"] = sd[p + "mlp.fc1.bias"]
        new[q + "fc2.weight"] = sd[p + "mlp.fc2.weight"]
        new[q + "fc2.bias"] = sd[p + "mlp.fc2.bias"]
        new[q + "norm1.weight"] = sd[p + "norm1.weight"]
        new[q + "norm1.bias"] = sd[p + "norm1.bias"]
        new[q + "norm2.weight"] = sd[p + "norm2.weight"]
        new[q + "norm2.bias"] = sd[p + "norm2.bias"]
    missing, unexpected = m.load_state_dict(new, strict=False)
    assert not missing, f"unpopulated parameters: {missing[:6]}"
    return m.eval()

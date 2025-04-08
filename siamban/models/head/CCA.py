import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super(SelfAttention2, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_x = norm_layer(dim)
        self.norm_z = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.a = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, z):
        '''
            Args:
                x (torch.Tensor): (B, L, C)
                q_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L, L), untied positional encoding
            Returns:
                torch.Tensor: (B, L, C)
        '''
        B, C, Hx, Wx = x.shape
        B, C, Hz, Wz = z.shape
        x = x.flatten(2).permute(0, 2, 1)  # B, N, C
        z = z.flatten(2).permute(0, 2, 1)  # B, N, C
        B, Nx, C = x.shape
        B, Nz, C = z.shape

        x = self.norm_x(x)
        z = self.norm_z(z)

        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(z).reshape(B, Nz, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        a = self.a(z).reshape(B, Nz, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn1 = k @ a.transpose(-2, -1)  # HxWx * HzWz
        attn1 = attn1 * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        v = attn1 @ v

        attn2 = a @ q.transpose(-2, -1)
        attn2 = attn2 * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        z = attn2 @ v

        # x = x.transpose(1, 2).reshape(B, Nx, C).permute(0, 2, 1).view(B, -1, Hx, Wx)
        # xaux = xaux.transpose(1, 2).reshape(B, Nx, C).permute(0, 2, 1).view(B, -1, Hx, Wx)
        z = z.transpose(1, 2).reshape(B, Nz, C)
        z = self.proj(z)
        z = self.proj_drop(z)

        return z.permute(0, 2, 1).view(B, -1, Hz, Wz)


class CCA2(nn.Module):
    def __init__(self, in_channels: list, out_channels=None):
        super(CCA2, self).__init__()
        self.in_channels = in_channels[0]
        if out_channels is None:
            self.out_channels = self.in_channels
        self.num_ins = len(in_channels)

        self.attn = nn.ModuleList()
        for i in range(self.num_ins):
            self.attn.append(SelfAttention2(self.in_channels))

    def forward(self, x1_fs, _, z_fs):
        # x1_out = []
        # x2_out = []
        z_out = []

        if not isinstance(x1_fs, list):
            x1_fs = [x1_fs]
            z_fs = [z_fs]

        for idx, (x1_f, z_f) in enumerate(zip(x1_fs, z_fs)):
            z_f = self.attn[idx](x1_f, z_f) + z_f
            z_out.append(z_f)
            # x1_out.append(x1_f)
            # x2_out.append(x2_f)

        # return x1_out, x2_out, z_out   # x1, x2 keep the same, z is updated
        return z_out   # x1, x2 keep the same, z is updated


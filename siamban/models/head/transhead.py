from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2D import VSSHead
from timm.models.layers import trunc_normal_

from .transhead_utils import *



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _ConvHead(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(_ConvHead, self).__init__()
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )


    def forward(self, search):
        search = self.conv_search(search)
        out = self.head(search)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv_mt = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, attn_pos, mt):
        '''
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        B, q_N, C = q.shape
        kv_N = kv.shape[1]
        mt_N = mt.shape[1]

        q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(B, kv_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        kv_mt = self.kv_mt(mt).reshape(B, mt_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_mt, v_mt = kv_mt[0], kv_mt[1]

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        attn_mt = q @ k_mt.transpose(-2, -1)
        attn_mt = attn_mt * self.scale
        if attn_pos is not None:
            if attn_pos.shape[-1] != attn.shape[-1]:  # expand the length of attn_pos with 0
                zero = torch.zeros([*attn_pos.shape[:-1], attn.shape[-1] - attn_pos.shape[-1]], device=attn.device, dtype=attn.dtype)
                attn_pos = torch.cat([attn_pos, zero], dim=-1)
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_mt = attn_mt.softmax(dim=-1)
        attn_mt = self.attn_drop(attn_mt)
        x = attn @ v + attn_mt @ v_mt
        x = x.transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Decoder, self).__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.norm1_mt = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)

        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv, attn_pos, mt):
        '''
            Args:
                mt:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        q = q + self.drop_path(self.attn(self.norm1_q(q), self.norm1_kv(kv), attn_pos, self.norm1_mt(mt)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))

        return q


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, attn_pos):
        '''
            Args:
                x (torch.Tensor): (B, L, C)
                q_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L, L), untied positional encoding
            Returns:
                torch.Tensor: (B, L, C)
        '''
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Encoder(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,  drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Encoder, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)

        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, cat, attn_pos):
        '''
            Args:
                cat (torch.Tensor): (B, Lx + Lz, C) x and z concatenated
            Returns:
                torch.Tensor: (B, L, C)
        '''
        cat = cat + self.drop_path(self.attn(self.norm1(cat), attn_pos))
        cat = cat + self.drop_path(self.mlp(self.norm2(cat)))

        return cat


class TransHead(nn.Module):
    def __init__(self, in_channels: tuple, enc_num, dec_num=1, cls_out_channels=2, num_heads=8, weighted=True,
                 norm_layer=nn.LayerNorm):
        super(TransHead, self).__init__()
        assert dec_num == 1
        for i in range(len(in_channels)):
            assert in_channels[i-1] == in_channels[i]
        dim = in_channels[0]

        self.enc = nn.ModuleList([Encoder(dim, num_heads) for _ in range(enc_num)])
        self.dec = Decoder(dim, num_heads)
        self.out_norm = norm_layer(dim)

        self.cls_mlp = _ConvHead(256, 256, cls_out_channels)
        self.reg_mlp = _ConvHead(256, 256, 4)

        self.loc_scale = nn.Parameter(torch.tensor([1., 2., 1.]))

        self.untied_z_pos_enc = Untied2DPositionalEncoder(dim, num_heads, 7, 7)
        self.untied_x_pos_enc = Untied2DPositionalEncoder(dim, num_heads, 31, 31)
        self.enc_rpe_index = generate_2d_concatenated_self_attention_relative_positional_encoding_index((7, 7), (31, 31))  # outputsize+2=31
        self.enc_rpe_bias_table = RelativePosition2DEncoder(num_heads, self.enc_rpe_index.max() + 1)

        self.untied_z_pos_dec = Untied2DPositionalEncoder(dim, num_heads, 7, 7, with_q=False)
        self.untied_x_pos_dec = Untied2DPositionalEncoder(dim, num_heads, 31, 31)
        self.dec_rpe_index = generate_2d_concatenated_cross_attention_relative_positional_encoding_index((7, 7), (31, 31))
        self.dec_rpe_bias_table = RelativePosition2DEncoder(num_heads, self.dec_rpe_index.max() + 1)

        self.weighted = weighted
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))


    def forward(self, z, x, mt=None):
        z = z.copy()
        B, _, H, W = x[-1].shape

        # encoder
        attn_pos_enc = None
        z_q_pos, z_k_pos = self.untied_z_pos_enc()
        x_q_pos, x_k_pos = self.untied_x_pos_enc()
        attn_pos_enc = (torch.cat((z_q_pos, x_q_pos), dim=1) @ torch.cat((z_k_pos, x_k_pos), dim=1).transpose(-2, -1)).unsqueeze(0)
        attn_pos_enc = attn_pos_enc + self.enc_rpe_bias_table(self.enc_rpe_index)  # (1, num_heads, Lx+Lz, Lx+Lz)

        for i in range(len(x)):
            z[i] = z[i].flatten(2).permute(0, 2, 1)
            x[i] = x[i].flatten(2).permute(0, 2, 1)
        for layer in self.enc:
            for i in range(len(x)):
                zxi = [z[i], x[i]]
                cat = layer(torch.cat(zxi, dim=1), attn_pos_enc)
                L = z[i].shape[1]
                z[i], x[i] = cat[:, :L, :], cat[:, L:, :]

        # decoder
        attn_pos_dec = None
        z_learned_pos_k = self.untied_z_pos_dec()
        x_learned_pos_q, x_learned_pos_k = self.untied_x_pos_dec()
        attn_pos_dec = x_learned_pos_q @ torch.cat((z_learned_pos_k, x_learned_pos_k), dim=1).transpose(-2, -1).unsqueeze(0)
        attn_pos_dec = attn_pos_dec + self.dec_rpe_bias_table(self.dec_rpe_index)  # (1, num_heads, Lx, Lx+Lz)

        for i in range(len(x)):
            zxi = [z[i], x[i]]
            x[i] = self.dec(zxi[-1], torch.cat(zxi, dim=1), attn_pos_dec, mt)

        cls = []
        bbox = []
        for i in range(len(x)):
            x[i] = self.out_norm(x[i])
            x[i] = x[i].permute(0, 2, 1).view(B, -1, H, W)

            # head
            cls.append(self.cls_mlp(x[i]))
            bbox.append(self.reg_mlp(x[i]))  # .sigmoid()
            bbox[i] = torch.exp(bbox[i] * self.loc_scale[i])

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            return weighted_avg(cls, cls_weight), weighted_avg(bbox, loc_weight)  # weighted Gate at the end
        else:
            return avg(cls), avg(bbox)  # (B, 2, H, W), (B, 4, H, W)



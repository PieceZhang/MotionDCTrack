from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class Learned2DPositionalEncoder(nn.Module):
    def __init__(self, dim, w, h):
        super(Learned2DPositionalEncoder, self).__init__()
        self.w_pos = nn.Parameter(torch.empty(w, dim))
        self.h_pos = nn.Parameter(torch.empty(h, dim))
        trunc_normal_(self.w_pos, std=0.02)
        trunc_normal_(self.h_pos, std=0.02)

    def forward(self):
        w = self.w_pos.shape[0]
        h = self.h_pos.shape[0]
        return (self.w_pos[None, :, :] + self.h_pos[:, None, :]).view(h * w, -1)


class Untied2DPositionalEncoder(nn.Module):
    def __init__(self, dim, num_heads, w, h, scale=None, with_q=True, with_k=True):
        super(Untied2DPositionalEncoder, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.pos = Learned2DPositionalEncoder(dim, w, h)
        self.norm = nn.LayerNorm(dim)
        self.pos_q_linear = None
        self.pos_k_linear = None
        if with_q:
            self.pos_q_linear = nn.Linear(dim, dim)
        if with_k:
            self.pos_k_linear = nn.Linear(dim, dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5

    def forward(self):
        pos = self.norm(self.pos())
        seq_len = pos.shape[0]
        if self.pos_q_linear is not None and self.pos_k_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_q, pos_k
        elif self.pos_q_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            return pos_q
        elif self.pos_k_linear is not None:
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_k
        else:
            raise RuntimeError


def generate_2d_concatenated_self_attention_relative_positional_encoding_index(z_shape, x_shape):
    '''
        z_shape: (z_h, z_w)
        x_shape: (x_h, x_w)
    '''
    z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    concatenated_2d_index_h = torch.cat((z_2d_index_h, x_2d_index_h))
    concatenated_2d_index_w = torch.cat((z_2d_index_w, x_2d_index_w))

    diff_h = concatenated_2d_index_h[:, None] - concatenated_2d_index_h[None, :]
    diff_w = concatenated_2d_index_w[:, None] - concatenated_2d_index_w[None, :]

    z_len = z_shape[0] * z_shape[1]
    x_len = x_shape[0] * x_shape[1]
    a = torch.empty((z_len + x_len), dtype=torch.int64)
    a[:z_len] = 0
    a[z_len:] = 1
    b=a[:, None].repeat(1, z_len + x_len)
    c=a[None, :].repeat(z_len + x_len, 1)

    diff = torch.stack((diff_h, diff_w, b, c), dim=-1)
    _, indices = torch.unique(diff.view((z_len + x_len) * (z_len + x_len), 4), return_inverse=True, dim=0)
    return indices.view((z_len + x_len), (z_len + x_len))


def generate_2d_concatenated_cross_attention_relative_positional_encoding_index(z_shape, x_shape):
    '''
        z_shape: (z_h, z_w)
        x_shape: (x_h, x_w)
    '''
    z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    concatenated_2d_index_h = torch.cat((z_2d_index_h, x_2d_index_h))
    concatenated_2d_index_w = torch.cat((z_2d_index_w, x_2d_index_w))

    diff_h = x_2d_index_h[:, None] - concatenated_2d_index_h[None, :]
    diff_w = x_2d_index_w[:, None] - concatenated_2d_index_w[None, :]

    z_len = z_shape[0] * z_shape[1]
    x_len = x_shape[0] * x_shape[1]

    a = torch.empty(z_len + x_len, dtype=torch.int64)
    a[: z_len] = 0
    a[z_len:] = 1
    c = a[None, :].repeat(x_len, 1)

    diff = torch.stack((diff_h, diff_w, c), dim=-1)
    _, indices = torch.unique(diff.view(x_len * (z_len + x_len), 3), return_inverse=True, dim=0)
    return indices.view(x_len, (z_len + x_len))


def generate_2d_cross_attention_relative_positional_encoding_index(z_shape, x_shape):
    '''
        z_shape: (z_h, z_w)
        x_shape: (x_h, x_w)
    '''
    z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    concatenated_2d_index_h = z_2d_index_h
    concatenated_2d_index_w = z_2d_index_w

    diff_h = x_2d_index_h[:, None] - concatenated_2d_index_h[None, :]
    diff_w = x_2d_index_w[:, None] - concatenated_2d_index_w[None, :]

    z_len = z_shape[0] * z_shape[1]
    x_len = x_shape[0] * x_shape[1]

    a = torch.empty(z_len, dtype=torch.int64)
    a[: z_len] = 0
    a[z_len:] = 1
    c = a[None, :].repeat(x_len, 1)

    diff = torch.stack((diff_h, diff_w, c), dim=-1)
    _, indices = torch.unique(diff.view(x_len * z_len, 3), return_inverse=True, dim=0)
    return indices.view(x_len, z_len)


class RelativePosition2DEncoder(nn.Module):
    def __init__(self, num_heads, embed_size):
        super(RelativePosition2DEncoder, self).__init__()
        self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads, embed_size)))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, attn_rpe_index):
        '''
            Args:
                attn_rpe_index (torch.Tensor): (*), any shape containing indices, max(attn_rpe_index) < embed_size
            Returns:
                torch.Tensor: (1, num_heads, *)
        '''
        return self.relative_position_bias_table[:, attn_rpe_index].unsqueeze(0)


def avg(lst):
    return sum(lst) / len(lst)


def weighted_avg(lst, weight):
    s = 0
    for i in range(len(weight)):
        s += lst[i] * weight[i]
    return s


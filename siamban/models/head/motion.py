import torch
import torch.nn as nn
from torch.nn import functional as F
from mamba_ssm.modules.mamba_simple import Mamba, MambaBi


class MambaMT2(nn.Module):
    """ Mamba Motion Tokenizer"""

    def __init__(self, n, s, L, outdim=256, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        out_features = outdim
        self.mamba_xy = MambaBi(d_model=2, d_state=16, d_conv=4, expand=32, dim_reduce=True)
        self.mamba_wh = MambaBi(d_model=2, d_state=16, d_conv=4, expand=32, dim_reduce=True)
        # self.act = act_layer()
        self.fc = nn.Linear(L*2, out_features)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x_xy = self.mamba_xy(torch.cat([x[:, :, :2].flip(-1), x[:, :, :2]], dim=1))
        x_wh = self.mamba_wh(torch.cat([x[:, :, -2:].flip(-1), x[:, :, -2:]], dim=1))
        x = torch.cat((x_xy, x_wh), dim=-1)
        # x = self.act(x)
        x = self.fc(x.transpose(-2, -1))
        return x
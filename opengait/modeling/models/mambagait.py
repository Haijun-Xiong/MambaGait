import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import BasicConv3d, PackSequenceWrapper, HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks
from mamba_ssm import Mamba
from torch.nn import functional as F
from einops import rearrange, repeat


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(1,2,2), in_chans=256, embed_dim=256, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = BasicConv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B C D Wh Ww
        _, _, _, h, w = x.size()
        x = rearrange(x, 'b c s h w -> b (s h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (s h w) c -> b c s h w', h=h, w=w)
        return x

def switch(x, shift=1):
    n, s, h, w, c = x.size()
    w_l = w // 4
    w_r = w_l * 3
    
    out = x.clone()

    shift_abs = abs(shift)
    if shift > 0:
        out[:, shift_abs:, :, :w_l] = x[:, :-shift_abs, :, :w_l]
        out[:, :shift_abs, :, :w_l] = x[:, -shift_abs:, :, :w_l]
        out[:, -shift_abs:, :, w_r:] = x[:, :shift_abs, :, w_r:]
        out[:, :-shift_abs, :, w_r:] = x[:, shift_abs:, :, w_r:]

    elif shift < 0:
        out[:, :-shift_abs, :, :w_l] = x[:, shift_abs:, :, :w_l]
        out[:, -shift_abs:, :, :w_l] = x[:, :shift_abs, :, :w_l]
        out[:, shift_abs:, :, w_r:] = x[:, :-shift_abs, :, w_r:]
        out[:, :shift_abs, :, w_r:] = x[:, -shift_abs:, :, w_r:]

    return out

class FourPathMamba(nn.Module):
    def __init__(self, dim):
        super(FourPathMamba, self).__init__()
        self.mamba = Mamba(d_model=dim, d_state=1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=dim*4, out_features=dim)
        self.drop_path = nn.Dropout(0.1)  # Drop path for stochastic depth

    def forward_part1(self, x):

        x = self.norm1(x)
        n, s, h, w, c = x.size()
        x1 = rearrange(x, 'b s h w c -> b (s h w) c')
        x2 = torch.flip(x1, dims=[1])
        x3 = rearrange(x, 'b s h w c -> b (s w h) c')
        x4 = torch.flip(x3, dims=[1])
        x1 = self.mamba(x1)
        x2 = self.mamba(x2)
        x2 = torch.flip(x2, dims=[1])
        x12 = x1 + x2
        x12 = rearrange(x12, 'b (s h w) c -> b s h w c', s=s, h=h, w=w)
        x3 = self.mamba(x3)
        x4 = self.mamba(x4)
        x4 = torch.flip(x4, dims=[1])
        x34 = x3 + x4
        x34 = rearrange(x34, 'b (s w h) c -> b s h w c', s=s, h=h, w=w)

        x = x12 + x34
        return x

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))
    
    def forward(self, x):
        T = 3
        b = x.size(0)
        x = rearrange(x, 'b (s T) h w c -> (b s) T h w c', T=T)

        x = x + self.drop_path(self.forward_part1(x))
        x = x + self.drop_path(self.forward_part2(x))
        x = rearrange(x, '(b s) T h w c -> b (s T) h w c', b=b)

        return x


class TWSM(nn.Module):
    def __init__(self, patch_size=(1, 2, 2), in_chans=256, embed_dim=256,):
        super(TWSM, self).__init__()

        self.pathch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)


        self.mamba1 = FourPathMamba(dim=embed_dim)
        self.mamba2 = FourPathMamba(dim=embed_dim)
        self.mamba3 = FourPathMamba(dim=embed_dim)
        self.mamba4 = FourPathMamba(dim=embed_dim)

        self.mamba5 = FourPathMamba(dim=embed_dim * 2)
        self.mamba6 = FourPathMamba(dim=embed_dim * 2)

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
        )


    def forward(self, x):
        x = self.pathch_embed(x)

        x = rearrange(x, 'b c s h w -> b s h w c')

        x = switch(x, shift=1)
        x = self.mamba1(x)
        x = switch(x, shift=-1)
        x = switch(x, shift=-1)
        x = self.mamba2(x)
        x = switch(x, shift=1)

        x = switch(x, shift=1)
        x = self.mamba3(x)
        x = switch(x, shift=-1)
        x = switch(x, shift=-1)
        x = self.mamba4(x)
        x = switch(x, shift=1)

        x = self.mlp(x)

        x = switch(x, shift=1)
        x = self.mamba5(x)
        x = switch(x, shift=-1)
        x = switch(x, shift=-1)
        x = self.mamba6(x)
        x = switch(x, shift=1)

        x = rearrange(x, 'b s h w c -> b c s h w')

        return x


class MambaGait(BaseModel):
    def __init__(self, cfgs, training):
        self.T_max_iter = cfgs['trainer_cfg']['T_max_iter']
        super(MambaGait, self).__init__(cfgs, training=training)

    def build_network(self, model_cfg):

        self.layer0 = nn.Sequential(
            BasicConv3d(in_channels=3, out_channels=32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.layer11 = nn.Sequential(
            BasicConv3d(in_channels=32, out_channels=64),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.layer12 = nn.Sequential(
            BasicConv3d(in_channels=32, out_channels=64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.pooling = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.layer23 = nn.Sequential(
            BasicConv3d(in_channels=64, out_channels=128),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            BasicConv3d(in_channels=128, out_channels=256),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.twsm = TWSM(patch_size=(1, 2, 2), in_chans=256, embed_dim=256)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.FCs = SeparateFCs(model_cfg['SeparateBNNecks']['parts_num'], in_channels=512, out_channels=256)
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])



    def forward(self, inputs):
        if self.training:
            adjust_learning_rate(self.optimizer, self.iteration, T_max_iter=self.T_max_iter)

        ipts, labs, _, _, seqL = inputs

        sils = ipts[0].unsqueeze(1)
        del ipts
        
        n, c, s, h, w = sils.size()

        K = s % 3
        if K:
            K = 3 - K
            sils = torch.cat([sils, sils[:, :, s-K:]], dim=2)


        x = self.layer0(sils)

        x = self.layer11(x) + self.layer12(x)

        x = self.pooling(x)

        x = self.layer23(x)

        x = self.twsm(x)


        x = self.TP(x, seqL, options={'dim': 2})[0]
        x = self.HPP(x)

        embed_1 = self.FCs(x)
        embed_2, logits = self.BNNecks(embed_1)

        return {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {'embeddings': embed_2}
        }


import math
def adjust_learning_rate(optimizer, iteration, iteration_per_epoch=1000, T_max_iter=10000, min_lr=1e-6):
    """Decay the learning rate based on schedule"""
    if iteration < T_max_iter:
        if iteration % iteration_per_epoch == 0:
            alpha = 0.5 * (1. + math.cos(math.pi * iteration / T_max_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['initial_lr'] * alpha, min_lr)
    elif iteration == T_max_iter:
        for param_group in optimizer.param_groups:
                param_group['lr'] = min_lr
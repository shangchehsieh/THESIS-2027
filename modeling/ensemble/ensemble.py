import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from modeling.ensemble.unet3d_parallel import UNet as UNetPara
from modeling.ensemble.fusion import ModalityAttnFusion, enumerate_subsets


class Ensemble(nn.Module):
    def __init__(self, in_channels, out_channels,
                 output='list', exchange=False, feature=False, modality_specific_norm=True, width_ratio=1., sharing=True,
                 fusion='mean', self_bias=0.0, **kwargs):
        super().__init__()

        # 初始化参数
        self.in_channels = in_channels
        self.output = output
        self.feature = feature
        self.modality_specific_norm = modality_specific_norm
        self.width_ratio = width_ratio
        self.sharing = sharing
        self.fusion = fusion
        self.module = UNetPara(1, out_channels, num_modalities=in_channels, parallel=True,
                            exchange=exchange, feature=feature, width_multiplier=width_ratio)

        if fusion == 'mattn':
            self.attn = ModalityAttnFusion(self.module.channels[0], in_channels, self_bias=self_bias)
        elif fusion != 'mean':
            raise ValueError(f'unknown fusion {fusion!r}')


    def forward(self, x, channel=[], weights=None):
        x = [x[:, i:i + 1] for i in range(self.in_channels)]
        out, df, feat = self.module(x)

        if self.fusion == 'mattn':
            terms = self.attn.scores_and_terms(self.module.outc.conv.module.weight, *feat)
            fuse = lambda subset: checkpoint(self.attn.fuse, out, *terms, subset, use_reentrant=False)
        else:
            fuse = lambda subset: torch.mean(torch.stack([out[i] for i in subset], dim=0), dim=0)

        if self.training:
            return [fuse(subset) for subset in enumerate_subsets(self.in_channels)], df

        preserved = list(range(self.in_channels))
        for c in channel:
            preserved.remove(c)

        return fuse(tuple(preserved))

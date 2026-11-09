import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from modeling.ensemble.unet3d_parallel import UNet as UNetPara
from modeling.ensemble.fusion import ModalityAttnFusion, enumerate_subsets


class Ensemble(nn.Module):
    def __init__(self, in_channels, out_channels,
                 output='list', exchange=False, feature=False, modality_specific_norm=True, width_ratio=1., sharing=True,
                 fusion='mean', self_bias=0.0, qk_dw_kernel=0, **kwargs):
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
            self.attn = ModalityAttnFusion(self.module.channels[0], in_channels, self_bias=self_bias,
                                           qk_dw_kernel=qk_dw_kernel)
        elif fusion == 'gate':
            # Region-aware gating (2026-09-21, stopped at ep 180), restored for inference only so its
            # checkpoint can be evaluated. Training it needs the gate-weighted EnumerationLoss, which
            # lives in git stash@{0}, not in this branch.
            self.gate = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        elif fusion != 'mean':
            raise ValueError(f'unknown fusion {fusion!r}')


    def forward(self, x, channel=[], weights=None):
        x = [x[:, i:i + 1] for i in range(self.in_channels)]
        out, df, feat = self.module(x)

        if self.fusion == 'mattn':
            terms = self.attn.scores_and_terms(self.module.outc.conv.module.weight, *feat)
            fuse = lambda subset: checkpoint(self.attn.fuse, out, *terms, subset, use_reentrant=False)
        elif self.fusion == 'gate':
            if self.training:
                raise NotImplementedError("fusion='gate' is inference-only in this branch")
            def fuse(subset):  # softmax over modalities of a shared 3^3 conv on each stream's logits
                w = torch.softmax(torch.stack([self.gate(out[i]) for i in subset], dim=0), dim=0)
                return torch.sum(w * torch.stack([out[i] for i in subset], dim=0), dim=0)
        else:
            fuse = lambda subset: torch.mean(torch.stack([out[i] for i in subset], dim=0), dim=0)

        if self.training:
            return [fuse(subset) for subset in enumerate_subsets(self.in_channels)], df

        preserved = list(range(self.in_channels))
        for c in channel:
            preserved.remove(c)

        return fuse(tuple(preserved))

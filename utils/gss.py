"""Category Aware Group Self-Support distillation (GSS, Qiu et al., ICCV 2023), adapted to HD.

GSS has no teacher network: the modalities vote on each voxel, and the vote is the soft
target for every single-modality prediction. The paper works on four exclusive softmax
classes; HD has three overlapping sigmoid channels, so each channel votes on its own and
the category normalisation (only meaningful across exclusive classes) is dropped.

The vote runs on logits, with the thresholds mapped through logit(). Comparisons, max and
min then give the same decision as on probabilities, and trained logits (several hundred
in magnitude) keep their value: on probabilities they saturate, and the teacher softened
by tau would be far softer than the student it is teaching.
"""
import math

import torch
import torch.nn.functional as F

# Modality indices, channel order of the *_vol.npy volumes: FLAIR, T1c, T1, T2.
# Output channels WT, TC, ET. T1c leading NCR/NET and ET, FLAIR + T2 leading ED is the
# paper's grouping; FLAIR + T2 leading WT is our mapping of it onto the WT channel.
LEADERS = ((0, 3), (1,), (1,))


def vote(x, leaders, t_l, t_m, rho):
    """Soft target for one channel: Algorithm 1 (one leader) or 2 (two). x: [B, M, D, H, W]."""
    members = [m for m in range(x.shape[1]) if m not in leaders]
    lead, memb = x[:, list(leaders)], x[:, members]
    hi, lo = lead.amax(dim=1), lead.amin(dim=1)
    out = torch.where((memb > t_m).all(dim=1), memb.mean(dim=1), torch.minimum(lo, memb.amin(dim=1)))
    if len(leaders) == 1:
        return torch.where(hi > t_l, hi, out)
    out = torch.where(hi > t_l, hi - rho * (hi - lo), out)
    return torch.where(lo > t_l, hi, out)


def gss_kd(logits, tau=8., t_l=0.65, t_m=0.75, rho=0.3, mask_rate=0.2):
    """Mean binary KL(vote || student) over the voxels the random mask keeps.
    logits: the M single-modality outputs, each [B, 3, D, H, W]."""
    s = torch.stack(logits, dim=1)
    with torch.no_grad():
        l_t, l_m = math.log(t_l / (1 - t_l)), math.log(t_m / (1 - t_m))
        t = torch.stack([vote(s[:, :, c], LEADERS[c], l_t, l_m, rho) for c in range(s.shape[2])], dim=1) / tau
        p = t.sigmoid()
        entropy = F.binary_cross_entropy_with_logits(t, p, reduction='none')
        keep = (torch.rand_like(t[:, :1]) >= mask_rate).to(t.dtype)
    bce = F.binary_cross_entropy_with_logits(s / tau, p.unsqueeze(1).expand_as(s), reduction='none')
    kl = bce.mean(dim=1) - entropy
    return (kl * keep).sum() / (keep.sum() * kl.shape[1]).clamp_min(1)

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def enumerate_subsets(num_modalities):
    """All non-empty modality subsets, largest first (same order as the paper's loss)."""
    return [s for l in reversed(range(1, num_modalities + 1))
            for s in itertools.combinations(range(num_modalities), l)]


class ModalityAttnFusion(nn.Module):
    """Cross-modal self-attention over the available modalities' features (mmFormer-style
    inter-modal attention, per voxel).

    Token i is f_i + emb_i (a learnable modality-type embedding, as in mmFormer's
    inter-modal transformer). Each modality i (query) attends over every available
    modality j (keys/values) at the same voxel and adds the attended value back
    through a zero-initialised output projection, so at initialisation the module is
    exactly the identity and the downstream uniform mean reproduces the baseline.
    Missing modalities are simply absent from `subset`, which is how they get masked
    out of the attention.

    `self_bias` is a FIXED (not learned) constant added to the diagonal of the scores,
    so every modality attends to itself by default. Trained v1 sent every query to
    FLAIR, including T1c (T1c's own row: FLAIR 2.65 vs itself 0.81), which diluted
    T1c's ET signal; a learnable version of this bias was driven negative by training,
    which is why it is fixed here. 0 reproduces v1 exactly. It is a buffer so the
    checkpoint records the value it was trained with.
    """

    def __init__(self, channels, num_modalities, qk_channels=None, self_bias=0.0):
        super().__init__()
        self.register_buffer('self_bias', torch.tensor(float(self_bias)))
        C = channels
        Cqk = qk_channels or channels
        self.emb = nn.Parameter(torch.empty(num_modalities, C))
        nn.init.normal_(self.emb, std=0.02)
        self.q = nn.Conv3d(C, Cqk, kernel_size=1)
        self.k = nn.Conv3d(C, Cqk, kernel_size=1)
        self.v = nn.Conv3d(C, C, kernel_size=1)
        self.o = nn.Conv3d(C, C, kernel_size=1)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
        self.scale = Cqk ** -0.5

    def forward(self, feats, subset):
        """Reference path: refined features for the modalities in `subset`."""
        C = feats[0].shape[1]
        toks = [f + self.emb[m].view(1, C, 1, 1, 1) for f, m in zip(feats, subset)]
        q = torch.stack([self.q(t) for t in toks], dim=1)
        k = torch.stack([self.k(t) for t in toks], dim=1)
        v = torch.stack([self.v(t) for t in toks], dim=1)
        attn = torch.einsum('bicdhw,bjcdhw->bijdhw', q, k) * self.scale
        S = len(feats)
        attn = (attn + self.self_bias * torch.eye(S, device=attn.device).view(1, S, S, 1, 1, 1)).softmax(dim=2)
        a = torch.einsum('bijdhw,bjcdhw->bicdhw', attn, v)
        return [f + self.o(a[:, i]) for i, f in enumerate(feats)]

    # ------------------------------------------------------------------ fast path
    # Because `o`, the 1x1 head and the mean are all linear, the head applied to
    # the refined feature i equals  head(f_i) + sum_j attn_ij * u_j + c  with
    # u_j = W_head W_o W_v tok_j (3 channels) and c = W_head (W_o b_v + b_o).
    # So per subset only a masked softmax over precomputed M x M scores and a
    # 3-channel contraction remain. The modality embedding enters every 1x1 conv
    # as a per-modality bias, so the convs run on the raw features directly.

    @staticmethod
    def _proj(x, w, off):
        return F.conv3d(x, w.view(*w.shape, 1, 1, 1)) + off.view(1, -1, 1, 1, 1)

    def _score_pair(self, fi, fj, w_q, w_k, off_q, off_k):
        # Per-voxel dot product as an elementwise multiply-sum (einsum lowers this
        # to millions of tiny GEMMs, which is far slower on this GPU).
        return (self._proj(fi, w_q, off_q) * self._proj(fj, w_k, off_k)).sum(dim=1) * self.scale

    def scores_and_terms(self, head_weight, *feats):
        """Shared work for all subsets: scores [B,M,M,...], u [B,M,3,...], c."""
        M = len(feats)
        w_q, w_k = self.q.weight.flatten(1), self.k.weight.flatten(1)
        w_head, w_o, w_v = head_weight.flatten(1), self.o.weight.flatten(1), self.v.weight.flatten(1)
        w_u = w_head @ w_o @ w_v
        off_q = self.emb @ w_q.t() + self.q.bias
        off_k = self.emb @ w_k.t() + self.k.bias
        off_u = self.emb @ w_u.t()
        # Recompute each pair's q_i/k_j on backward: storing q and k for all
        # modalities is 8.6 GB at batch 8, the per-pair transient is ~5 GB.
        pair = (lambda i, j: checkpoint(self._score_pair, feats[i], feats[j], w_q, w_k, off_q[i], off_k[j],
                                        use_reentrant=False)) if torch.is_grad_enabled() else \
               (lambda i, j: self._score_pair(feats[i], feats[j], w_q, w_k, off_q[i], off_k[j]))
        scores = torch.stack([torch.stack([pair(i, j) for j in range(M)], dim=1) for i in range(M)], dim=1)
        u = torch.stack([self._proj(f, w_u, off_u[m]) for m, f in enumerate(feats)], dim=1)
        c = (w_head @ (w_o @ self.v.bias + self.o.bias)).view(1, -1, 1, 1, 1)
        return scores, u, c

    @torch.no_grad()
    def diagnostics(self, head_weight, feats=None):
        """Numbers for spotting the two pathologies this module is prone to: an
        output projection that grows until it swamps the direct path, and attention
        that collapses onto a single modality. `gain` is the attention path's gain
        relative to the direct path (1.0 means comparable); `self_w` and `top_w` are
        the mean diagonal and the mean largest key weight (1/M means no preference).
        """
        w_head, w_o, w_v = head_weight.flatten(1), self.o.weight.flatten(1), self.v.weight.flatten(1)
        d = {"o_norm": self.o.weight.norm().item(),
             "gain": ((w_head @ w_o @ w_v).norm() / w_head.norm()).item()}
        if feats is not None:
            sc = self.scores_and_terms(head_weight, *feats)[0]
            M = sc.shape[1]
            sc = sc + self.self_bias * torch.eye(M, device=sc.device).view(1, M, M, 1, 1, 1)
            a = sc.softmax(dim=2).mean(dim=(0, 3, 4, 5))
            d["self_w"] = a.diagonal().mean().item()
            d["top_w"] = a.mean(dim=0).max().item()
            d["attn"] = a
        return d

    def fuse(self, head_out, scores, u, c, subset):
        """Fused head output for one subset (mean over its modalities)."""
        idx = list(subset)
        S, M = len(idx), scores.shape[1]
        pairs = torch.tensor([i * M + j for i in idx for j in idx], device=scores.device)
        attn = scores.flatten(1, 2).index_select(1, pairs).view(-1, S, S, *scores.shape[3:])
        attn = (attn + self.self_bias * torch.eye(S, device=attn.device).view(1, S, S, 1, 1, 1)).softmax(dim=2)
        corr = 0
        for jj, j in enumerate(idx):
            corr = corr + attn[:, :, jj].unsqueeze(2) * u[:, j].unsqueeze(1)
        h = torch.stack([head_out[i] for i in idx], dim=1)
        return (h + corr).mean(dim=1) + c

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # v1 checkpoints predate the buffer; they were trained with no bias.
        state_dict.setdefault(prefix + 'self_bias', torch.tensor(0.0))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

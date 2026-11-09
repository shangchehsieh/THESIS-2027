"""Per-case Dice of every main-line model on the test split, all 15 modality combos.

    python tools/per_case_eval.py [--only TAG ...] [--limit N]

For each model, case, combo and region (WT, TC, ET) it stores the sums that any
Dice aggregation can be rebuilt from, in results/per_case/<tag>.npz:
    soft_i = sum(g * p), soft_s = sum(g + p)   p = sigmoid(logit)   (HD's eval formula)
    bin_i,  bin_s                              p = logit > 0 (sigmoid > 0.5)
Per-case Dice = (2 i + 1) / (s + 1), the smooth=1 formula of utils/metrics.py.

The 4 modality streams do not depend on each other, so each case runs the streams
once and fuses the 15 subsets from that one pass. Before each model, the first test
case is checked against the real eval path (dropped channels zeroed, model(x, channel=drop))
for all 15 combos; the run stops if they differ.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dataloaders.datasets.brats_npy import BratsNpy, make_split_lists  # noqa: E402
from modeling.ensemble.ensemble import Ensemble  # noqa: E402
from modeling.ensemble.fusion import ModalityAttnFusion  # noqa: E402,F401
from mypath import Path as DataPath  # noqa: E402

RUNS = REPO / "results" / "brats3d-acn"
OUT = REPO / "results" / "per_case"
P = "gmd-enum-2-adam-acn-split2"
MODELS = [  # (tag = log name, checkname, model kwargs); self_bias is a checkpoint buffer
    ("0_HD_original", P, dict(fusion="mean")),
    ("1_baseline", f"{P}-lrfix", dict(fusion="mean")),
    ("2_gating", f"{P}-lrfix-gate", dict(fusion="gate")),
    ("3_attn", f"{P}-lrfix-mattn", dict(fusion="mattn", qk_dw_kernel=0)),
    ("4_selfbias", f"{P}-lrfix-v1-selfbias2", dict(fusion="mattn", qk_dw_kernel=0)),
    ("5_dwconv", f"{P}-lrfix-v1-selfbias2-dwqk3", dict(fusion="mattn", qk_dw_kernel=3)),
    ("6_gss", f"{P}-lrfix-v1-selfbias2-dwqk3-gss", dict(fusion="mattn", qk_dw_kernel=3)),
]
MODS = ["Fl", "T1c", "T1", "T2"]
# same order as eval_test.py: drop 0 channels first, then 1, 2, 3 dropped
import itertools  # noqa: E402
DROPS = [d for l in range(4) for d in itertools.combinations(range(4), l)]
COMBOS = [",".join(m for i, m in enumerate(MODS) if i not in d) for d in DROPS]


def all_subsets(m, x):
    """Fused logits for every combo from one pass of the 4 streams."""
    xs = [x[:, i:i + 1] for i in range(4)]
    out, _, feat = m.module(xs)
    if m.fusion == "mattn":
        terms = m.attn.scores_and_terms(m.module.outc.conv.module.weight, *feat)
        fuse = lambda s: m.attn.fuse(out, *terms, s)
    elif m.fusion == "gate":
        def fuse(s):
            w = torch.softmax(torch.stack([m.gate(out[i]) for i in s]), dim=0)
            return torch.sum(w * torch.stack([out[i] for i in s]), dim=0)
    else:
        fuse = lambda s: torch.mean(torch.stack([out[i] for i in s]), dim=0)
    return [fuse(tuple(i for i in range(4) if i not in d)) for d in DROPS]


def sums(logit, g):
    p = torch.sigmoid(logit.float())
    b = (logit > 0).float()
    dims = (0, 2, 3, 4)
    return [t.sum(dims).cpu().numpy() for t in (g * p, g + p, g * b, g + b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    root = DataPath.getPath("brats3d-acn")
    _, _, test_ids = make_split_lists(root)
    if args.limit > 0:
        test_ids = test_ids[:args.limit]
    ds = BratsNpy(root, test_ids, crop_size=(128, 160, 128), train=False)
    dev = "cuda"

    for tag, ck, kw in MODELS:
        if args.only and tag not in args.only:
            continue
        state = torch.load(RUNS / ck / "model_best.pth", map_location="cpu", weights_only=False)
        m = Ensemble(4, 3, width_ratio=0.5, **kw)
        m.load_state_dict(state["state_dict"])
        m = m.to(dev).eval()
        res = np.zeros((4, len(test_ids), len(DROPS), 3))  # soft_i, soft_s, bin_i, bin_s
        with torch.no_grad():
            for n in range(len(test_ids)):
                s = ds[n]
                x, g = s["image"][None].to(dev), s["label"][None].float().to(dev)
                outs = all_subsets(m, x)
                if n == 0:  # equality with the real eval path
                    for d, o in zip(DROPS, outs):
                        xz = x.clone()
                        xz[:, list(d)] = 0
                        ref = m(xz, channel=list(d))
                        err = (ref - o).abs().max().item()
                        if err > 1e-3:
                            sys.exit(f"{tag}: one-pass differs from eval path on drop {d}: {err}")
                for c, o in enumerate(outs):
                    res[:, n, c] = sums(o, g)
        np.savez(OUT / f"{tag}.npz", soft_i=res[0], soft_s=res[1], bin_i=res[2], bin_s=res[3],
                 cases=np.array(test_ids), combos=np.array(COMBOS), epoch=state["epoch"])
        d = (2 * res[0] + 1) / (res[1] + 1)
        print(f"{tag:14s} ep {state['epoch']:3d}  per-case soft Dice (15-combo mean) "
              f"WT {100*d[..., 0].mean():.2f} TC {100*d[..., 1].mean():.2f} ET {100*d[..., 2].mean():.2f}",
              flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

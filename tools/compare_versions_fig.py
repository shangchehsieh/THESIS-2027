"""One test case, one axial slice: the 4 MRI modalities, the ground truth, and the
prediction of every main-line version (all 4 modalities as input).

    python tools/compare_versions_fig.py [--case ID] [--seed N] [--out PNG] [--tiny]

Colours (BraTS convention): green = edema, red = necrotic / non-enhancing core,
blue = enhancing tumor. The model outputs overlapping WT/TC/ET sigmoid channels;
they are turned back into the three sub-regions as ET, TC minus ET, WT minus TC.
The Dice under each panel is binary (sigmoid > 0.5) over this one case's crop,
not the batch-pooled soft Dice of eval_test.
--tiny runs on a 32^3 crop around the tumour: a cheap pipeline check only.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dataloaders.datasets.brats_npy import BratsNpy, make_split_lists  # noqa: E402
from modeling.ensemble.ensemble import Ensemble  # noqa: E402
from mypath import Path as DataPath  # noqa: E402

RUNS = REPO / "results" / "brats3d-acn"
VERSIONS = [  # (panel title, checkname, model kwargs); numbering = the results slide
    ("#1 baseline (HD†)", "gmd-enum-2-adam-acn-split2-lrfix", dict(fusion="mean")),
    ("#2 + region-aware gating\n(stopped at ep 180)", "gmd-enum-2-adam-acn-split2-lrfix-gate", dict(fusion="gate")),
    ("#3 + cross-modal attn", "gmd-enum-2-adam-acn-split2-lrfix-mattn", dict(fusion="mattn")),
    ("#4 = #3 + self-bias", "gmd-enum-2-adam-acn-split2-lrfix-v1-selfbias2", dict(fusion="mattn")),
    ("#5 = #4 + DW conv (Q, K)", "gmd-enum-2-adam-acn-split2-lrfix-v1-selfbias2-dwqk3", dict(fusion="mattn", qk_dw_kernel=3)),
    ("#6 = #5 + GSS", "gmd-enum-2-adam-acn-split2-lrfix-v1-selfbias2-dwqk3-gss", dict(fusion="mattn", qk_dw_kernel=3)),
]
MODS = ["FLAIR", "T1c", "T1", "T2"]
COLOURS = {"ED": (0.10, 0.80, 0.20), "NCR": (0.90, 0.10, 0.10), "ET": (0.15, 0.45, 1.00)}


def regions(wt, tc, et):
    """Overlapping WT/TC/ET masks -> the three disjoint BraTS sub-regions."""
    return {"ED": wt & ~tc, "NCR": tc & ~et, "ET": et}


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * (p & g).sum() / s


def norm(img):
    lo, hi = np.percentile(img[img != 0], [1, 99]) if (img != 0).any() else (0, 1)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def overlay(ax, bg, masks, alpha=0.55):
    ax.imshow(np.rot90(bg), cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros((*bg.shape, 4))
    for name, m in masks.items():
        rgba[m] = (*COLOURS[name], alpha)
    ax.imshow(np.rot90(rgba))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--seed", type=int, default=1104)
    ap.add_argument("--out", default=str(REPO / "figures" / "version_comparison.png"))
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    root = DataPath.getPath("brats3d-acn")
    _, _, test_ids = make_split_lists(root)
    if args.case:
        case = args.case
    else:  # random test case that shows all three sub-regions clearly
        rng = np.random.default_rng(args.seed)
        for case in rng.permutation(test_ids):
            seg = np.load(Path(root) / "seg" / f"{case}_seg.npy")
            if all((seg == k).sum() > 2000 for k in (1, 2, 3)):
                break
    sample = BratsNpy(root, [case], crop_size=(128, 160, 128), train=False)[0]
    x, y = sample["image"], sample["label"].bool()          # [4,H,W,D], [3,H,W,D]
    gt = regions(*y)

    # axial slice (last axis) with the most tumour among slices that show all three regions
    area = sum(m.sum(dim=(0, 1)) for m in gt.values())
    allthree = torch.stack([m.any(0).any(0) for m in gt.values()]).all(0)
    z = int(torch.where(allthree, area, torch.zeros_like(area)).argmax())

    if args.tiny:  # 32^3 crop centred on the tumour in this slice
        hw = torch.nonzero(y[0][:, :, z]).float().mean(0).long()
        h0 = int(hw[0].clamp(16, x.shape[1] - 16)) - 16
        w0 = int(hw[1].clamp(16, x.shape[2] - 16)) - 16
        z0 = max(0, min(z - 16, x.shape[3] - 32))
        x, y = x[:, h0:h0 + 32, w0:w0 + 32, z0:z0 + 32], y[:, h0:h0 + 32, w0:w0 + 32, z0:z0 + 32]
        gt, z = regions(*y), z - z0

    dev = "cuda" if torch.cuda.is_available() and not args.tiny else "cpu"
    preds, titles = [], []
    for title, ck, kw in VERSIONS:
        path = RUNS / ck / "model_best.pth"
        state = torch.load(path, map_location="cpu", weights_only=False)
        m = Ensemble(4, 3, width_ratio=0.5, **kw)
        m.load_state_dict(state["state_dict"])
        m = m.to(dev).eval()
        with torch.no_grad():
            p = m(x[None].to(dev), channel=[])[0].cpu() > 0
        preds.append(p)
        t = title
        d = [100 * dice(p[i], y[i]) for i in range(3)]
        titles.append((t, d))
        print(f"{t.replace(chr(10), ' '):44s} case Dice WT {d[0]:.1f}  TC {d[1]:.1f}  ET {d[2]:.1f}")
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    # 2 x 6 (landscape): modalities + GT + legend / #1-#6
    imgs = [norm(x[i, :, :, z].numpy()) for i in range(4)]
    fig, axes = plt.subplots(2, 6, figsize=(17.5, 8.8), gridspec_kw=dict(hspace=0.3, wspace=0.06))
    for i in range(4):
        axes[0, i].imshow(np.rot90(imgs[i]), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(MODS[i], fontsize=14)
    overlay(axes[0, 4], imgs[0], {k: v[:, :, z].numpy() for k, v in gt.items()})
    axes[0, 4].set_title("Ground truth", fontsize=14, fontweight="bold")
    for ax, p, (t, d) in zip(axes[1], preds, titles):
        overlay(ax, imgs[0], {k: v[:, :, z].numpy() for k, v in regions(*p).items()})
        ax.set_title(t, fontsize=12.5)
        ax.text(0.5, -0.04, f"WT {d[0]:.1f}   TC {d[1]:.1f}   ET {d[2]:.1f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=12)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    axes[0, 5].axis("off")
    axes[0, 5].legend(handles=[Patch(color=COLOURS["ED"], label="Edema"),
                               Patch(color=COLOURS["NCR"], label="Necrotic /\nnon-enhancing core"),
                               Patch(color=COLOURS["ET"], label="Enhancing tumor")],
                      loc="center", fontsize=13, frameon=False, labelspacing=1.2)
    fig.suptitle(f"{case}  ·  axial slice {z}  ·  input: all 4 modalities  ·  "
                 "Dice = this case, binary (sigmoid > 0.5)", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.05)
    fig.savefig(args.out, dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()

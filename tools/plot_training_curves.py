"""Validation Dice over training, all main-line runs overlaid.

    python tools/plot_training_curves.py [--out figures/training_curves.png]

Val Dice = 15-combo mean over the 50 val cases, every 5 epochs (the number model_best
is picked on). Thin = raw, thick = centred moving average of 5 evals (25 epochs);
dot = the saved model_best. Also prints, per run, the numbers used to judge overfitting.
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[1]
P = "gmd-enum-2-adam-acn-split2-lrfix"
RUNS = [  # (label, checkname, colour) - colour follows the run, fixed order (validated palette)
    ("#1 baseline (HD†)", P, "#2a78d6"),
    ("#2 + region-aware gating (stopped ep 180)", f"{P}-gate", "#eb6834"),
    ("#3 + cross-modal attn", f"{P}-mattn", "#1baf7a"),
    ("#4 = #3 + self-bias", f"{P}-v1-selfbias2", "#eda100"),
    ("#5 = #4 + DW conv (Q, K)", f"{P}-v1-selfbias2-dwqk3", "#e87ba4"),
    ("#6 = #5 + GSS", f"{P}-v1-selfbias2-dwqk3-gss", "#008300"),
    ("HD + GSS", f"{P}-hd-gss", "#6250d6"),
]
INK, MUTED, GRID = "#1D2230", "#66707F", "#E3E6EB"


def load(ck):
    val = {}
    for f in sorted(glob.glob(str(REPO / "results" / "brats3d-acn" / ck / "experiment_*" / "events*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        per = {}
        for t in ea.Tags()["scalars"]:
            if re.fullmatch(r"test/dice_drop.*", t):
                for e in ea.Scalars(t):
                    per.setdefault(e.step, []).append(e.value)
        for s, v in per.items():
            if len(v) == 15:
                val[s] = float(np.mean(v))
    ep = np.array(sorted(val))
    return ep, np.array([val[e] for e in ep])


def smooth(y, k=5):
    out = np.full_like(y, np.nan)
    h = k // 2
    for i in range(h, len(y) - h):
        out[i] = y[i - h:i + h + 1].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures" / "training_curves.png"))
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(figsize=(13, 7))
    print(f"{'run':44s}{'best (raw)':>16s}{'smoothed peak':>18s}{'end (last 5)':>14s}"
          f"{'end-peak':>10s}{'val slope 300-end':>24s}")
    for label, ck, col in RUNS:
        ep, v = load(ck)
        if len(ep) == 0:
            continue
        s = smooth(v)
        ax.plot(ep, v, color=col, lw=0.8, alpha=0.25)
        ax.plot(ep, s, color=col, lw=2.2, label=label, solid_capstyle="round")
        b = int(np.argmax(v))
        ax.plot(ep[b], v[b], "o", ms=8, color=col, mec="white", mew=2, zorder=5)

        pk = int(np.nanargmax(s))
        end = v[-5:].mean()
        late = ep >= 300
        slope = t = np.nan
        if late.sum() > 10:
            A = np.vstack([ep[late], np.ones(late.sum())]).T
            coef, *_ = np.linalg.lstsq(A, v[late], rcond=None)
            r = v[late] - A @ coef
            se = np.sqrt(r @ r / (late.sum() - 2) / ((ep[late] - ep[late].mean()) ** 2).sum())
            slope, t = coef[0] * 100, coef[0] / se
        print(f"{label:44s}{v[b]:9.4f} @{ep[b]:4d}{s[pk]:11.4f} @{ep[pk]:4d}{end:14.4f}{end - s[pk]:+10.4f}"
              + (f"{slope:+14.4f}/100ep t={t:+5.1f}" if not np.isnan(slope) else f"{'—':>24s}")
              + f"  (last ep {ep[-1]})")

    ax.set_ylim(0.70, 0.83)
    ax.set_xlim(0, 605)
    ax.set_ylabel("val Dice, 15-combo mean (y-axis starts at 0.70)")
    ax.set_xlabel("epoch")
    ax.set_title("Validation Dice over training (val = 50 cases, every 5 epochs)",
                 fontsize=14, loc="left", color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", fontsize=10.5, frameon=False, ncol=2)
    ax.text(0.005, 0.985, "thin = raw · thick = moving average of 5 evals (25 epochs) · dot = saved model_best",
            transform=ax.transAxes, va="top", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.09)
    fig.savefig(args.out, dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()

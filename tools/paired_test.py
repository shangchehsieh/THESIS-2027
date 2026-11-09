"""Paired tests between the main-line models, on per-case Dice (results/per_case/*.npz).

    python tools/paired_test.py > logs/paired_test_100.log

Plan, fixed before looking at the per-case numbers (2026-11-09):
- Unit = test case (n = 100). A case's score = its Dice averaged over the 15 combos
  (or over a named subgroup of combos); "avg" = mean of WT, TC, ET.
- Primary metric = per-case soft Dice, the formula of HD's eval code (utils/metrics.py,
  smooth = 1) applied to one case instead of a batch of 8. Secondary = per-case binary
  Dice (sigmoid > 0.5), same smoothing.
- Test = two-sided Wilcoxon signed-rank; also mean difference with a paired bootstrap
  95% CI (10,000 resamples, seed 0). Holm correction inside each family.
- Family A, steps: #3-#1, #4-#3, #5-#4, #6-#5 (x WT, TC, ET, avg = 16 tests).
- Family B, vs baseline #1: #0, #2, #3, #4, #5, #6 (x 4 = 24 tests).
- Family C, hypotheses from the 15-combo tables (3 tests):
  C1 #5-#4 ET over the 8 combos with T1c   (DW raises ET when T1c is there)
  C2 #5-#4 WT over T1 / T1c / T1c+T1        (DW hurts WT without FLAIR and T2)
  C3 #6-#5 WT over T1 / T1c / T1c+T1        (GSS repairs that WT)
Sanity check first: batch-of-8 pooled soft Dice rebuilt from the per-case sums must
match the eval_test logs.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import compare_hd  # noqa: E402

D = REPO / "results" / "per_case"
TAGS = ["0_HD_original", "1_baseline", "2_gating", "3_attn", "4_selfbias", "5_dwconv", "6_gss"]
R = ["WT", "TC", "ET"]
SPARSE = ["T1", "T1c", "T1c,T1"]


def load(tag):
    z = np.load(D / f"{tag}.npz")
    return {k: z[k] for k in z.files}


def per_case(z, kind):
    return (2 * z[f"{kind}_i"] + 1) / (z[f"{kind}_s"] + 1)  # [case, combo, region]


def pooled(z, bs=8):
    """eval_test's number: soft Dice pooled over each batch of 8, mean over batches, mean over combos."""
    n = z["soft_i"].shape[0]
    b = [((2 * z["soft_i"][s:s + bs].sum(0) + 1) / (z["soft_s"][s:s + bs].sum(0) + 1)) for s in range(0, n, bs)]
    return 100 * np.mean(b, axis=0).mean(0)  # [region]


def score(d, combos, sel, region):
    idx = [i for i, c in enumerate(combos) if sel(c)]
    x = d[:, idx]
    return (x.mean(2) if region == "avg" else x[..., R.index(region)]).mean(1)


def boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def family(name, specs, data, combos, kind):
    """specs: (model a, model b, region, combo filter, label). Holm over the whole family."""
    rows = []
    for a, b, r, sel, label in specs:
        diff = 100 * (score(data[a][kind], combos, sel, r) - score(data[b][kind], combos, sel, r))
        rows.append((label, r, diff.mean(), boot_ci(diff), (diff > 0).sum(), wilcoxon(diff).pvalue))
    adj = holm([row[-1] for row in rows])
    print(f"\n{name}")
    print(f"  {'comparison':26s}{'':5s}{'Δ mean':>8s}{'95% CI':>18s}{'cases up':>10s}{'p':>9s}{'p Holm':>9s}")
    for (label, r, m, ci, up, p), pa in zip(rows, adj):
        flag = " *" if pa < 0.05 else ""
        print(f"  {label:26s}{r:5s}{m:+8.2f}{f'[{ci[0]:+.2f}, {ci[1]:+.2f}]':>18s}{up:>6d}/100{p:9.4f}{pa:9.4f}{flag}")


ALL = lambda c: True  # noqa: E731


def pairs(ps, regions=("WT", "TC", "ET", "avg")):
    return [(a, b, r, ALL, f"#{a[0]} vs #{b[0]}") for a, b in ps for r in regions]


def main():
    data = {}
    for t in TAGS:
        z = load(t)
        data[t] = z | {"soft": per_case(z, "soft"), "bin": per_case(z, "bin")}
    combos = list(data[TAGS[0]]["combos"])
    assert all((data[t]["cases"] == data[TAGS[0]]["cases"]).all() for t in TAGS)

    print(f"Per-case paired tests, {len(data[TAGS[0]]['cases'])} test cases x {len(combos)} combos")
    print("\nSanity: batch-of-8 pooled soft Dice rebuilt from per-case sums vs eval_test log (15-combo mean)")
    worst = 0.0
    for t in TAGS:
        log = compare_hd.table(str(REPO / "logs" / f"eval_test_{t}.log"))
        ref = [np.mean([v[i] for v in log.values()]) for i in range(3)]
        got = pooled(data[t])
        worst = max(worst, max(abs(g - r) for g, r in zip(got, ref)))
        print(f"  {t:14s} rebuilt " + " ".join(f"{g:.2f}" for g in got) + "   log " + " ".join(f"{r:.2f}" for r in ref))
    print(f"  max |diff| = {worst:.3f} pt" + ("  OK" if worst < 0.05 else "  MISMATCH"))

    print("\nPer-case means, 15-combo (soft | binary):")
    for t in TAGS:
        s = [100 * score(data[t]["soft"], combos, lambda c: True, r).mean() for r in R]
        b = [100 * score(data[t]["bin"], combos, lambda c: True, r).mean() for r in R]
        print(f"  {t:14s} soft " + " ".join(f"{v:6.2f}" for v in s) + f"  avg {np.mean(s):6.2f}"
              + "  | bin " + " ".join(f"{v:6.2f}" for v in b) + f"  avg {np.mean(b):6.2f}")

    steps = [("3_attn", "1_baseline"), ("4_selfbias", "3_attn"), ("5_dwconv", "4_selfbias"), ("6_gss", "5_dwconv")]
    vsb = [(t, "1_baseline") for t in ["0_HD_original", "2_gating", "3_attn", "4_selfbias", "5_dwconv", "6_gss"]]
    for kind, label in (("soft", "PRIMARY: per-case soft Dice"), ("bin", "SECONDARY: per-case binary Dice")):
        print(f"\n===== {label} =====")
        family("Family A — each step vs the one before", pairs(steps), data, combos, kind)
        family("Family B — vs baseline #1", pairs(vsb), data, combos, kind)
        family("Family C — hypotheses on combo subgroups", [
            ("5_dwconv", "4_selfbias", "ET", lambda c: "T1c" in c.split(","), "C1 #5 vs #4, 8 T1c combos"),
            ("5_dwconv", "4_selfbias", "WT", lambda c: c in SPARSE, "C2 #5 vs #4, T1/T1c/T1c+T1"),
            ("6_gss", "5_dwconv", "WT", lambda c: c in SPARSE, "C3 #6 vs #5, T1/T1c/T1c+T1"),
        ], data, combos, kind)


if __name__ == "__main__":
    main()

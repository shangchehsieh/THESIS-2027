"""Compare an eval_test log against HD† (our reproduction of the baseline).

    python3 tools/compare_hd.py logs/eval_test_<MMDD>.log [baseline_log]

Parses the "per-combination summary" table that eval_test.py prints and reports
the 15-combo mean, the delta per region, and how many of the 15 combos beat HD†.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = "logs/eval_test_base_lr_fix.log"   # HD†: 88.80 / 83.25 / 69.57
NOISE = 1.0                                     # points; within this = on par


def table(path):
    """{combo: [WT, TC, ET]} in percent, from an eval_test log."""
    rows, on = {}, False
    for line in open(REPO / path):
        if "per-combination summary" in line:
            on = True
            continue
        if on:
            m = re.match(r"\s+([A-Za-z0-9,]+)\s+(\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if m:
                rows[m.group(1)] = [100 * float(m.group(i)) for i in (3, 4, 5)]
    if len(rows) != 15:
        raise ValueError(f"{path}: found {len(rows)} combos, expected 15 (eval not finished?)")
    return rows


def compare(run_log, base_log=BASELINE):
    run, base = table(run_log), table(base_log)
    mean = lambda t: [sum(v[i] for v in t.values()) / 15 for i in range(3)]
    r, b = mean(run), mean(base)
    ra, ba = sum(r) / 3, sum(b) / 3
    out = [f"{'':10}{'WT':>8}{'TC':>8}{'ET':>8}{'Avg':>8}",
           f"{'this run':10}" + "".join(f"{v:8.2f}" for v in r + [ra]),
           f"{'HD†' if base_log == BASELINE else 'reference':10}" + "".join(f"{v:8.2f}" for v in b + [ba]),
           f"{'Δ':10}" + "".join(f"{v:+8.2f}" for v in [x - y for x, y in zip(r, b)] + [ra - ba])]
    wins = [sum(run[c][i] > base[c][i] for c in run) for i in range(3)]
    out.append(f"{'wins /15':10}" + "".join(f"{w:8d}" for w in wins))
    out.append("\nper combo Δ (WT / TC / ET):")
    out += [f"  {c:<14}" + " ".join(f"{run[c][i] - base[c][i]:+6.2f}" for i in range(3)) for c in run]
    d = ra - ba
    out.append(f"\nverdict on avg: {'on par (within ±%.0f pt noise)' % NOISE if abs(d) <= NOISE else ('better' if d > 0 else 'worse')}")
    return "\n".join(out)


if __name__ == "__main__":
    try:
        print(compare(*sys.argv[1:3]))
    except ValueError as e:
        sys.exit(str(e))

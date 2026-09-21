"""Regenerate the train/val/test split for the BraTS2020 npy dataset.

Sizes follow RFNet's BraTS2020 protocol (219 / 50 / 100), reduced to 99 test
cases because the local copy is missing BraTS20_Training_355 -- 368 cases are
available, not 369.

Two modes:

  --stratify (default)  balance the splits on the two things that actually skew
                        the score: how hard the enhancing tumour is to find
                        (absent / tiny / normal) and whole-tumour volume.
                        27 cases have no ET at all and 10 more have under 500
                        ET voxels; a plain random draw scatters these unevenly
                        across only 50 val and 99 test cases, which moves the
                        ET number without telling you anything about the model.

  --no-stratify         plain uniform random draw.

Seeded, so the split is reproducible. Every available case is used exactly once
and the three lists never overlap. Overwrites train/val/test.txt in place after
taking a timestamped backup; --dry-run reports the distribution and writes
nothing.

    python tools/make_random_split.py [--seed N] [--no-stratify] [--dry-run]
"""
import argparse
import os
import random
import shutil
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "BRATS2020_Training_none_npy")
SIZES = [("train", 219), ("val", 50), ("test", 99)]
TINY_ET = 500


def available_cases():
    vol = {f[:-8] for f in os.listdir(os.path.join(ROOT, "vol"))
           if f.endswith("_vol.npy")}
    seg = {f[:-8] for f in os.listdir(os.path.join(ROOT, "seg"))
           if f.endswith("_seg.npy")}
    return sorted(vol & seg)


def features(ids):
    """Per case: ET-difficulty tier (0 absent, 1 tiny, 2 normal) and WT volume."""
    out = {}
    for cid in ids:
        s = np.asarray(np.load(os.path.join(ROOT, "seg", f"{cid}_seg.npy"),
                               mmap_mode="r"))
        et = int((s == 3).sum())
        tier = 0 if et == 0 else (1 if et < TINY_ET else 2)
        out[cid] = (tier, int((s > 0).sum()))
    return out


def deal(groups, rng):
    """Hand each stratum out to the splits in proportion to their sizes."""
    total = sum(n for _, n in SIZES)
    out = {name: [] for name, _ in SIZES}
    for key in sorted(groups, key=lambda k: (-len(groups[k]), str(k))):
        g = groups[key][:]
        rng.shuffle(g)
        i = 0
        for name, n in SIZES:
            take = int(len(g) * n / total)
            out[name].extend(g[i:i + take])
            i += take
        for j, c in enumerate(g[i:]):            # rounding remainder
            out[SIZES[j % len(SIZES)][0]].append(c)

    # correct any drift so the target sizes are hit exactly
    rng2 = random.Random(12345)
    target = dict(SIZES)
    while True:
        over = [n for n in target if len(out[n]) > target[n]]
        under = [n for n in target if len(out[n]) < target[n]]
        if not over or not under:
            break
        out[under[0]].append(out[over[0]].pop(rng2.randrange(len(out[over[0]]))))
    return out


def describe(name, ids, feat):
    n = len(ids)
    no_et = sum(1 for c in ids if feat[c][0] == 0)
    tiny = sum(1 for c in ids if feat[c][0] == 1)
    v = np.array([feat[c][1] for c in ids], dtype=float)
    return (f"  {name:<7}{n:>5}{no_et:>7}{tiny:>7}{no_et + tiny:>8}"
            f"{100 * (no_et + tiny) / n:>9.1f}%{np.median(v) / 1000:>12.1f}k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ids = available_cases()
    if len(ids) != sum(n for _, n in SIZES):
        raise SystemExit(f"have {len(ids)} cases but sizes sum to "
                         f"{sum(n for _, n in SIZES)}")
    feat = features(ids)
    rng = random.Random(args.seed)

    if args.no_stratify:
        shuffled = ids[:]
        rng.shuffle(shuffled)
        split, i = {}, 0
        for name, n in SIZES:
            split[name] = shuffled[i:i + n]
            i += n
        mode = "plain random"
    else:
        vols = np.array([feat[c][1] for c in ids])
        t1, t2 = np.percentile(vols, [33.3, 66.6])
        groups = {}
        for c in ids:
            tier, v = feat[c]
            bucket = 0 if v <= t1 else (1 if v <= t2 else 2)
            groups.setdefault((tier, bucket), []).append(c)
        split = deal(groups, rng)
        mode = f"stratified on ET tier (absent / <{TINY_ET} vox / normal) x WT tertile"

    for n in split:
        split[n] = sorted(split[n])

    sets = {k: set(v) for k, v in split.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    assert set().union(*sets.values()) == set(ids), "not every case used"
    assert all(len(split[n]) == c for n, c in SIZES), "size mismatch"

    print(f"cases: {len(ids)}   seed: {args.seed}   mode: {mode}\n")
    print(f"  {'split':<7}{'n':>5}{'no-ET':>7}{'tinyET':>7}{'hard':>8}{'hard%':>10}{'medWT':>12}")
    print("  " + "-" * 58)
    for name, _ in SIZES:
        print(describe(name, split[name], feat))
    print("  " + "-" * 58)
    print(describe("ALL", ids, feat))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bk = os.path.join(ROOT, f"split_backup_{stamp}")
    os.makedirs(bk, exist_ok=True)
    for name, _ in SIZES:
        src = os.path.join(ROOT, f"{name}.txt")
        if os.path.exists(src):
            shutil.copy2(src, bk)
    print(f"\nprevious lists backed up to {bk}/")
    for name, _ in SIZES:
        with open(os.path.join(ROOT, f"{name}.txt"), "w") as f:
            f.write("\n".join(split[name]) + "\n")
    print("train.txt / val.txt / test.txt rewritten")


if __name__ == "__main__":
    main()

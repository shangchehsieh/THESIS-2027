"""Build a clean per-evaluation training log from a run's TensorBoard events.

Source of truth is the tfevents file inside the run's experiment directory, not
the raw stdout log: tfevents is written incrementally, survives stdout being
truncated or deleted, and carries every train loss and every eval scalar.

Safe to run at any time, including while training is still going -- it reports
whatever has been written so far. One shot, no daemon.

    python tools/make_clean_log.py                    # newest run
    python tools/make_clean_log.py --checkname NAME   # a specific run
    python tools/make_clean_log.py --all              # every run found

Writes logs/train_clean_<MMDD>.log, named
by the date the run started (taken from the first event's wall time), so runs
are identified by when you launched them rather than by checkname.
"""
import argparse
import datetime
import glob
import os
import statistics as st

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "logs")
PAPER = (0.8960, 0.8450, 0.7030)


def find_runs():
    """Every experiment dir that holds a tfevents file, newest first."""
    runs = []
    for ev in glob.glob(os.path.join(RESULTS, "*", "*", "experiment_*",
                                     "events.out.tfevents.*")):
        exp = os.path.dirname(ev)
        checkname = os.path.basename(os.path.dirname(exp))
        runs.append((os.path.getmtime(ev), checkname, exp))
    runs.sort(reverse=True)
    return runs


def build(checkname, exp_dir):
    ea = EventAccumulator(exp_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if "train/total_loss_epoch" not in tags:
        print(f"  {checkname}: no epoch scalars yet, skipping")
        return

    scal = ea.Scalars("train/total_loss_epoch")
    epochs = [(s.step, round(s.value, 3)) for s in scal]
    started = datetime.datetime.fromtimestamp(scal[0].wall_time)
    tag = started.strftime("%m%d")
    # two runs on the same day: keep both, disambiguate by checkname suffix
    if os.path.exists(os.path.join(OUT, f"train_clean_{tag}.log")):
        with open(os.path.join(OUT, f"train_clean_{tag}.log")) as f:
            head = f.readline()
        if checkname not in head:
            tag = f"{tag}_{checkname.split('-')[-1]}"

    combos = sorted({t[len("test/dice_drop"):] for t in tags
                     if t.startswith("test/dice_drop")})
    evals = []
    for c in combos:
        series = {k: {s.step: s.value for s in ea.Scalars(f"test/dice{p}drop{c}")}
                  for k, p in [("dice", "_"), ("wt", "_WT_"),
                               ("tc", "_TC_"), ("et", "_ET_")]}
        for e in series["dice"]:
            n = 0 if c == "" else len([x for x in c.split("_") if x])
            evals.append({"epoch": e, "dropped": c or "(none)", "n_dropped": n,
                          **{k: round(series[k][e], 6) for k in series}})
    evals.sort(key=lambda r: (r["epoch"], r["n_dropped"], r["dropped"]))

    os.makedirs(OUT, exist_ok=True)

    by = {}
    for v in evals:
        by.setdefault(v["epoch"], []).append(v)

    path = os.path.join(OUT, f"train_clean_{tag}.log")
    best, best_ep = -1.0, None
    with open(path, "w") as f:
        f.write(f"# run: {checkname}\n")
        f.write(f"# started: {started:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# epochs completed: {len(epochs)}"
                f"{'  (still training)' if len(epochs) < 600 else ''}\n")
        f.write(f"# paper targets: WT {PAPER[0]:.4f}  TC {PAPER[1]:.4f}  "
                f"ET {PAPER[2]:.4f}\n\n")
        for e in sorted(by):
            v = by[e]
            if len(v) != len(combos):        # eval still in progress
                continue
            m = lambda k: st.mean(r[k] for r in v)
            d = m("dice")
            mark = ""
            if d > best:
                best, best_ep, mark = d, e, "   <- best so far"
            f.write(f"epoch {e:3d}{mark}\n")
            f.write("  15-combo mean:   dice {:.4f}   WT {:.4f}   TC {:.4f}   "
                    "ET {:.4f}\n".format(d, m("wt"), m("tc"), m("et")))
            full = [r for r in v if r["dropped"] == "(none)"]
            if full:
                r = full[0]
                f.write("  all-4-modality:  dice {:.4f}   WT {:.4f}   TC {:.4f}   "
                        "ET {:.4f}\n".format(r["dice"], r["wt"], r["tc"], r["et"]))
            f.write("\n")
        if best_ep is not None:
            f.write(f"# best: epoch {best_ep}, 15-combo mean dice {best:.4f}\n")

    done = sum(1 for e in by if len(by[e]) == len(combos))
    print(f"  {checkname}: {len(epochs)} epochs, {done} complete evals"
          + (f", best {best:.4f} @ epoch {best_ep}" if best_ep is not None else "")
          + f"  ->  logs/train_clean_{tag}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkname")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    runs = find_runs()
    if not runs:
        raise SystemExit("no runs with tfevents found under results/")
    if args.checkname:
        runs = [r for r in runs if r[1] == args.checkname]
        if not runs:
            raise SystemExit(f"no run named {args.checkname}")
    elif not args.all:
        runs = runs[:1]

    seen = set()
    for _, checkname, exp in runs:
        if checkname in seen:
            continue          # newest experiment dir per checkname wins
        seen.add(checkname)
        build(checkname, exp)


if __name__ == "__main__":
    main()

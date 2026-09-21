#!/usr/bin/env python3
"""Rebuild logs/ from the TensorBoard event file — the durable source of truth.
Used after train.log was clobbered. Reproduces the same formats as parse_log.py.
"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os, csv, statistics as st, datetime

ROOT="/home/nthu_sc/Brain-Tumor-Segmentation"
EXP=os.path.join(ROOT,"results/brats3d-acn/gmd-enum-2-adam-acn/experiment_00")
OUT=os.path.join(ROOT,"logs"); os.makedirs(OUT,exist_ok=True)

ea=EventAccumulator(EXP,size_guidance={'scalars':0}); ea.Reload()
tags=ea.Tags()['scalars']

# combos, from the tag names
combos=sorted({t[len('test/dice_drop'):] for t in tags
               if t.startswith('test/dice_drop')})

# epochs
epochs=[{"epoch":s.step,"train_loss":round(s.value,3),
         "wall_clock":datetime.datetime.fromtimestamp(s.wall_time).strftime("%Y-%m-%d %H:%M:%S")}
        for s in ea.Scalars('train/total_loss_epoch')]

# evals
evals=[]
for c in combos:
    d ={s.step:s.value for s in ea.Scalars(f'test/dice_drop{c}')}
    wt={s.step:s.value for s in ea.Scalars(f'test/dice_WT_drop{c}')}
    tc={s.step:s.value for s in ea.Scalars(f'test/dice_TC_drop{c}')}
    et={s.step:s.value for s in ea.Scalars(f'test/dice_ET_drop{c}')}
    wall={s.step:s.wall_time for s in ea.Scalars(f'test/dice_drop{c}')}
    for e in d:
        n=0 if c=="" else len([x for x in c.split("_") if x])
        evals.append({"epoch":e,"dropped":c if c else "(none)","n_dropped":n,
                      "dice":round(d[e],6),"wt":round(wt[e],6),
                      "tc":round(tc[e],6),"et":round(et[e],6),
                      "wall_clock":datetime.datetime.fromtimestamp(wall[e]).strftime("%Y-%m-%d %H:%M:%S")})
evals.sort(key=lambda r:(r["epoch"],r["n_dropped"],r["dropped"]))

with open(os.path.join(OUT,"epochs.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,["epoch","train_loss","wall_clock"]); w.writeheader(); w.writerows(epochs)
with open(os.path.join(OUT,"evals.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,["epoch","dropped","n_dropped","dice","wt","tc","et","wall_clock"])
    w.writeheader(); w.writerows(evals)

by={}
for v in evals: by.setdefault(v["epoch"],[]).append(v)
best=-1.0; best_ep=None
with open(os.path.join(OUT,"train_clean.log"),"w") as f:
    f.write("# run: gmd-enum-2-adam-acn (pristine baseline), 600 epochs, batch 8, adam lr 8e-4\n")
    f.write("# one entry per eval, every 5 epochs (epoch 4, 9, 14, ... 599)\n")
    f.write("# paper targets: WT 89.6  TC 84.5  ET 70.3\n")
    f.write("# rebuilt from TensorBoard events after train.log was lost\n\n")
    for e in sorted(by):
        v=by[e]
        if len(v)!=15: continue
        m=lambda k: st.mean(r[k] for r in v)
        d=m("dice")
        mark=""
        if d>best: best=d; best_ep=e; mark="   <- best so far"
        f.write(f"[{v[-1]['wall_clock']}] epoch {e:3d}{mark}\n")
        f.write("  15-combo mean:   dice {:.4f}   WT {:.1f}   TC {:.1f}   ET {:.1f}\n".format(
            d,m("wt")*100,m("tc")*100,m("et")*100))
        fu=[r for r in v if r["dropped"]=="(none)"]
        if fu:
            r=fu[0]
            f.write("  all-4-modality:  dice {:.4f}   WT {:.1f}   TC {:.1f}   ET {:.1f}\n".format(
                r["dice"],r["wt"]*100,r["tc"]*100,r["et"]*100))
        f.write("\n")
print(f"epochs={len(epochs)} evals={len(evals)} combos={len(combos)}")
print(f"BEST: epoch {best_ep}  15-combo mean dice {best:.4f}")

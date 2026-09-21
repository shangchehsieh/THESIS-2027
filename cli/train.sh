#!/bin/bash
# Single-GPU training on the pre-processed BraTS2020 .npy data.
# Activate the env first:  source ~/nthu_sc/bin/activate
# batch_size=8 matches the paper; drop to 4 if you hit an OOM.
set -e
cd "$(dirname "$0")/.."

python test1.py \
    hydra.job.chdir=False \
    distributed=False \
    world_size=1 \
    mode="train" \
    epochs=600 \
    eval_interval=5 \
    optim=adam \
    optim.lr=8e-4 \
    batch_size=8 \
    test_batch_size=8 \
    model="ensemble" \
    model.output="list" \
    model.feature="False" \
    model.width_ratio="0.5" \
    dataset="brats3d_acn" \
    loss="enumeration" \
    loss.missing_num="2" \
    loss.output="list" \
    workers=4 \
    gpu_ids="'0'" \
    trainer="trainer" \
    trainer.method="gmd" \
    checkname="gmd-enum-2-adam-acn" \
    "$@"

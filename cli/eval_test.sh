#!/bin/bash
# Evaluate a trained checkpoint on the held-out test split, over all 15
# modality-drop combinations. Does not touch the training run's artifacts.
#
#   bash cli/eval_test.sh                    # newest run
#   bash cli/eval_test.sh <checkname>        # a specific run
#
# Each invocation writes one log, named by the date you ran it (same <MMDD>
# convention as logs/train_clean_<MMDD>.log):
#   logs/eval_test_<MMDD>.log          per-combination report + 15-combo mean
# A second eval on the same day gets an _HHMM suffix so nothing is overwritten.
# The checkname and checkpoint path are recorded in the log header.
set -e
cd "$(dirname "$0")/.."

# Callers may invoke this outside the venv, so `python` may not be on PATH.
VENV="$HOME/nthu_sc/bin/activate"
[ -f "$VENV" ] && source "$VENV"
command -v python >/dev/null || { echo "python not found (venv: $VENV)" >&2; exit 1; }

CHECKNAME="$1"
if [ -z "$CHECKNAME" ]; then
    # newest checkname that actually has a best checkpoint
    CHECKNAME=$(ls -td results/*/*/ 2>/dev/null \
        | while read -r d; do
              [ -f "$d/model_best.pth" ] && basename "$d" && break
          done)
fi
[ -n "$CHECKNAME" ] || { echo "no run with model_best.pth found" >&2; exit 1; }

RUNDIR=$(ls -d results/*/"$CHECKNAME" 2>/dev/null | head -1)
CKPT="$RUNDIR/model_best.pth"
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }

# This branch is the plain HD baseline (mean fusion only). Checkpoints of the
# cross-modal attention / gating runs need their own branch to be evaluated.
if grep -q "'fusion': '\(mattn\|gate\)'" "$RUNDIR"/experiment_*/parameters.txt 2>/dev/null \
   || [[ "$CHECKNAME" == *-gate* ]]; then
    echo "$CHECKNAME is not a mean-fusion run; evaluate it on branch crossmodal_atten_v1_dwqk_gss" >&2
    exit 1
fi

mkdir -p logs
TAG=$(date +%m%d)
[ -e "logs/eval_test_${TAG}.log" ] && TAG="${TAG}_$(date +%H%M)"
LOG="logs/eval_test_${TAG}.log"
{
    echo "checkname:  $CHECKNAME"
    echo "checkpoint: $CKPT"
    echo "date:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "log:        $LOG"
    echo
} | tee "$LOG"

python eval_test.py \
    hydra.job.chdir=False \
    hydra.run.dir=outputs/eval_hydra \
    distributed=False world_size=1 \
    mode="eval" \
    resume="$CKPT" \
    batch_size=8 test_batch_size=8 \
    model="ensemble" model.output="list" model.feature="False" model.width_ratio="0.5" \
    dataset="brats3d_acn" \
    loss="enumeration" loss.missing_num="2" loss.output="list" \
    workers=4 gpu_ids="'0'" \
    trainer="trainer" trainer.method="gmd" \
    optim=adam optim.lr=8e-4 \
    checkname="${CHECKNAME}-eval" 2>&1 | tee -a "$LOG"

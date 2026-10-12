#!/bin/bash
# Single-GPU training on the pre-processed BraTS2020 .npy data.
#
#   bash cli/train.sh [hydra overrides...]        # detached, returns immediately
#   FG=1 bash cli/train.sh [hydra overrides...]   # foreground (smoke tests, debugging)
#
# Detached is the default because a full run takes ~2 days: it survives SSH /
# VPN / VS Code disconnects, refuses to start a second run on the single GPU,
# and keeps logs/train_clean_<MMDD>.log refreshed every 5 minutes (the training
# process itself only writes tfevents; cli/watch_clean_log.sh is what turns
# those into the readable log).
#
# The raw stdout is not kept. It is filtered down to the lines around a Python
# traceback / CUDA error / OOM kill, and written to
# outputs/crash_<checkname>_<timestamp>.log only if there are any -- tfevents
# records no crash, so this is the only place a failure reason would survive.
#
# batch_size=8 matches the paper; drop to 4 if you hit an OOM.
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)

VENV="$HOME/nthu_sc/bin/activate"
[ -f "$VENV" ] && source "$VENV"
command -v python >/dev/null || { echo "python not found (venv: $VENV)" >&2; exit 1; }

# ---------------------------------------------------------------- inner: train
if [ "$TRAIN_DETACHED" = "1" ] || [ "$FG" = "1" ]; then
    exec python test1.py \
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
fi

# -------------------------------------------------------- outer: launch detached
if pgrep -f "test1[.]py" >/dev/null; then
    echo "error: a training run is already in progress; starting a second one would" >&2
    echo "       make them contend for the single GPU and likely OOM:" >&2
    pgrep -af "test1[.]py" | head -1 >&2
    echo "to stop it first: kill \$(pgrep -f 'test1[.]py' | head -1)" >&2
    exit 1
fi

# checkname only names the crash file and feeds the watcher; keep the default in
# sync with the one passed to test1.py above.
CHECKNAME="gmd-enum-2-adam-acn"
for a in "$@"; do case "$a" in checkname=*) CHECKNAME="${a#checkname=}";; esac; done

mkdir -p "$REPO/logs" "$REPO/outputs"
CRASH="$REPO/outputs/crash_${CHECKNAME}_$(date +%Y%m%d_%H%M%S).log"

# setsid: new session, fully detached from the controlling terminal
# nohup:  ignore SIGHUP
# < /dev/null: do not inherit stdin, so a background read cannot stop it
setsid nohup bash -c "TRAIN_DETACHED=1 bash '$REPO/cli/train.sh' \"\$@\" 2>&1 \
    | stdbuf -oL tr '\\r' '\\n' \
    | grep -E -B5 -A40 'Traceback|Error|error:|Killed|out of memory' > '$CRASH.tmp'; \
    if [ -s '$CRASH.tmp' ]; then mv '$CRASH.tmp' '$CRASH'; else rm -f '$CRASH.tmp'; fi" \
    _ "$@" > /dev/null 2>&1 < /dev/null &

sleep 3
echo "started"
if pgrep -f "test1[.]py" >/dev/null; then
    PID=$(pgrep -f 'test1[.]py' | head -1)
    echo "  PID:        $PID"
    echo "  checkname:  $CHECKNAME"
    setsid nohup bash "$REPO/cli/watch_clean_log.sh" "$PID" "$CHECKNAME" 60 \
        > /dev/null 2>&1 < /dev/null &
    echo
    echo "progress:    cat logs/train_clean_$(date +%m%d).log   (refreshed every 60s)"
    echo "still alive: pgrep -af test1.py"
    echo "on crash:    ls outputs/crash_*.log"
else
    echo "  warning: test1.py not seen yet; the failure reason will be in $CRASH" >&2
fi

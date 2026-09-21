#!/bin/bash
# 在背景啟動訓練,脫離終端機,SSH / VPN / VS Code 斷線都不會殺掉它。
#
#   bash cli/train_bg.sh [hydra overrides...]
#
# 和 cli/train.sh 的差別:這個會立刻返回,訓練繼續在背景跑。
# 之後用 pgrep -af test1.py 確認還活著,看 logs/train_clean_<MMDD>.log 看進度。
#
# 只產生一個檔案:logs/train_clean_<MMDD>.log(由 cli/watch_clean_log.sh 每 5 分鐘
# 從 tfevents 重建)。原始 stdout 不落地;只有在程式崩潰時,traceback 才會被
# 存到 outputs/crash_<checkname>_<timestamp>.log。

set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)

VENV="$HOME/nthu_sc/bin/activate"
[ -f "$VENV" ] || { echo "找不到 venv: $VENV" >&2; exit 1; }

# 單張 GPU,拒絕重複啟動第二個訓練。
if pgrep -f "test1[.]py" >/dev/null; then
    echo "錯誤:已經有訓練在跑了,不要再開一個(單張 GPU 會互搶並可能 OOM):" >&2
    pgrep -af "test1[.]py" | head -1 >&2
    echo "要先停掉的話: kill \$(pgrep -f 'test1[.]py' | head -1)" >&2
    exit 1
fi

# checkname 只用來命名 crash 檔和餵給 watcher;預設值要和 cli/train.sh 一致。
CHECKNAME="gmd-enum-2-adam-acn"
for a in "$@"; do case "$a" in checkname=*) CHECKNAME="${a#checkname=}";; esac; done

mkdir -p "$REPO/logs" "$REPO/outputs"
CRASH="$REPO/outputs/crash_${CHECKNAME}_$(date +%Y%m%d_%H%M%S).log"

# setsid: 開新 session,徹底脫離 controlling terminal
# nohup:  忽略 SIGHUP
# < /dev/null: 不繼承 stdin,避免背景讀取 stdin 被停住
# The raw stdout/stderr stream is not kept. It is filtered down to the lines
# around a Python traceback / CUDA error / OOM kill; that excerpt is written to
# $CRASH only if there is one (tfevents records no crash, so this is the only
# place a failure reason would survive). A clean exit leaves no file behind.
# Any extra arguments are forwarded to train.sh as Hydra overrides, e.g.
#   bash cli/train_bg.sh checkname=gmd-enum-2-adam-acn-split2-lrfix
setsid nohup bash -c "source '$VENV' && bash '$REPO/cli/train.sh' \"\$@\" 2>&1 \
    | stdbuf -oL tr '\\r' '\\n' \
    | grep -E -B5 -A40 'Traceback|Error|error:|Killed|out of memory' > '$CRASH.tmp'; \
    if [ -s '$CRASH.tmp' ]; then mv '$CRASH.tmp' '$CRASH'; else rm -f '$CRASH.tmp'; fi" \
    _ "$@" > /dev/null 2>&1 < /dev/null &

sleep 3
echo "已啟動"
if pgrep -f "test1[.]py" >/dev/null; then
    PID=$(pgrep -f 'test1[.]py' | head -1)
    echo "  PID:        $PID"
    echo "  checkname:  $CHECKNAME"
    # Keep logs/train_clean_<MMDD>.log refreshed for the life of this run.
    setsid nohup bash "$REPO/cli/watch_clean_log.sh" "$PID" "$CHECKNAME" 300 \
        > /dev/null 2>&1 < /dev/null &
    echo
    echo "看進度:  cat logs/train_clean_$(date +%m%d).log   (每 5 分鐘更新,第一次 eval 後才有內容)"
    echo "確認活著: pgrep -af test1.py"
    echo "若崩潰:  ls outputs/crash_*.log"
else
    echo "  警告:還沒看到 test1.py。啟動失敗的原因會在 $CRASH" >&2
fi

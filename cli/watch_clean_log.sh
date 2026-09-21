#!/bin/bash
# Keep logs/train_clean_<MMDD>.log refreshed while a training run is alive.
#
#   bash cli/watch_clean_log.sh <train_pid> [checkname] [interval_seconds]
#
# Exits as soon as that PID is gone, then writes one final update.
#
# The exit check is `kill -0 <pid>` on a specific PID -- deliberately NOT a
# `pgrep -f` pattern. An earlier version of this watcher matched its own parent
# shell's command line, concluded training was still alive, and ran for seven
# days. There is also a hard deadline as a second line of defence.
set -e
cd "$(dirname "$0")/.."

PID="$1"
CHECKNAME="$2"
INTERVAL="${3:-300}"
MAX_HOURS=100

[ -n "$PID" ] || { echo "usage: $0 <train_pid> [checkname] [interval]" >&2; exit 1; }
kill -0 "$PID" 2>/dev/null || { echo "PID $PID is not running" >&2; exit 1; }

ARGS=()
[ -n "$CHECKNAME" ] && ARGS=(--checkname "$CHECKNAME")

DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
echo "watching PID $PID, refreshing every ${INTERVAL}s (deadline ${MAX_HOURS}h)"

while kill -0 "$PID" 2>/dev/null; do
    [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "deadline reached, stopping"; break; }
    python tools/make_clean_log.py "${ARGS[@]}" >/dev/null 2>&1 || true
    sleep "$INTERVAL"
done

python tools/make_clean_log.py "${ARGS[@]}" || true
echo "training PID $PID has exited; final clean log written"

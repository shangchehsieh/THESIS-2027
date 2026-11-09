#!/bin/bash
# One-off queue (2026-11-08): after the GSS run finishes, re-evaluate every
# main-line checkpoint on the 100-case test split (case 355 added), then render
# figures/version_comparison.png. Progress: outputs/requeue_eval_100.log
cd "$(dirname "$0")/.."
while pgrep -f "test1[.]py" >/dev/null; do sleep 300; done
sleep 60
echo "training finished at $(date '+%F %T')"
for ck in \
    gmd-enum-2-adam-acn-split2 \
    gmd-enum-2-adam-acn-split2-lrfix \
    gmd-enum-2-adam-acn-split2-lrfix-mattn \
    gmd-enum-2-adam-acn-split2-lrfix-v1-selfbias2 \
    gmd-enum-2-adam-acn-split2-lrfix-v1-selfbias2-dwqk3 \
    gmd-enum-2-adam-acn-split2-lrfix-v1-selfbias2-dwqk3-gss; do
    echo "[$(date '+%F %T')] eval $ck"
    bash cli/eval_test.sh "$ck" > /dev/null 2>&1
    log=$(ls -t logs/eval_test_*.log | head -1)
    echo "    -> $log : $(grep 'Average scores' "$log" | tail -1)"
done
echo "[$(date '+%F %T')] figure"
~/nthu_sc/bin/python tools/compare_versions_fig.py 2>&1 | grep -v Warn | tail -8
echo "[$(date '+%F %T')] all done"

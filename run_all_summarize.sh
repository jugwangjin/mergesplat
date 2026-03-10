#!/bin/bash
# GHAP + mini-splatting 결과 종합. 각 summarize 스크립트를 해당 디렉토리에서 실행.
#
# GHAP: before_prune / after_prune + merge_thresh_0.01, 0.05, 0.1 (output: experiments_ghap_v2)
# Mini: ms, ms_d (final) + ms_merge_thresh_*, ms_d_merge_thresh_*
#
# Usage:
#   bash run_all_summarize.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " run_all_summarize: GHAP → mini-splatting"
echo "============================================"

# 1) GHAP summarize (out_root = GHAP/outputs/experiments_ghap_v2)
echo ""
echo "########## [1/2] GHAP summarize ##########"
cd "${REPO_ROOT}/ges-splatting"
if [ -d "${REPO_ROOT}/GHAP/outputs/experiments_ghap_v2" ]; then
    python summarize_ghap_results.py "${REPO_ROOT}/GHAP/outputs/experiments_ghap_v2" \
        --ghap_experiments_dir "${REPO_ROOT}/GHAP/experiments" \
        --no_visualize
    echo "  → ${REPO_ROOT}/GHAP/outputs/experiments_ghap_v2/summary.csv, summary.html"
else
    echo "  Skip: GHAP/outputs/experiments_ghap_v2 not found"
fi

# 2) Mini-splatting summarize
echo ""
echo "########## [2/2] Mini-splatting summarize ##########"
cd "${REPO_ROOT}/mini-splatting"
if [ -d "outputs/experiments_mini" ]; then
    python summarize_merge_results.py outputs/experiments_mini
    echo "  → mini-splatting/outputs/experiments_mini/summary.csv, summary.html"
else
    echo "  Skip: mini-splatting/outputs/experiments_mini not found"
fi

echo ""
echo "============================================"
echo " Summarize 완료"
echo "============================================"

#!/bin/bash
# GHAP 전체 파이프라인 → mini-splatting 전체 파이프라인 순서대로 실행.
# cd GHAP → run_ghap_and_merge.sh → cd mini-splatting → run_mini_and_merge.sh
#
# 씬: 360_v2 = room treehill kitchen garden counter bonsai (6개), TAT = Ballroom Church Train (3개)
#
# Usage:
#   bash run_all_ghap_and_mini_merge.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GHAP_SCRIPT="${REPO_ROOT}/GHAP/run_ghap_and_merge.sh"
MINI_SCRIPT="${REPO_ROOT}/mini-splatting/run_mini_and_merge.sh"

echo "============================================"
echo " run_all_ghap_and_mini_merge: GHAP → mini-splatting 순차 실행"
echo "============================================"

exit_ghap=0
exit_mini=0

# 1) GHAP: cd → 실행
echo ""
echo "########## [1/2] GHAP ##########"
cd "${REPO_ROOT}/GHAP"
bash "${GHAP_SCRIPT}" || exit_ghap=$?
echo ""
echo "########## GHAP 종료 (exit $exit_ghap) ##########"

# 2) mini-splatting: cd → 실행
echo ""
echo "########## [2/2] mini-splatting ##########"
cd "${REPO_ROOT}/mini-splatting"
bash "${MINI_SCRIPT}" || exit_mini=$?
echo ""
echo "########## mini-splatting 종료 (exit $exit_mini) ##########"

echo ""
echo "============================================"
echo " 전체 실행 완료"
echo "  GHAP:           exit $exit_ghap"
echo "  mini-splatting: exit $exit_mini"
echo "============================================"
if [ "$exit_ghap" -ne 0 ] || [ "$exit_mini" -ne 0 ]; then
    exit 1
fi

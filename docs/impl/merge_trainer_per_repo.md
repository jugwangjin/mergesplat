# Merge trainer per repo (디렉토리 왔다갔다 제거)

## 목표
- `run_ghap_and_merge.sh`: GHAP 디렉토리만 사용. merge 단계에서 ges-splatting으로 `cd` 하지 않음.
- `run_mini_and_merge.sh`: mini-splatting 디렉토리만 사용. merge 단계에서 ges-splatting으로 `cd` 하지 않음.

## 방식
- **기존 스크립트 덮어쓰기 아님**: `train_and_prune.py`, `ms/train.py`는 그대로 둠.
- **추가**: 각 레포에 **merge 전용 트레이너**와 **graph_merge** 모듈을 두고, run 스크립트는 해당 레포 안에서만 실행.

## GHAP
- `GHAP/graph_merge/`: ges-splatting의 graph_merge와 동일 로직 (graph.py, edge_cost_3dgs.py, merge.py, merge_3dgs.py).
- `GHAP/train_merge.py`: COLMAP scene + 초기 PLY → merge 학습. GHAP의 Scene, GaussianModel, render, dataset 사용. `replace_after_merge` 활용.
- `run_ghap_and_merge.sh`: (1) GHAP from_pointcloud (2) **GHAP에서** `python train_merge.py ...` → 출력: `GHAP/outputs/experiments_ghap/<scene>/merge_thresh_*`. (3) visualize는 선택: ges-splatting 스크립트를 경로로 호출하거나 생략.

## mini-splatting
- `mini-splatting/graph_merge/`: 동일하게 graph_merge 복사.
- `mini-splatting/train_merge.py`: mini-splatting의 Scene, GaussianModel, render 사용.
- `run_mini_and_merge.sh`: (1) ms 학습 (2) **mini-splatting에서** `python train_merge.py ...` → 출력: `mini-splatting/outputs/experiments_mini/<scene>/ms_merge_thresh_*` 등. ges-splatting으로 cd 없음.

## 효과
- save/load, dataset, rasterizer 차이로 인한 불일치 회피 (각 레포가 자기 데이터·자기 렌더러만 사용).
- run 스크립트가 한 레포 안에서만 동작해 디렉토리 왔다갔다 제거.

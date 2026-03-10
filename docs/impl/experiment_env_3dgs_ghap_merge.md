# 3DGS → GHAP → Merge 실험 환경 검증

전체 파이프라인과 결과 비교 흐름을 한 번에 확인하기 위한 문서.

## 1. 파이프라인 개요

```
COLMAP scene (source_path)
    → [GHAP] train_and_prune (from_pointcloud)  →  PLY before_prune(30001) / after_prune(35000)
    → [GHAP] render.py  →  test/, test_before_prune/ (동일 test set)
    → [복사] experiment_ghap/<scene>/from_pointcloud/  ←  final_pointcloud_*.ply, summary_*.json
    → [ges-splatting] train_merge_ghap  →  merge_thresh_0.005, 0.01, 0.015 (beta=2 fix)
    → [ges-splatting] visualize_experiment_results.py  →  comparison_step0_vs_final.html, compare_ghap_vs_merge_iter0.html
    → [선택] summarize_ghap_results.py  →  summary.csv / summary.html
```

- **입력**: 동일 `source_path` (COLMAP: `sparse/0`, `images/`).
- **해상도**: 360_v2 → res=4, full_tat_colmap → res=2 (run_ghap_and_merge.sh에서 일관 적용).

## 2. 단계별 검증

### 2.1 GHAP (3DGS → GHAP)

| 항목 | 위치 | 확인 |
|------|------|------|
| 학습 | `GHAP/run_from_pointcloud.sh` → `train_and_prune.py` | `-s source_path`, `-r resolution`, `--eval`, `--compact`, `--sampling_iter 30001`, iteration 35000 |
| before_prune PLY | `experiments/<scene>_0.1_from_pointcloud/point_cloud/iteration_30001_before_prune/point_cloud.ply` | train_and_prune 저장 |
| after_prune PLY | `.../iteration_35000_after_prune/point_cloud.ply` | train_and_prune 저장 |
| GHAP test 렌더 | `run_ghap_and_merge.sh` 내: render 30001 → `test/` 복사 → `test_before_prune/` 생성, render 35000 → `test/` 복구 | before/after 비교용 |
| 복사 출력 | `experiment_ghap/<scene>/from_pointcloud/` | `final_pointcloud_before_prune.ply`, `final_pointcloud_after_prune.ply`, `summary_*.json` |

### 2.2 Merge (ges-splatting)

| 항목 | 위치 | 확인 |
|------|------|------|
| PLY 입력 | `train_merge_ghap.resolve_ghap_ply_path(ghap_result_dir)` | `experiment_ghap/<scene>/from_pointcloud/` → `final_pointcloud_after_prune.ply` 사용 |
| 씬/카메라 | `sceneLoadTypeCallbacks["Colmap"](..., use_eval=True)` | ges-splatting `dataset_readers.readColmapSceneInfo` (동일 source_path) |
| Test GT 저장 | `merge_process/test_view{i}_gt.jpg` | train_merge_ghap이 test_cameras[i] 원본 이미지 저장 |
| Step0/Final 렌더 | `renders/step00000_test_view*_*.png`, `step{ITER}_test_view*_*.png` | 동일 test_cameras 기준 |
| Run 3종 | `merge_thresh_0.005`, `merge_thresh_0.01`, `merge_thresh_0.015` | `--merge_dist_l2_thresh` 만 다름, beta=2 fix 공통 |

### 2.3 결과 비교 (동일 test set)

| 항목 | 내용 |
|------|------|
| Test set 정의 | `docs/impl/test_set_alignment_ghap_ges.md` 참고. 양쪽 모두 image_name(stem) 정렬 → idx % 8 == 0 → test, 이미지 없으면 카메라 스킵. |
| GT 기준 | **merge_process/test_view*_gt.jpg** = train_merge_ghap이 저장한 test view 원본. 모든 PSNR/SSIM은 이 GT 대비. |
| GHAP 메트릭 | `visualize`/`summarize`: GHAP `test/` 또는 `test_before_prune/` 렌더(00000.png, …)를 merge의 **같은 인덱스** test_view*_gt와 매칭. `_find_ghap_test_renders`(정렬된 파일명) ↔ `_find_merge_gt`(test_view0,1,…) 순서 일치 가정. |
| 비교 순서 | 테이블: GHAP (before prune) → GHAP (after prune) → merge_thresh_0.005 → 0.01 → 0.015. Step0 = GHAP after prune와 동일 PLY이므로 cross-check로 GHAP after vs ours step0 수치가 근접해야 함. |

### 2.4 diff-gaussian-rasterization 재설치

- **위치**: `run_ghap_and_merge.sh`에서 (1) 시작 시 GHAP/GES 각각, (2) 씬별 GHAP 단계 전, (3) 씬별 merge 단계 전에 `pip install .../diff-gaussian-rasterization/ --upgrade`.
- **목적**: GHAP 측 추가 데이터 추출(compaction 등)에 맞는 빌드 유지. **렌더링 결과는 동일**하다고 가정.
- 스크립트 주석에 위 내용 반영됨.

## 3. 경로 요약

| 역할 | 경로 |
|------|------|
| GHAP 학습/렌더 출력 | `GHAP/experiments/<scene>_0.1_from_pointcloud/` (test/, test_before_prune/, point_cloud/) |
| Merge 입력 PLY | `GHAP/experiment_ghap/<scene>/from_pointcloud/` |
| Merge 출력 | `ges-splatting/outputs/experiments_ghap/<scene>/merge_thresh_{0.005,0.01,0.015}/` |
| Visualize GHAP 디렉터리 | `--ghap_experiment_dir` = `GHAP/experiments/<scene>_0.1_from_pointcloud` (test 렌더용) |
| Summarize 기본 GHAP 루트 | `--ghap_experiments_dir` 기본값 `../GHAP/experiments` |

## 4. 수정 이력에 따른 검증 (train_and_prune에서 test 렌더 저장)

- **변경**: run_ghap_and_merge.sh에서 별도 `render.py` 호출 제거. 대신 train_and_prune.py가  
  - iteration 30001 (prune 직전): `test_before_prune/<last_folder>/renders`, `gt` 저장  
  - iteration 35000 (학습 종료): `test/<last_folder>/renders`, `gt` 저장  
- **호환**: visualize의 `_find_ghap_test_renders(ghap_experiment_dir, subdir="test"|"test_before_prune")`는  
  `model_path/<subdir>/<last_folder>/renders` 구조를 기대하므로 그대로 사용 가능.  
- **run_from_pointcloud.sh**: 끝의 `python render.py -m "$directory1"`은 train/ 생성 및 metrics용으로 유지.  
  test/는 이미 35000에서 저장되어 있어 덮어쓰기만 함.

## 5. 체크리스트 (실험 전)

- [ ] `source_path`에 `sparse/0`, `images/` 존재.
- [ ] `GHAP_DIR`, `GES_DIR` (및 데이터 루트)가 실행 환경에 맞게 설정됨.
- [ ] 360_v2 → resolution 4, full_tat_colmap → resolution 2 일치.
- [ ] Merge 완료 시 `point_cloud/iteration_5000/point_cloud.ply` 존재 (ITERATIONS=5000 기준).
- [ ] Visualize 시 merge_thresh 세 디렉터리 모두 있으면 세 run 모두 테이블에 포함; 하나라도 있으면 summarize는 존재하는 run만 집계.

## 6. Adaptive-ratio GHAP (비교용)

`run_ghap_adaptive_ratio.sh`: **run_ghap_and_merge.sh 결과를 본 뒤**, merge가 만든 감소 비율(n_final/n_step0)을 씬별로 sampling_ratio에 반영해 GHAP만 다시 학습하는 스크립트.

- **입력**: `summary.csv` (summarize_ghap_results.py 출력). 기본 경로 `GES_DIR/outputs/experiments_ghap/summary.csv`.
- **계산**: 기준 merge run(기본 `merge_thresh_0.01`)의 `n_step0`(GHAP 수), `n_final`(merge 후 수).  
  `ratio = 0.1 * (n_final / n_step0)` 후 `RATIO_MIN`~`RATIO_MAX`(기본 0.02~0.15)로 clamp.
- **실행**: 각 씬에 대해 `run_from_pointcloud.sh`만 호출 (merge/visualize 없음).
- **출력**: `GHAP/experiments/<scene>_<ratio>_from_pointcloud` (예: `bicycle_0.07_from_pointcloud`).

**Usage**: `SUMMARY_CSV=/path/to/summary.csv bash GHAP/run_ghap_adaptive_ratio.sh`  
비교: 고정 0.1 결과(run_ghap_and_merge) vs 이 스크립트의 adaptive ratio 결과.

---

**스크립트 의존 관계 요약**

| 스크립트 | 의존 | 비고 |
|----------|------|------|
| run_ghap_and_merge.sh | run_from_pointcloud, train_merge_ghap, visualize | GHAP 시 `--compact`로 test_before_prune 생성 |
| run_from_pointcloud.sh | train_and_prune, render.py, metrics.py | 항상 --compact, --eval |
| train_and_prune.py | Scene (PLY: iteration_N / _after_prune / _before_prune) | 30001에서 test_before_prune/, 35000에서 test/ 저장 |
| run_ghap_adaptive_ratio.sh | summary.csv, run_from_pointcloud.sh | GHAP만 실행, merge/visualize 없음 |
| run_visualize_all_ghap.sh | merge_thresh_* 디렉터리, (선택) GHAP_EXPERIMENTS_DIR | GHAP은 scene_0.1_from_pointcloud만 사용 |
| summarize_ghap_results.py | merge_thresh_*, (선택) ghap_experiments_dir | ghap_sampling_ratio 기본 0.1 |

이 문서는 `run_ghap_and_merge.sh`, `run_ghap_adaptive_ratio.sh`, `train_merge_ghap.py`, `visualize_experiment_results.py`, `summarize_ghap_results.py`, `test_set_alignment_ghap_ges.md`와 함께 보면 됨.

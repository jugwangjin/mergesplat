# train_merge_colmap.py (gsplat) — COLMAP 2DGS + merge

## 목적

- **위치**: `gsplat/train_merge_colmap.py`
- **역할**: COLMAP 데이터로 **2DGS만** 학습하면서 **edge-collapse merge** 적용. GES 없음. **Runner 상속 없이** 단일 **MergeTrainer** 클래스로 동작.
- **데이터**: `datasets.colmap.Parser` + `Dataset` (sparse/0, images). GT depth/normal 없음 → **primitive-only edge cost** (primitive color + primitive normal; rgb/depth/normal 비용은 빈 리스트로 0).

## 실행

- **실행 위치**: `gsplat` 디렉터리에서 실행 (데이터셋 경로 gsplat 기준).
  ```bash
  cd gsplat
  python train_merge_colmap.py --data_dir <DATA_ROOT>/<SCENE> --result_dir experiment_densification/<SCENE>/merge/cost_v1
  ```
- **실험 스크립트**: `gsplat/run_experiments_merge_colmap.sh`
  - `DATA_ROOT`(기본: `/mnt/d/tat_dataprocessing/full_tat_colmap`), `SCENES=(Auditorium Church)`.
  - 출력: `experiment_densification/<scene>/{merge,no_merge}/cost_v1|cost_v2`.
  - no_merge: `--merge_start_iter 40000` (merge 비실행).
  - cost_v2: `--use_edge_cost_v2 --merge_dist_l2_thresh 1.0`.

## 구현 요지

### MergeConfig (Config 확장)

- **merge**: `merge_start_iter`, `merge_stop_iter`, `merge_every`, `knn_k`, `use_edge_cost_v2`, `merge_dist_l2_thresh`, `merge_rgb_thresh`, `merge_primitive_color_thresh`, `merge_primitive_normal_thresh`.
- **threshold schedule**: `merge_threshold_warmup_steps`(기본 5000), `merge_threshold_initial_scale`(기본 0.5).  
  - step 0에서 threshold = 기본값 × 0.5 (**더 엄격**, 적게 merge), step 5000까지 **선형 완화**하여 step 5000 이후에는 기본값 × 1.0 유지.  
  - `_compute_merge_pairs(step)` 내부: `initial_scale >= 1` 이면 scale을 1.0까지 선형 감소, `initial_scale < 1` 이면 scale을 1.0까지 선형 증가.  
  - v2: `merge_dist_l2_thresh * scale`, `merge_rgb_thresh * scale`. non-v2: `merge_primitive_color_thresh * scale`, `merge_primitive_normal_thresh * scale`.  
  - 목적: 초반에 빡빡하게 merge 후 점차 완화하여 과도한 collapse 방지.
- **densification 제외**: `merge_protect_steps`(기본 500). 방금 densification 된 가우시안은 이 step 수만큼 merge 대상에서 제외. 0이면 비활성화.
- **depth**: `use_rendered_depth` (기본 True). True면 train view마다 RGB+ED 렌더 → median depth → edge cost에 사용, normal 미사용, `depth_occlusion_margin=0.05`.
- **로깅**: `metric_interval`(기본 500) → val psnr/ssim/lpips/num_gs를 `stats/val_step{step:04d}.json`에 저장. `visualization_interval`(기본 500) → merge 직전/직후 `merge_process/step{step:05d}_*_view*.png` 시각화.
- **비교용 저장**: `compare_save_interval`(기본 2500). 매 2500 step마다 `result_dir/compare/`에 저장.  
  - **merge 구간** (`merge_start_iter <= step <= merge_stop_iter` 이고 `self._edges` 존재): **lineset** → `step{step:05d}_lineset.ply` (Open3D LineSet: means + edges).  
  - **그 외**: **pointcloud** → `step{step:05d}_pointcloud.ply` (means + sh0 기반 RGB, ASCII PLY).  
  - 동일 step 간격으로 merge 시 lineset, 비-merge 시 pointcloud를 남겨 비교용으로 사용.

### MergeTrainer (standalone)

- **상속**: `simple_trainer_2dgs.Runner` 사용 안 함. `create_splats_with_optimizers`, **MergeStrategy**(DefaultStrategy 확장), `rasterization_2dgs` 등만 사용.
- **train()**: 한 루프에서 optimizer step → (metric_interval이면 `measure_metric_and_save`) → 매 step에서 `_merge_step(step)` 호출.
- **merge 조건**:  
  - `merge_start_iter <= step <= merge_stop_iter`  
  - `(step - merge_start_iter) % merge_every == 0`  
  - `(step % reset_every) >= refine_every` (reset 직후 refine 구간에는 merge 안 함)
- **merge 단계**:  
  - use_rendered_depth면 train view 렌더 depth → `compute_edge_costs` / `compute_edge_costs_v2`  
  - `build_knn_graph` + `connect_components` → **엣지 필터** (양 끝 중 하나라도 `merge_protect_until_step > step`이면 제외) → `_compute_merge_pairs(..., step)` → greedy matching → `compute_merged_gaussians` → `apply_merge` → **`_merge_protect_until_step` 갱신** → `update_graph_after_merge`  
  - **strategy.check_sanity** + **strategy_state = strategy.initialize_state()** 로 merge 후 strategy 재초기화
- **시각화**: `visualization_interval`일 때 merge 직전 `_save_merge_vis_frames(step, "_pre")`, 직후 `_post`.
- **비교 저장**: 매 step에서 `_merge_step(step)` 호출 후, `step % compare_save_interval == 0` 이면 `_save_compare_geometry(step)` → merge 구간이면 `_save_compare_lineset_ply(step)`, 아니면 `_save_compare_pointcloud_ply(step)`. lineset은 Open3D 의존.

### graph_merge 쪽 (gsplat)

- **edge_cost.py**: `_cov2d_from_rot_scale`; `images` 빈 리스트면 primitive-only 반환. **use_gt_normal**, **depth_occlusion_margin** (렌더 depth 시 0.05).
- **edge_cost_v2.py**: unified L2² × normal gate; L2²에 **.abs()** 적용; `use_gt_normal` 전달.
- **merge.py**: `greedy_edge_matching_7` 등.
- **graph_merge_trainer**: use_mapanything_data 옵션 (MapAnything 데이터 쓸 때만; train_merge_colmap은 use_rendered_depth로 렌더 depth 사용).

---

## Densification과 Merge 상호작용

**채택 방향**: **densification으로 생긴 / 방금 densification 된 가우시안은 merge에서 제외**한다. Sibling(같은 부모에서 나온 쌍)을 merge 후보에서 빼서, split으로 늘린 표현력은 한동안 유지하고, 보호 기간이 지나면 일반 threshold로 merge 가능하게 한다.

**Neighbor graph 끊기**: densification 유래 가우시안의 이웃만 그래프에서 끊는 방식도 생각할 수 있으나, **별도 이슈**로 두고, 당장은 “merge 후보 엣지에서만 제외”만 적용한다.

### 구현 요지 (merge 제외)

- **보호 기간**: Strategy가 split/duplicate로 점을 추가한 뒤, 해당 인덱스에 `merge_protect_until_step = step + merge_protect_steps`를 부여. `step < merge_protect_until_step`인 가우시안은 merge 대상에서 제외.
- **추가 감지**: `step_post_backward` 직후 `len(means)`가 이전보다 크면, **새로 붙은 인덱스** (기존 N ~ 새 N-1)를 “방금 densification 된” 것으로 간주. (Strategy가 새 점을 항상 뒤에 append한다고 가정.)
- **엣지 필터**: `_merge_step`에서 edge matching 전에, 양 끝 중 하나라도 보호 중이면 해당 엣지를 제거한 뒤 `_compute_merge_pairs` 호출.
- **merge 후 갱신**: `apply_merge` 후 인덱스가 바뀌므로 `_merge_protect_until_step`을 새 인덱스 체계에 맞게 갱신. (kept → 기존 값 유지, merged → 두 원소의 max(보호 step) 유지.)
- **옵션**: `MergeConfig.merge_protect_steps` (기본 500). 0이면 제외 로직 비활성화.
- **merge 타이밍**: `merge_timing=True`로 실행하면 merge 시 구간별 시간을 출력한다.  
  `render_views`: train 뷰 전부 렌더(use_rendered_depth일 때 지배적), `cost`: 엣지 비용 계산, `greedy`: greedy 매칭, `apply`: merge 적용·그래프 갱신.  
  병목이 cost인지 greedy인지 확인할 때 사용.

### Densification 시 엣지 갱신 (MergeStrategy + update_graph_after_densify)

- **목적**: N이 바뀔 때 `_edges`를 무효화해 다음 merge에서 KNN 전체 재구축하는 대신, clone/split/prune 결과에 맞춰 엣지 인덱스만 갱신.
- **MergeStrategy** (`gsplat/gsplat/strategy/merge_strategy.py`): `DefaultStrategy` 상속. `step_post_backward` 내부에서 `_grow_gs_with_info`로 clone/split 인덱스 수집, `_prune_mask`로 keep_mask 계산 후, `_last_densify_graph_info`에 `N0`, `clone_parents`, `split_sources`, `N_after_clone`, `keep_mask` 저장.
- **트레이너**: `step_post_backward` 호출 후 `getattr(strategy, "_last_densify_graph_info", None)`이 있으면 `update_graph_after_densify(self._edges, **graph_info)`로 `_edges` 갱신, `_edges_N`을 keep_mask.sum()으로 동기화. N 변경 시 `_edges = None`으로 무효화하던 로직은 제거.

---

데이터셋 경로는 **gsplat 실행 디렉터리 기준**이며, `run_experiments_merge_colmap.sh`에서 `DATA_ROOT`/`SCENE`으로 `--data_dir`를 넘긴다.

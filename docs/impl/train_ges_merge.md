# train_ges_merge.py 상세 설명

## 1. 목적과 위치

- **역할**: GES(Generalized Elliptical Splatting) 학습에 **그래프 기반 edge-collapse merge**를 붙인 학습 스크립트.
- **데이터**: `train_ges.py`와 동일하게 **Scene** (COLMAP/Blender) 기반. MapAnything·dense unprojection 없음.
- **모델**: **LaplacianModel** (3DGS + GES shape). 2DGS + beta 고정 옵션(`--gsplat_like`) 지원.
- **차이점**: `train_merge.py`는 MapAnything dense init + 4-neighbor+KN hybrid 그래프. 여기는 **K-NN만** 쓰고, densification 후 그래프를 **갱신**하며, 주기적으로 **3-ring K-NN 재구축**을 한다.

---

## 2. 진입점과 인자

- `if __name__ == "__main__"`: `ModelParams`, `OptimizationParams`, `PipelineParams` + merge/그래프 관련 인자 파싱.
- `args.iterations = 25000` 고정.
- `training(dataset, opt, pipe, merge_every=500, merge_start_iter=500, ...)` 호출.

---

## 3. 초기화 (training 함수 상단)

- **Scene + LaplacianModel** 생성, `gaussians.training_setup(opt)`, 체크포인트 복원(있으면).
- **2DGS + shape 고정**:
  - `gaussians._scaling[:, 2] = log(1e-6)` (normal 축 스케일 고정).
  - `gaussians._shape` = `beta_use` (기본 2.0, `--gsplat_like` 시 2.0), `requires_grad_(False)`.
- **Edge cost용 카메라 텐서**: `_camera_tensors_for_edge_cost(train_cams, device)` → `ec_images`, `ec_normals`(dummy), `ec_c2ws`, `ec_Ks`. Depth는 merge 시점에 **현재 모델로 렌더**한 `render_laplacian_depth`로 채움.
- **그래프 상태**: `edges = None`, `merge_run_count = 0`.

---

## 4. 메인 루프 (iteration)

매 iteration에서 **순서**는 대략 다음과 같다.

1. Learning rate / SH degree 업데이트  
2. 랜덤 뷰 선택 → 렌더 → 손실( L1 + DSSIM + DoG mask ) → `loss.backward()`  
3. **Densification** (그래프 있으면 **갱신**)  
4. **그래프가 없고** `iteration >= merge_start_iter` 이면 **최초 그래프 구축**  
5. **Merge 조건**이면 edge cost → greedy matching → merge 적용 → 그래프 갱신 → (선택) 3-ring 재구축  
6. `optimizer.step()`  
7. 2DGS clamp / beta 고정  
8. saving / checkpoint

---

## 5. Densification과 그래프 갱신

- **조건**: `iteration < opt.densify_until_iter` 안에서만 densification 수행.
- **통계 누적**: `max_radii2D`, `add_densification_stats(viewspace_point_tensor, visibility_filter)`.
- **densify_and_prune** (주기: `densification_interval`):
  - `return_graph_info=(edges is not None)` 이면 clone/split/prune 정보 반환.
  - `edges`가 있을 때: `update_graph_after_densify(edges, N0, clone_parents, split_sources, N_after_clone, keep_mask)` 호출 후 `edges`를 갱신된 그래프로 교체. **그래프를 버리지 않고** clone/split/prune을 반영.
- **size_prune** (주기: `shape_pruning_interval`):
  - `return_keep_mask=(edges is not None)` 이면 유지 마스크 반환.
  - `edges`가 있을 때: `update_graph_after_prune(edges, keep_mask)` 로 엣지 재인덱싱.
- **reset_opacity / reset_shape**: 주기적으로 호출 (train_ges와 동일).

즉, **densification이 노드를 늘리거나 줄여도** 그래프는 invalidate하지 않고, `update_graph_after_densify` / `update_graph_after_prune`로 **인덱스만 맞춰서 유지**한다.

---

## 6. 그래프 최초 구축

- **조건**: `edges is None` 이고 `iteration >= merge_start_iter`.
- **방법**:
  - `build_knn_graph(gaussians.get_xyz.detach(), k=knn_k, normals=prim_n, normal_scale_at_180=...)`  
    → PyKeOps(있으면) 또는 scipy cKDTree로 **전역 K-NN** (normal-scaled 거리).
  - `connect_components(gaussians.get_xyz.detach(), edges)`  
    → 여러 연결 성분이 있으면 **bridge 엣지** 추가해 하나로 연결.
- 이후 `edges`를 CPU로 두고, merge 단계에서만 GPU로 올려서 사용.

---

## 7. Merge 단계

- **조건**:  
  `edges is not None` 이고  
  `merge_start_iter <= iteration <= merge_stop_iter` 이고  
  `iteration % merge_every == 0`.

- **순서**:
  1. **Depth 렌더**: `ec_depths = [render_laplacian_depth(cam, ...) for cam in train_cams]`  
     → Edge cost에서 visibility·depth 비용에 사용.
  2. **Edge cost**:  
     `compute_edge_costs(edges_gpu, xyz, prim_features, prim_normals, ec_images, ec_depths, ec_normals, ec_c2ws, ec_Ks, ...)`  
     → 7종 비용: primitive color, rgb, depth, normal, primitive normal, normal_dist, scale_diff.
  3. **Greedy matching**:  
     `greedy_edge_matching(edges_gpu, prim_c, rgb_c, dep_c, nrm_c, prim_nrm_c, normal_dist_c, scale_diff_c, t1..t7)`  
     → 각 비용이 threshold 이하인 엣지만 후보로, combined cost 오름차순으로 **비중첩 pair** 선택.
  4. **Merge 적용**:  
     `compute_merged_ges(gaussians, pairs)` → 병합된 파라미터 계산.  
     `apply_merge_to_model(gaussians, pairs, merged)` → 모델에 반영.
  5. **그래프 갱신**:  
     `edges = update_graph_after_merge(edges_gpu, pairs, N_before).cpu()`  
     → Merge된 쌍에 따라 노드 인덱스만 재매핑 (K-NN 재계산 없음).
  6. **merge_run_count** 증가.  
     `graph_rebuild_every_merge_runs > 0` 이고 `merge_run_count % graph_rebuild_every_merge_runs == 0` 이면 **그래프 재구축** (아래 8).

---

## 8. 그래프 재구축 (3-ring K-NN)

- **시점**: Merge를 실행한 뒤, `merge_run_count`가 `graph_rebuild_every_merge_runs`의 배수일 때.
- **방법**:
  - `knn_from_graph_ring(edges, gaussians.get_xyz.detach(), k=knn_k, ring=3, normals=prim_n, normal_scale_at_180=...)`  
    → **현재 그래프의 3-ring** 안에서만 거리 재계산 후, 노드당 상위 k개로 엣지 재정의. 전역 K-NN/PyKeOps 호출 없음.
  - `connect_components(...)` 로 island 다시 연결.
- **역할**: Merge로 위치가 바뀐 뒤에도 이웃 관계를 “가까운 3-hop 이내”로 다시 정리해, 다음 merge 품질을 유지.

---

## 9. Edge cost용 카메라 (_camera_tensors_for_edge_cost)

- **입력**: Scene의 train cameras.
- **출력**:
  - `images`: (H, W, 3), `original_image` permute.
  - `c2ws`: `world_view_transform` 역행렬로 c2w (4x4).
  - `Ks`: (fx, fy, cx, cy)로 3x3 K.
  - `normals`: 더미 (H, W, 3), z=1. 실제 normal 비용은 primitive normal 등으로 계산.
- Depth는 **merge 시점**에 `render_laplacian_depth`로 채우므로, 이 함수는 이미지·K·c2w·더미 normal만 준비.

---

## 10. CLI 인자 요약

| 인자 | 기본값 | 설명 |
|------|--------|------|
| merge_every | 500 | Merge 주기 (iteration). |
| merge_start_iter | 500 | Merge 시작 iteration. |
| merge_stop_iter | 25000 | Merge 종료 iteration. |
| merge_prim_color_thresh ~ merge_scale_diff_thresh | (각각) | 7종 edge cost threshold. |
| depth_occlusion_margin | 0.3 | Visibility 판정. |
| knn_k | 5 | K-NN / 3-ring 재구축 시 노드당 이웃 수. |
| graph_rebuild_every_merge_runs | 10 | N번 merge마다 3-ring 재구축; 0이면 재구축 없음. |
| graph_rebuild_normal_scale_at_180 | 2.0 | Normal 반대일 때 effective 거리 배율. |
| init_beta | 2.0 | GES shape (고정). |
| gsplat_like | false | True면 2DGS + beta=2 + shape 고정. |
| test_iterations, save_iterations, checkpoint_iterations | (각각) | 테스트/저장/체크포인트 iteration. |
| start_checkpoint | None | 복원할 체크포인트. |
| seed | 0 | 랜덤 시드. |

---

## 11. train_ges.py와의 차이

- **train_ges**: Densification + 일반 최적화만. Merge·그래프 없음.
- **train_ges_merge**:
  - `merge_start_iter` 이후 주기적으로 **그래프 구축 → edge cost → greedy merge → 그래프 갱신**.
  - Densification으로 N이 바뀌어도 **그래프를 버리지 않고** `update_graph_after_densify` / `update_graph_after_prune`로 유지.
  - Merge를 N번 할 때마다 **3-ring K-NN 재구축**으로 이웃 관계 갱신.

---

## 12. train_merge.py(MapAnything)와의 차이

- **train_merge**: MapAnything dense init, **hybrid 그래프**(4-neighbor + cross-view K-NN), depth 해상도·뷰 구조 활용.
- **train_ges_merge**: COLMAP/Blender Scene, **K-NN만** 사용. 그래프는 `build_knn_graph`(최초)와 `knn_from_graph_ring`(재구축). Edge cost의 depth는 **현재 모델 렌더**로만 채움.

이 문서는 `train_ges_merge.py`의 동작을 코드 기준으로 정리한 것이다. Merge 공식·edge cost 식·GES beta 정의는 `ges_merge_training.md`, `graph_merge_edge_collapse.md` 등을 참고하면 된다.

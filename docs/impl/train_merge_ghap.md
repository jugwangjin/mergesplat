# train_merge_ghap.py 구현 정리

COLMAP 씬 + GHAP PLY 로드 후 **LaplacianModel**만 사용해 3DGS 학습·edge-collapse merge를 수행하는 스크립트. 2DGS clamp 없음.

## 진입·의존성

- **진입점**: `ges-splatting/train_merge_ghap.py` → `training(args)`
- **씬**: `sceneLoadTypeCallbacks["Colmap"](source_path, images)` + `cameraList_from_camInfos`
- **모델**: `scene.laplacian_model.LaplacianModel` 만 사용 (GaussianModel 분기 제거됨)
- **렌더**: `gaussian_renderer.render_laplacian` (eval@0, 훈련 스텝, merge process 저장, 최종 eval)
- **Merge 적용**: `train_merge.apply_merge_to_model` (LaplacianModel 7개 param group: xyz, f_dc, f_rest, opacity, shape, scaling, rotation)

## PLY 경로 해석 (resolve_ghap_ply_path)

- 인자가 **.ply 파일**이면 그 경로 그대로 반환.
- 인자가 **디렉터리**면 아래 순서로 첫 번째 존재하는 파일 반환:
  1. `final_pointcloud_after_prune.ply`
  2. `final_pointcloud.ply`
  3. `point_cloud/iteration_35000_after_prune/point_cloud.ply`
  4. `point_cloud/iteration_35000/point_cloud.ply`
- 없으면 `FileNotFoundError`.

## 모델 생성 (create_model_from_ghap_ply)

- **항상 LaplacianModel(sh_degree)** 생성.
- PLY에서 읽는 필드: xyz, opacity, scale_*, rot_*, f_dc_0~2, f_rest_*.
- `_shape`: `nn.Parameter(torch.full((N,1), init_beta, ...))`, `model.shape_strngth = shape_strength`.
- **init_beta**: 함수 인자 기본값 2.0. 호출부에서 `args.init_beta`가 None이면 2.0으로 바꾸고 `fix_beta=True`로 둠 → 3DGS와 동일 동작.

## 훈련 설정 (FakeOpt)

- LaplacianModel.training_setup에 넘기는 FakeOpt: position_lr_*, feature_lr, opacity_lr, scaling_lr, rotation_lr, percent_dense, **prune_shape_threshold**, **shape_strngth**, **shape_lr**.
- `shape_lr = 0` when `fix_beta`, else `args.shape_lr`.

## 그래프

- `build_knn_graph(means, k=knn_k)`, `connect_components(means, edges)` → CPU 텐서로 유지.
- Merge 후 `update_graph_after_merge(edges_gpu, pairs, N_before)`.  
- 주기적 재구축: `graph_rebuild_every_merge_runs`마다 `knn_from_graph_ring` + `connect_components`.

## Edge cost (3DGS)

- **scales_eff** = `(model.get_scaling * model.get_shape).detach()` (effective scale).
- `compute_edge_costs_3dgs(edges, means, prim_features, scales_eff, rotations, opacity)`.
- 내부에서 **질량 보존** (V_Q = V_u + V_v), **bloat_penalty** = vol_Q / (vol_u + vol_v). 자세한 수식은 [graph_merge_edge_collapse.md](graph_merge_edge_collapse.md#3dgs-edge-cost-edge_cost_3dgspy--질량-보존) 참고.

## Merge 단계 (한 번에)

1. **cost** 기준 `greedy_edge_matching` → `pairs`.
2. **eff_scale** = get_scaling.detach() * get_shape.detach().
3. **merged** = `compute_merged_gaussians_3dgs(xyz, eff_scale, rotation, opacity, f_dc, f_rest, pairs)`  
   → xyz, scaling(log), rotation(quat), opacity(logit), f_dc, f_rest.
4. **merged["shape"]** = opacity 가중 평균 (w_u * shape_u + w_v * shape_v).
5. **merged["scaling"]** 보정: effective = exp(scaling)*var_approx(shape) 이므로  
   `merged["scaling"] = merged["scaling"] - log(var_approx(merged["shape"]))`.
6. **apply_merge_to_model(model, pairs, merged)** (train_merge 쪽; LaplacianModel 7그룹 처리).
7. `update_graph_after_merge`, opacity_decay, 필요 시 그래프 재구축.

## 2DGS clamp 없음

- 훈련 루프에서 `model._scaling[:, 2]` 등 세 번째 축 고정하지 않음. 항상 3DGS.

## 저장

- `point_cloud/iteration_0/point_cloud.ply`, `iteration_{N}/point_cloud.ply` (save_iterations + 최종).
- `renders/step*_view*`, `merge_process/iteration_*_view*_rgb.jpg` 등 (render_laplacian 사용).

## CLI 요약

- `--source_path` (COLMAP), `--ghap_result_dir` (디렉터리 또는 .ply), `--model_path`.
- `--init_beta`: None이면 2.0 + fix_beta. `--fix_beta`, `--shape_lr`, `--shape_strength`.
- `--no_merge`, `--knn_k`, `--merge_start_iter`, `--merge_stop_iter`, `--merge_every`, `--merge_dist_l2_thresh`, `--opacity_decay`, `--graph_rebuild_every_merge_runs` 등.

## 실험 스크립트·시각화

- **run_experiments_ghap.sh**: 여러 씬에 대해 **merge만 2 runs** (비교용 no_merge 없음). GHAP 결과는 **beta=2, fixed**로 읽음. **(1) merge_beta2fixed** (`--init_beta 2 --fix_beta`), **(2) merge_beta8** (`--init_beta 8`, shape trainable). 출력: `merge_beta2fixed`, `merge_beta8`. 마지막에 `visualize_experiment_results.py`로 두 run 비교 HTML 생성.
- **visualize_experiment_results.py**: GHAP vs merge 비교 HTML 생성. **공용 GHAP** = `merge_beta2fixed` run의 step0 (beta2 fixed만 step0이 진짜 GHAP). `merge_beta8` run은 step0이 beta8 로드라 GHAP이 아니므로 Step0 열은 "—", **vs GHAP 감소율**만 Final N 기준으로 표시. 렌더는 뷰별로 [GHAP(β2 fix)] [merge_beta2fixed final] [merge_beta8 final] 나란히 표시.
  - 단일 경로: `<model_path>/comparison_step0_vs_final.html`
  - 복수 경로: 공통 부모에 `comparison_step0_vs_final.html`.

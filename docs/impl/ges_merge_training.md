# GES Edge-Collapse Merge Training (train_merge.py)

## 개요

`ges-splatting/train_merge.py`: MapAnything dense init + 그래프 기반 edge-collapse merge로 2DGS(LaplacianModel) 학습.

- **Dense init**: MapAnything depth/normal → 픽셀 단위 3D 역투영으로 초기 Gaussian 생성
- **GES beta (shape)**: generalized Gaussian shape 파라미터. 아래 "GES beta 정의" 참고.
- **Edge-collapse merge**: image-space 4-neighbor + cross-view K-NN 그래프, photometric edge cost로 유사 쌍 병합
- **Optimize–merge 반복**: 학습 → (prune) → merge → 그래프 갱신

## 데이터 해상도 정책

- **캐메라 / GT 렌더링**: 항상 **full-res** 이미지와 K. `images_uint8`, `images_float`, `Ks_np`(이미지 해상도).
- **Create from MapAnything (언프로젝션·그래프)**: **depth/normal 해상도** 유지.  
  - depth/normal/mask: 원본 (H_d, W_d)  
  - RGB: 이미지를 **Lanczos**로 (H_d, W_d)로 다운샘플 → `images_float_at_depth_size`  
  - K: depth 해상도용 `Ks_at_depth_size_np`  
- 의미 없는 포인트 증가를 막기 위해 depth/normal을 이미지 해상도로 업샘플하지 않음.

## 파이프라인

```
[1] MapAnything 데이터 로드
    ├─ depth, normal, mask, K, c2w (원본 해상도)
    ├─ RGB full-res → cameras/GT용
    └─ RGB at depth size (Lanczos) + K at depth size → create_model_from_mapanything 전용
         │
         ▼
[2] Dense Unprojection → LaplacianModel (view_indices=train_indices 시 학습 뷰만)
    ├─ _xyz, _features_dc/rest, _scaling (view-aware tangent + depth discontinuity)
    ├─ _rotation: normal → quat, _opacity, _shape (init_beta)
    └─ 2DGS: _scaling[:,2] = log(1e-6) 고정
         │
         ▼
[3] 그래프 구축 (merge 사용 시)
    ├─ Image-space 4-neighbor (per view) + depth discontinuity 필터
    ├─ Cross-view K-NN (torch cdist + topk)
    └─ connect_components로 연결
         │
         ▼
[4] Optimize–Merge 루프
    ├─ 매 step: render → L1 + SSIM + DoG mask loss (+ optional edge normal align)
    ├─ 매 100 step: low-opacity prune (graph 인덱스 갱신)
    └─ merge_start_iter ~ merge_stop_iter, merge_every마다:
        ├─ Edge cost (depth-res 이미지/K 사용): primitive color·normal, GT rgb·depth·normal, tangent_2d
        ├─ Opacity 기반 cost scale: 낮은 opacity 엣지 → cost 축소 (merge 유리)
        ├─ 6 threshold 통과 + combined cost greedy matching
        ├─ Merge: opacity×volume 가중치, 2D tangent-plane moment matching (scale)
        ├─ Opacity regularization: merge run 후 alpha -= opacity_decay (Revising Densification 3.5)
        └─ update_graph_after_merge (K-NN 재구축 없음)
```

## GES beta (shape) 정의

- **저장**: `LaplacianModel._shape` (학습 가능, (N,1)). 초기값 `init_beta` (기본 10).
- **활성화**: `get_shape = shape_activation(_shape, shape_strngth)` 이고 `shape_activation = var_approx` (utils.general_utils).
- **var_approx(beta, strength)** = `relu(2 * sigmoid(strength * beta))`. strength 기본 0.1이면 beta=10 → 약 1.46.
- **공분산**: `get_covariance`는 `get_scaling * get_shape`를 scaling modifier로 사용. 즉 effective scale = exp(_scaling) × var_approx(_shape, shape_strngth).
- beta가 크면 var_approx가 2에 가까워져 sharp edge (box-like); 작으면 더 부드러운 Gaussian.

## Merge 공식 (GES)

| 속성 | 공식 |
|------|------|
| weight | w = (α×vol) / (α_u×vol_u + α_v×vol_v), vol = s1·s2·s3 |
| position | μ_new = w_u·μ_u + w_v·μ_v |
| normal/quat | n_new = normalize(w_u·n_u + w_v·n_v) → tangent frame → quat |
| scale (3축) | **2D tangent-plane moment matching**: tangent (t1,t2) 위에서 Σ1,Σ2 = P@R@diag(s²)@R'@P', Σ_mix = w_u Σ1 + w_v Σ2 + w_u w_v ΔΔ' (Δ=μ_u−μ_v의 2D 투영), 2×2 해석적 고유값 → s1,s2; normal 축 1e-6 고정 |
| opacity | α_new = α_u + α_v − α_u·α_v |
| shape, SH | w_u·x_u + w_v·x_v |

## Edge Cost (graph_merge/edge_cost.py)

- **입력**: depth 해상도 이미지·K·depth·normal로 프로젝션·샘플링 일치.
- **비용**: primitive color L2, primitive normal (1−cos), GT rgb/depth/normal (multi-view 평균), **normal-dist** \|(μ_u−μ_v)·n_avg\| (world 단위, scale-invariant), **depth** (scale-dependent when scales given), **scale_diff** \|r_u−r_v\|/(r_u+r_v).
- **Normal-dist threshold**: normal 방향 거리 상한 (world 단위). 이보다 크면 merge 안 함 (scale-invariant). 예: 0.02 = 2cm.
- **Scale-diff threshold**: 크기 차이 허용; 0=동일 크기만, 0.5=중간, 1=제한 없음.
- **Opacity scale**: α 작을수록 cost 감소 → 저투명도 쌍 먼저 merge.

## Merge 시 좌표계 (position / camera space)

- **merge 경로에서 camera space 사용 없음**: `compute_merged_ges`는 `model._xyz`(world)만 사용하고, `mu_new = w_u*μ_u + w_v*μ_v` 역시 같은 텐서에서 인덱싱한 값의 가중합이므로 **model과 동일한 좌표계**이다. `apply_merge_to_model`도 `merged["xyz"]`와 `model._xyz.data[keep_t]`만 concat하므로 좌표 변환은 없다.
- **초기 xyz**: `create_model_from_mapanything`에서 `_unproject_pixels(..., c2w)`로 **c2w 기준 world**로 언프로젝트한다. MapAnything pose가 **c2w(camera-to-world)**가 맞아야 하고, **w2c로 들어오면** 전체가 잘못된 공간에 저장되어 merge 후에도 그대로 잘못된 공간으로 보일 수 있다.
- **렌더러**: `means3D = pc.get_xyz`, `viewmatrix = world_view_transform`으로 world → view 변환만 하므로, 저장된 xyz가 world면 정상 동작한다.
- **Merge 후 점이 한곳으로 몰려 보일 때 점검할 것**  
  - (1) MapAnything pose가 c2w인지 확인  
  - (2) merge 직후 `merged["xyz"]`가 원본 두 점 사이(또는 AABB 내)에 있는지 `--debug_merge_position` 등으로 검사  
  - (3) PLY 뷰어/카메라 설정(원점, up 축)으로 인한 시각적 몰림이 아닌지 확인

## Greedy Matching (7 threshold)

primitive_color, rgb, depth, normal, primitive_normal, normal_dist, scale_diff 각각 threshold 이하만 후보 → combined cost 오름차순 → 비중첩 pair 선택.

## 평가·기타

- **Eval**: MapAnything-only면 test_indices 뷰 전부 MapAnything pose로 렌더 → PSNR, SSIM, LPIPS (lpips 패키지 있으면) 저장.
- **merge_process**: 학습 중 50 step마다 2개 학습 뷰 rgb/normal/gaussian_id 저장. 학습 종료 후 2fps GIF (view0/1별 rgb, normal, distribution).
- **run_experiments.sh**: `--skip_completed`로 `point_cloud/iteration_<N>/point_cloud.ply` 있는 run 스킵. `ITERATIONS=N`으로 iteration 수 맞춤.

## 핵심 CLI (일부)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| init_beta | 10.0 | GES shape |
| init_stride | 4 | 픽셀 샘플링 간격 |
| merge_every | 50 | merge 주기 |
| merge_start_iter | 500 | merge 시작 |
| merge_stop_iter | 8000 | merge 종료 |
| merge_prim_color_thresh | 0.09 | primitive color L2 |
| merge_prim_normal_thresh | 0.12 | primitive normal |
| merge_rgb_thresh | 0.08 | GT rgb cost |
| merge_normal_dist_thresh | 0.02 | max normal-direction distance (world units, e.g. 2cm) |
| merge_scale_diff_thresh | 0.5 | scale-diff \|r_u−r_v\|/(r_u+r_v) |
| opacity_decay | 0.001 | merge run 후 alpha 감소 (0=비활성) |
| lambda_edge_normal_align | 0.0 | 유사 엣지(prim rgb+normal)에 대한 inter-Gaussian normal 정렬 (0=비활성) |

## 파일 위치

- **진입점**: `ges-splatting/train_merge.py`
- **그래프/비용/merge**: `ges-splatting/graph_merge/` (graph.py, edge_cost.py, merge.py) — gsplat 의존 없음
- **실험 스크립트**: `ges-splatting/run_experiments.sh` (no_merge, no_merge_ds, abl_prim_color def/lo/hi)

## 참고

- Normal: MapAnything가 global(shared) normal 내보내므로 별도 camera→world 변환 없음.
- COLMAP 데이터: `--dataset_dir`에 sparse/ 있으면 eval 시 COLMAP test_images + Umeyama 사용. Merge 그래프·edge cost는 MapAnything depth/normal 기반이므로 MapAnything 출력 경로 필요.

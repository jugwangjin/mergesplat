# Graph-based Edge Collapse (알고리즘 참조)

## 개요

Dense init된 2DGS에서 **k-NN + image-space 그래프** 상의 edge에 photometric cost를 부여하고, threshold를 만족하는 edge를 **greedy pairing**으로 선택한 뒤 **moment-matching merge**로 한 쌍씩 병합하는 방식.

**현재 구현**:
- **2DGS (MapAnything)**: `ges-splatting/train_merge.py` + `graph_merge/` (graph.py, edge_cost.py / edge_cost_v2.py, merge.py).
- **3DGS (GHAP)**: `ges-splatting/train_merge_ghap.py` + `graph_merge/` (graph.py, **edge_cost_3dgs.py**, merge.py, **merge_3dgs.py**).

전체 파이프라인·CLI·데이터 정책은 [ges_merge_training.md](ges_merge_training.md) 참고.

---

## 3DGS Edge Cost (edge_cost_3dgs.py) — 질량 보존

**파일**: `graph_merge/edge_cost_3dgs.py`. **train_merge_ghap.py**에서만 사용 (LaplacianModel, 3DGS 전용).

### L2² 목표와 “평가용 알파” vs “렌더링 알파”

- 목표: ∫(P_u + P_v − Q)² 최소화. 여기서 알파는 **밀도(질량)** 역할이므로 **1.0을 넘어가도 됨** (렌더링 opacity가 아님).
- **질량 보존**: 병합된 가우시안 Q가 가져야 할 총 질량은 항상 원본 합과 같아야 함.
  - V_Q = V_u + V_v  
  - V = α × (2π)^(3/2) × √det(Σ) (부피×밀도).
- 따라서 평가용 “밀도 알파”는 부피에 맞춰 역산:
  - **a_Q = V_Q / ((2π)^(3/2) × vol_Q)**  
  - 완전 겹침 (vol_Q = vol_u = vol_v): **a_Q = a_u + a_v** (합).  
  - 부피 2배 (vol_Q = 2·vol_u): **a_Q = (a_u + a_v)/2** (평균).

이렇게 하지 않고 a_Q = a_u + a_v 고정 후 V_Q = a_Q × vol_Q 로 두면, 겹침이 작을 때 vol_Q가 커지면서 V_Q가 폭발하고 **완전 겹침 후보가 cost 폭발로 탈락**하는 버그가 발생함. 질량 보존으로 수식 하나로 두 케이스를 모두 처리.

### 구현 요약

- **V_u, V_v**: a_u × (2π)^(3/2) × vol_u 등으로 계산.
- **Sigma_q**: moment matching (w_u×Sigma_u + w_v×Sigma_v + w_u×w_v×outer(μ_u−μ_v)); vol_q = √det(Sigma_q).
- **V_q = V_u + V_v** (평가 시만 사용; 실제 파라미터 업데이트는 merge 단계에서 alpha compositing 등으로 처리).
- **L2² 전개**: d_uu×i_uu + d_vv×i_vv + d_qq×i_qq + 2×d_uv×i_uv − 2×d_uq×i_uq − 2×d_vq×i_vq.  
  - i_ab: 3D Gaussian 내적(unnormalized), d_ab: 1 + λ²×(sh·sh) 색 항.
- **Bloat penalty**: vol_Q / (vol_u + vol_v). 멀리 떨어진 쌍(부피 팽창) 억제.
- **최종 cost**: `cost = l2_sq * bloat_penalty`, `cost.clamp(min=0.0)`.

실제 **merge 적용** 시(예: `merge_3dgs.compute_merged_gaussians_3dgs` → `train_merge.apply_merge_to_model`)에는 α_new = α_u + α_v − α_u·α_v 등 렌더링용 alpha compositing을 쓰거나, 필요 시 1.0 클리핑을 적용하면 됨.

---

## Merge 공식 (요약)

### 2DGS (train_merge.py)

- **Weight**: opacity×volume 기반 `w_u`, `w_v` (같은 비율로 position, normal, scale, SH 등 블렌딩).
- **Opacity**: α_new = α_u + α_v − α_u·α_v.
- **Scale**: merged tangent frame (t1, t2, n) 기준 per-axis **total variance** → s_k = √(w_u(var_u_k + d_u_k²) + w_v(var_v_k + d_v_k²)).  
  d² 항으로 인해 떨어진 두 surfel 병합 시 scale이 자연스럽게 커짐.

### 3DGS (merge_3dgs.py)

- **Position / Covariance**: opacity 비율 w_u = a_u/(a_u+a_v), w_v = a_v/(a_u+a_v)로 moment matching (C_new = w_u C_u + w_v C_v + w_u w_v outer(μ_u−μ_v)); eigendecomposition → scaling(log), rotation(quat).
- **Opacity**: α_new = α_u + α_v − α_u·α_v (렌더링용).
- **SH**: 동일 가중치로 블렌딩. LaplacianModel 사용 시 **shape**는 opacity 가중 평균 후, effective scale 보정으로 merged["scaling"] 역산 (train_merge_ghap에서 var_approx 사용).

## Edge Cost 종류 (2DGS: edge_cost.py / edge_cost_v2)

| 비용 | 설명 |
|------|------|
| primitive color | 두 Gaussian의 SH0→RGB L2 |
| primitive normal | 1 − cos(n_u, n_v) |
| GT rgb/depth/normal | 모든 view에 투영 후 가시인 view에서 샘플링한 GT 차이 평균 |
| normal_dist | \|(μ_u−μ_v)·n_avg\| (world 단위, scale-invariant); normal 방향으로 떨어지면 merge 안 함 |
| scale_diff | \|r_u−r_v\|/(r_u+r_v) (scales 있을 때) |

(선택) **Opacity scale**: α가 작은 엣지는 cost를 줄여 merge 우선.

## Greedy Matching

1. 위 비용들이 각각 threshold 이하인 edge만 후보.
2. Combined cost(합) 오름차순 정렬.
3. 비중첩: 한 node는 최대 한 pair에만 포함. 선택된 edge의 양 끝 node는 “사용됨” 처리.
4. 결과: independent pair 집합 → 각 pair에 대해 moment-matching merge 적용.

## 그래프 구성 (ges-splatting)

- **Image-space 4-neighbor**: view별 픽셀 그리드, depth discontinuity로 일부 edge 제거.
- **Cross-view K-NN**: 3D xyz 기준 K-NN (torch), 같은 view 제외.
- **connect_components**: 연결 성분 연결용 bridge edge 추가.
- Merge 후에는 **update_graph_after_merge**로 인덱스만 재매핑 (K-NN 재구축 없음).

## Densification과 그래프 갱신 (train_ges_merge)

Densification 주기(densification_interval, shape_pruning_interval)와 merge 주기(merge_every)는 다르다. Densification이 clone/split/prune으로 node 수를 바꾸면, 그래프를 버리지 않고 **인덱스만 갱신**해 반영한다.

- **densify_and_prune** (clone + split + opacity/size prune): `return_graph_info=True` 시 `N0`, `clone_parents`, `split_sources`, `N_after_clone`, `keep_mask` 반환 → **update_graph_after_densify**로 엣지 확장·재인덱싱 후 keep_mask 기준 prune.
- **size_prune**: `return_keep_mask=True` 시 유지된 점 마스크 반환 → **update_graph_after_prune**로 엣지 재인덱싱.
- 그래프가 없을 때만(첫 merge_start_iter 등) **build_knn_graph** (scipy cKDTree) + **connect_components**로 초기 구축. 주기적 재구축은 `graph_rebuild_every_merge_runs`로 수행하며, 이때는 **knn_from_graph_ring**(기존 그래프의 3-ring 내에서만 K-NN 재계산) + connect_components를 사용해 faiss/cKDTree 전역 검색 없이 가벼우게 갱신.

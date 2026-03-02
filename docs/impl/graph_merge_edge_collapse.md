# Graph-based Edge Collapse (알고리즘 참조)

## 개요

Dense init된 2DGS에서 **k-NN + image-space 그래프** 상의 edge에 photometric cost를 부여하고, threshold를 만족하는 edge를 **greedy pairing**으로 선택한 뒤 **moment-matching merge**로 한 쌍씩 병합하는 방식.

**현재 구현**: `ges-splatting/train_merge.py` + `ges-splatting/graph_merge/` (graph.py, edge_cost.py, merge.py).  
전체 파이프라인·CLI·데이터 정책은 [ges_merge_training.md](ges_merge_training.md) 참고.

## Merge 공식 (요약)

- **Weight**: opacity×volume 기반 `w_u`, `w_v` (같은 비율로 position, normal, scale, SH 등 블렌딩).
- **Opacity**: α_new = α_u + α_v − α_u·α_v.
- **Scale**: merged tangent frame (t1, t2, n) 기준 per-axis **total variance** → s_k = √(w_u(var_u_k + d_u_k²) + w_v(var_v_k + d_v_k²)).  
  d² 항으로 인해 떨어진 두 surfel 병합 시 scale이 자연스럽게 커짐.

## Edge Cost 종류

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

# Edge/Merge 확장성: Million 단위 3DGS

3DGS는 학습 후 **수십만~수백만** 개 Gaussian이 남는 경우가 많다. 현재 K-NN 그래프 + 전역 cost + 매칭 파이프라인은 이 규모에서 **메모리·시간** 모두 병목이 된다.

---

## 1. 현재 파이프라인에서의 규모

| 단계 | N=100k | N=1M | 병목 |
|------|--------|------|------|
| **build_knn_graph** | E ≈ 500k | E ≈ 5M | cKDTree: CPU, O(N·k·log N). 1M이면 수십 초. PyKeOps 사용 시 GPU로 넘기지만 1M×1M LazyTensor argKmin도 부담. |
| **compute_edge_costs_3dgs** | E=500k | E=5M | 텐서 shape (E, 3, 3) 등 → 5M×9×4×여러 개 ≈ **수 GB**. Cholesky/det 등 5M번 배치 연산. |
| **greedy_edge_matching** | 500k 정렬 | 5M 정렬 | O(E log E) 정렬 + 스캔. 5M은 수 초 단위. |
| **min_cost_max_matching** | NetworkX | NetworkX | 노드 1M, 엣지 5M 그래프에서 max_weight_matching은 **실질적으로 불가**에 가깝다. |

정리하면:

- **그래프 빌드**: N이 크면 CPU cKDTree는 느리고, PyKeOps라도 1M급은 비용이 큼.
- **Cost**: E ∝ N·k 이므로 N=1M, k=10 → E≈5M. **모든 엣지**에 대해 (E,3,3) 공분산·Cholesky·내적을 하면 메모리·연산 모두 한 번에 처리하기 어렵다.
- **매칭**: Greedy는 5M 정렬로 버틸 수 있지만, min-cost maximum matching은 1M 노드에서는 비현실적.

그래서 **“전체 포인트 일괄 K-NN → 전 엣지 cost → 전역 매칭”** 구조는 million 단위에는 그대로 쓰기 어렵다.

---

## 2. 확장 가능한 방향 (개념)

### 2.1 엣지 수를 줄이기

- **k 줄이기**: k=10 → 4~6 등으로 줄여 E를 반 이하로.
- **공간 분할**: 전체가 아니라 **같은/인접 voxel 안의 점끼리만** 엣지를 만든다.  
  예: voxel당 평균 1만 점, k=6 → voxel당 E ≈ 3만. 100개 voxel이면 3M 엣지지만, **voxel별로 나눠 처리**하면 한 번에 올리는 엣지 수는 3만 개로 제한 가능.

### 2.2 Cost를 “전 엣지”가 아니게

- **엣지 캡**: 매 merge 라운드에서 **상위 C개 엣지만** cost 계산 (예: C=200k).  
  예: K-NN으로 후보 엣지는 많이 두되, **거리/볼륨 기반 휴리스틱**으로 후보를 줄인 뒤, 줄인 집합에만 `compute_edge_costs_3dgs` 적용.
- **배치 처리**: 전 엣지를 한 번에 말고, 50k~100k개씩 잘라 cost 계산 → 결과만 모아서 매칭에 사용. 메모리 상한을 고정.

### 2.3 매칭을 “전역”이 아니게

- **지역 매칭**: voxel(또는 그래프 클러스터) 단위로 **독립적으로** greedy matching.  
  전역 최적은 아니지만, 같은 voxel 안 이웃끼리만 merge해도 N을 크게 줄이기엔 충분할 수 있음.
- **Greedy만 사용**: min_cost_max_matching은 N이 크면 끄고, greedy만 쓰면 O(E log E)로 제한.

### 2.4 Merge 횟수/범위 제한

- **라운드당 merge 쌍 상한**: 예: 한 라운드에서 최대 5만 쌍만 merge.  
  Cost 계산도 “가장 merge 유력한” 10만~20만 엣지에만 하고, 그중 상위 5만 쌍만 실제 merge.
- **Merge 구간 조정**: `merge_every`를 크게 해서 merge 라운드 자체를 줄이고, 라운드당 처리량을 위 전략으로 제한.

---

## 3. 권장: Voxel 기반 지역 Merge (요지)

Million 단위에서 **구현 부담 대비 효과**를 고려하면, **voxel 단위 지역 그래프 + 지역 cost + 지역 greedy**가 적당하다.

1. **Voxel 분할**  
   - xyz 기준으로 그리드(예: 32³ 또는 64³) 나눔.  
   - 각 점을 하나의 voxel에 할당 (가능하면 GPU에서 한 번에).

2. **지역 K-NN (또는 지역 엣지)**  
   - **같은 voxel + 인접 voxel** 안의 점만 상대하여, per-voxel로 K-NN 또는 “같은 voxel 내 + 경계만 인접 voxel과” 엣지 생성.  
   - 전역 N에 대한 K-NN이 아니라 **voxel 내부 + 경계**만 보므로, 한 번에 올리는 점 수 ≈ voxel 크기 (예: 1M이면 64³일 때 voxel당 수백~수천 점).

3. **Voxel별 cost + greedy**  
   - voxel(또는 “voxel 쌍”)마다 `compute_edge_costs_3dgs`를 **그 voxel에 속한 엣지만** 넣어 호출.  
   - 같은 voxel 단위로 `greedy_edge_matching` 적용 → merge 쌍 결정.  
   - 메모리: 동시에 처리하는 엣지 수 = (voxel당 점 수)² 또는 k×(voxel당 점 수) 수준으로 제한.

4. **Merge 적용**  
   - 기존과 동일하게 `compute_merged_gaussians_3dgs` + `replace_after_merge`.  
   - 한 라운드에서 여러 voxel에서 나온 merge 쌍을 **한 번에** 적용할 때는, 서로 다른 voxel에서 나온 쌍끼리는 겹치지 않도록 설계 (한 점이 두 voxel에 동시에 merge되면 안 됨).  
   - 예: “점 → 소속 voxel”이 유일하면, voxel별로 독립 매칭해도 한 점은 최대 한 쌍에만 속함.

이렇게 하면:

- **그래프**: 전역 5M 엣지 대신, voxel당 수천~수만 엣지만 유지.
- **Cost**: 동시에 (E_local, 3, 3) 수준만 올리면 되므로 메모리 고정.
- **매칭**: voxel별 greedy만 해도 됨.

---

## 4. 체크리스트 (구현 시)

- [ ] **N 또는 E 임계값**  
  - N &gt; N_max 또는 E &gt; E_max 이면 “voxel 모드” 또는 “엣지 캡 모드”로 전환하는 분기.
- [ ] **Voxel 해상도**  
  - 32³ ~ 64³ 정도로 시작해, voxel당 점 수가 너무 많으면 해상도 올리기.
- [ ] **경계 처리**  
  - 인접 voxel과의 엣지를 넣을지, “같은 voxel만”으로 할지 결정. (같은 voxel만이면 구현 단순, 품질은 약간 떨어질 수 있음.)
- [ ] **Cost–Merge 정렬**  
  - voxel/캡 모드에서도 `edge_cost_3dgs`의 질량 정의는 기존과 동일하게 두어 **merge_3dgs**와 맞출 것.
- [ ] **update_graph_after_merge**  
  - voxel 모드는 “전역 그래프”를 유지하지 않고 라운드마다 voxel 기반으로만 엣지를 만들 수 있으므로, 그래프 갱신 규칙을 문서(예: edge_merge_update_checklist.md)에 반영.

---

## 5. 요약

- **현재 방식**: N=1M, k=10 → E≈5M. 전 엣지 cost + 전역 매칭은 **메모리·시간 모두 million 단위에 맞지 않음**.
- **대응**:  
  - **단기**: k 축소, merge 라운드당 쌍 상한, **greedy만** 사용, cost는 배치로 나눠 계산.  
  - **중기**: **voxel 기반 지역 그래프 + 지역 cost + 지역 greedy**로 전역 엣지 수와 동시 처리 엣지 수를 제한.

---

## 6. 구현 위치 (voxel 경로)

- **그래프**: `GHAP/graph_merge/graph_voxel.py`
  - `point_to_voxel_linear(means, grid_res)` — 점별 voxel 선형 id
  - `adjacent_voxel_linear_ids(vi, vj, vk, grid_res)` — 26 인접 voxel (3×3×3 − self)
  - `build_voxel_knn_edges(means, k, grid_res)` → `(edges, edge_voxel)` (voxel + 인접 voxel 내에서만 K-NN)
  - `connect_components_voxel(means, edges, edge_voxel, grid_res)` → 연결 성분 브릿지 추가 후 `(edges, edge_voxel)`
- **Cost 배치**: `GHAP/graph_merge/edge_cost_3dgs.py` — `compute_edge_costs_3dgs_voxel_batched(edges, edge_voxel, ..., batch_size_max=..., **kwargs)`
- **train_merge 옵션**:
  - `--voxel_merge`: 항상 voxel 경로 사용
  - `--voxel_merge_threshold 100000`: N ≥ 10만이면 자동으로 voxel 경로 (기본 100k)
  - `--voxel_grid_res 32`: 그리드 해상도 (기본 32 → 32³ voxel)
  - `--voxel_cost_batch_max 200000`: voxel당 cost 배치 상한

이 문서는 `docs/impl/` 아래 두었으며, 옵션/파일 변경 시 함께 갱신하면 된다.

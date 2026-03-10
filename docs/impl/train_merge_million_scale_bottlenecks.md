# Merge 파이프라인 병목: Edge cost → 매칭 → Graph update

렌더링은 제외. **Edge cost 계산 → merge 대상 추출(매칭) → merge 후 graph update** 구간만 정리.  
N ≈ 100만, voxel 경로 사용 가정.

---

## 1. 파이프라인 세 구간

```
[엣지 비용 계산] → [merge 대상 추출 (매칭)] → [merge 수행 + graph update]
compute_edge_costs_3dgs_voxel_batched   greedy_edge_matching / min_cost_max_matching   update_graph_after_merge + edge_voxel 갱신
```

---

## 2. 구간별 병목

### 2.1 Edge cost 계산

- **위치**: `compute_edge_costs_3dgs_voxel_batched(edges, edge_voxel, xyz, prim_features, ...)`  
  → 내부에서 voxel별로 `compute_edge_costs_3dgs(chunk, ...)` 반복.
- **특징**:
  - E ≈ k·N (예: k=10 → 수백만 엣지). voxel별 청크(예: `voxel_cost_batch_max=200_000`)로 나눠서 GPU에서 비용 계산.
  - voxel 수만큼 for 루프 + 청크 수만큼 GPU 커널. 메모리는 제어되지만 **연산량은 O(E)**.
- **예상**: merge 한 번당 **수 초 ~ 수십 초**. E가 500만이면 20만×25 청크 → 25번 비용 커널 + 인덱싱.
- **개선 방향**:
  - 청크 크기 조정(`--voxel_cost_batch_max`)으로 메모리/속도 트레이드오프.
  - 비용 커널 자체를 더 큰 배치로 한 번에 돌릴 수 있으면 (메모리 허용 시) 루프/런치 수 감소.

---

### 2.2 Merge 대상 추출 (매칭)

- **위치**: `greedy_edge_matching(edges_gpu, cost_3d, ...)` 또는 `min_cost_max_matching(...)`.
- **특징**:
  - **Greedy**: `valid = (cost_3d < thresh) & ...` → `ve = edges[valid]`, `order = combined.argsort()`, 그 다음 **Python for 루프**로 `for (u, v) in edges_np:` 돌면서 `used.add(u); used.add(v)` 로 비중복 쌍 선택.  
    threshold 통과 엣지가 많으면 **수백만 행**을 Python에서 순회.
  - **Min-cost max matching** (networkx): 그래프가 크면 maximum_weight_matching이 **수십 초~수 분** 걸릴 수 있음.
- **예상**: E가 크고 threshold가 느슨하면 **수십 초 ~ 수 분**. Greedy는 Python 바운드, networkx는 C이지만 그래프 크기에 민감.
- **개선 방향**:
  - Greedy를 PyTorch/CUDA로: `valid` 마스크 적용 후 정렬, 그 다음 “한 번에” 또는 청크 단위로 비중복 매칭 (예: argsort 후 스캔하면서 used 플래그 텐서로 체크).
  - 엣지 수 줄이기: 비용 상위 일부만 매칭 후보로 쓰기 (예: cost 상위 2E/k 개만 남기기). 품질 트레이드오프 필요.
  - networkx 대신 더 가벼운 매칭 구현(예: greedy만 GPU로 빠르게)으로 대체.

---

### 2.3 Merge 후 graph update

- **위치**:
  1. `update_graph_after_merge(edges_gpu, pairs, N_before)`  
     → `old_to_new` 매핑, `new_e = old_to_new[e[:,0]], old_to_new[e[:,1]]`, self-loop 제거, **`_unique_undirected_edges`** 호출.
  2. `_unique_undirected_edges`: `key = u*N + v`, **`torch.unique(key)`** 로 중복 제거. E가 수백만이면 이 연산이 **수 초**.
  3. (voxel 경로) `point_to_voxel_linear(gaussians.get_xyz)` 로 **전체 xyz**를 CPU로 복사 후 NumPy로 voxel id 계산 → `edge_voxel` 갱신. N=100만이면 **수 초**.
- **예상**: **수 초 ~ 수십 초** (E와 N 모두에 비례).
- **개선 방향**:
  - `torch.unique`: 이미 GPU 사용. 대안으로 정렬 기반 unique (예: key 정렬 후 diff≠0 위치만 유지)가 더 빠를 수 있음 (구현 난이도 있음).
  - `edge_voxel` 갱신: merge로 **변경된 노드**만 반영하는 증분 갱신. “kept 노드 + 새로 생긴 merged 노드”만 voxel 재계산하고, 기존 edge_voxel은 old_to_new로 인덱스만 바꾸면 됨. 단, edge 리스트가 바뀌므로 “어떤 edge가 어떤 voxel”인지는 새로 매겨야 함.  
    → **간단한 절충**: merge 직후에는 `edge_voxel`을 “기존 edge의 new endpoint 인덱스로 point_to_voxel 한 번만” 호출하는 식으로, 최소한 전체 N을 두 번 넘나들지 않게 하기.

---

## 3. 요약 표 (N ≈ 1M, E ≈ 수백만)

| 구간 | 현재 구현 | 예상 병목 | 개선 방향 |
|------|-----------|-----------|-----------|
| **Edge cost** | voxel-batched, GPU 청크 반복 | merge당 수 초~수십 초 | 배치 확대(메모리 허용 시), 커널 최적화 |
| **매칭** | Greedy Python 루프 / networkx | 수십 초~수 분 | GPU greedy, 또는 후보 엣지 수 축소 |
| **Graph update** | torch.unique(key) + 전체 xyz→CPU로 edge_voxel | 수 초~수십 초 | unique 대체(정렬 unique), edge_voxel 증분/최소 갱신 |

---

## 4. 결론

- **문제 구간**: edge cost 계산 ~ merge 대상 추출 ~ merge 후 graph update.  
  렌더링은 3DGS 학습에서 다루는 범위이므로 여기서는 제외.
- **가장 민감한 부분**:  
  - **매칭**: Python for 루프 또는 networkx가 E가 클 때 가장 쉽게 수 분 단위로 늘어남.  
  - **Graph update**: `torch.unique` + 전체 `point_to_voxel_linear`가 E·N 크기에 비례해 무거움.
- **우선 개선 후보**:  
  1) Greedy 매칭을 GPU/벡터화로 이전,  
  2) merge 후 edge_voxel 갱신을 전체 N 재계산이 아닌 최소/증분 방식으로 변경.

---

## 5. 구체적 개선 방법 제안

### 5.1 매칭: 후보 엣지 상한 (merge_candidate_cap)

- **의도**: 비용 순 정렬 후 상위 `cap`개만 greedy에 넣어 Python 루프 길이 제한.
- **구현**: `--merge_candidate_cap 0`(기본=무제한), 0이 아니면 `ve = ve[:cap]` 후 동일 greedy.
- **위치**: `train_merge.py`에서 `match_fn` 호출 전에 `edges_gpu`/`cost_3d` 등을 정렬해 두고, `cap > 0`이면 상위 cap개만 전달. 단, `greedy_edge_matching`은 7개 cost를 모두 받으므로 내부에서 정렬 후 cap 적용하거나, 호출 전에 `valid` 마스크 적용된 엣지·비용을 cap개로 자른 뒤 전달.
- **품질**: cap이 충분히 크면(예: 50만~100만) 품질 손실 적음. E가 500만이면 100만만 봐도 merge 수는 거의 포화.

### 5.2 매칭: Chunked greedy (GPU used 플래그)

- **의도**: `used`를 GPU 텐서로 두고, 엣지를 청크 단위로 처리. 청크마다 `valid = ~used[u] & ~used[v]`를 GPU에서 계산해, **valid인 엣지만** CPU로 가져와 짧은 Python 루프로 선택 후 `used` 갱신.
- **효과**: (1) 전체 E를 한 번에 CPU로 안 보냄. (2) 이미 끝난 노드가 많으면 valid 비율이 낮아져 Python이 도는 횟수 감소.
- **구현**: `greedy_edge_matching_chunked(edges, combined_cost, thresh, chunk_size=65536)` 형태. 정렬 후 for offset in range(0, E, chunk_size); chunk = ve[offset:offset+chunk], used_u = used[chunk[:,0]], used_v = used[chunk[:,1]]; valid = ~used_u & ~used_v; valid_idx = torch.where(valid)[0]; 선택은 이 인덱스만 CPU로 가져와 순서대로 greedy (한 엣지 선택 시 used 업데이트). `used`는 GPU 1D bool 텐서.

### 5.3 Edge cost: voxel-batched를 “정렬 후 연속 블록”으로

- **의도**: `edge_voxel`별로 mask를 E번 도는 대신, `edges`를 `edge_voxel`로 정렬한 뒤 같은 voxel이 연속된 구간만큼 잘라서 비용 계산.
- **구현**: `compute_edge_costs_3dgs_voxel_batched` 내부: `order = torch.argsort(edge_voxel)`; `edges_sorted = edges[order]`, `edge_voxel_sorted = edge_voxel[order]`. 그 다음 `edge_voxel_sorted`에서 값이 바뀌는 위치만 찾아 구간 [start, end) 단위로 청크(또는 batch_size_max 이하로 나눠) `compute_edge_costs_3dgs` 호출. 결과를 `cost_out[order]`에 역산.
- **효과**: voxel 수만큼 `(edge_voxel == v)` 스캔 제거, 캐시 친화적.

### 5.4 Graph update: sort 기반 unique (torch.unique 대체)

- **의도**: `torch.unique(key)`가 E가 클 때 무거우면, key 정렬 후 인접 diff로 “구간의 첫 원소”만 남기는 방식.
- **구현**: `key = u*N + v`; `idx = torch.argsort(key)`; `key_s = key[idx]`, `u_s = u[idx]`, `v_s = v[idx]`; `first = torch.cat([torch.ones(1, dtype=torch.bool, device=key.device), key_s[1:] != key_s[:-1]])`; `u_out = u_s[first]`, `v_out = v_s[first]`. 반환은 기존과 동일 (numpy stack 또는 tensor).
- **효과**: 정렬+선형 스캔만으로 unique. GPU에서 `torch.unique`보다 빠른 경우 많음.

### 5.5 edge_voxel 갱신: 엣지 endpoint만 voxel 계산

- **의도**: merge 직후 `point_to_voxel_linear(means)`를 전체 N에 대해 하지 말고, **새 edge 리스트의 endpoint로 등장하는 노드**만 voxel 부여.
- **구현**: `edges`(merge 후 새 인덱스)에 대해 `unique_nodes = torch.unique(edges.ravel())`. 전체 `xyz`의 min/max로 grid bbox는 동일하게 두고, `xyz[unique_nodes]`만 CPU로 보내 해당 인덱스에 대한 voxel id 계산 → `point_voxel_subset[node_id] = voxel_id`. 그 다음 `edge_voxel = point_voxel_subset[edges.min(dim=1)[0]]`는 불가(인덱스가 0..N-1가 아니라 unique_nodes의 값). 따라서 `node_to_voxel` 딕셔너리 또는 `voxel_of_node = full N 배열`에 unique_nodes 위치만 채우기. 즉 `voxel_of_node[unique_nodes] = point_to_voxel_linear_subset(means, unique_nodes, grid_res)`. 그러면 `edge_voxel = voxel_of_node[edges.min(1)[0]]`.  
  **point_to_voxel_linear_subset**: `means[indices]`의 xyz만 사용, 하지만 grid의 lo/hi는 `means` 전체로 한 번 계산 (또는 인자로 받음). 반환은 `(len(indices),)` voxel id.
- **효과**: N=100만, E=500만이어도 unique endpoint는 보통 2E 이하이고 중복 제거하면 더 적음. CPU 전송·연산을 N → 약 2E 또는 그 이하로 축소.

### 5.6 (선택) 매칭: 비용 상위만 비용 계산

- **의도**: 모든 엣지에 비용을 다 구하지 말고, “후보”를 먼저 줄인 뒤 비용 계산. 예: voxel 내에서 거리/scale 기반으로 상위 2k개만 남기고 비용 계산. 품질 트레이드오프 있음.
- **구현**: 그래프 구축 단계에서 엣지를 더 적게 두거나, 비용 계산 전에 엣지를 필터(예: L2 거리 < τ)해 E를 줄인 뒤 비용 계산. 현재는 이미 K-NN으로 E ≈ k·N이므로, 추가로 “비용 계산할 엣지 샘플”을 두는 방식은 설계 변경이 큼.

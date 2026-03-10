# Merge 알고리즘 개선 여지

현재 파이프라인: **k-NN 그래프 → 3DGS L2² edge cost (bloat/size-mismatch) → 단일 threshold 이하만 후보 → cost 합 오름차순 greedy 비중첩 매칭 → moment-matching merge → 그래프 갱신(재인덱스, 주기적 k-ring 재구성)**.

아래는 개선할 만한 지점 정리.

---

## 1. 매칭: Greedy → (선택) 최대 매칭 / min-cost 매칭

**현재**: `merge.greedy_edge_matching`: cost 합 오름차순 정렬 후, 앞에서부터 “양 끝이 아직 사용 안 됨”이면 선택. 한 번 선택된 노드는 다른 pair에 쓰이지 않음.

**한계**: greedy는 전역적으로 최적이 아님. 비싼 edge 하나를 골라서 두 노드를 묶으면, 그 두 노드와 연결된 더 좋은(저렴한) pair들을 영구히 놓칠 수 있음.

**개선**:
- **Maximum cardinality matching**: threshold 이하인 edge만 남긴 부분 그래프에서, “선택 edge 수”를 최대화하는 매칭 (Blossom, `networkx.max_weight_matching` 등). 동일 개수라면 cost 합 최소인 매칭을 구하려면 **min-cost maximum matching** (예: edge weight = -cost, max weight matching).
- 구현 부담: CPU 그래프 + scipy/networkx 또는 C++ Blossom; merge 빈도가 높지 않으면 비용 대비 이득이 있을 수 있음.

---

## 2. Eigh 연산: CPU 배치 → GPU 배치 + fallback

**현재**: `merge_3dgs.compute_merged_gaussians_3dgs`에서 합쳐진 공분산 `C_new`에 대해 **eigh를 CPU에서 배치로** 수행 (주석: cusolver batched eigh의 INVALID_VALUE/NaN 회피).

**한계**: merge할 pair 수가 많을 때 CPU↔GPU 복사 + CPU eigh가 병목이 될 수 있음.

**개선**:
- GPU `torch.linalg.eigh` (또는 batched cusolver) 사용 시도; NaN/비정상 입력만 감지해서 해당 pair만 CPU로 fallback.
- 또는 merge 배치 크기 제한(cap)을 두어 한 번에 처리하는 pair 수를 줄임.

---

## 3. 그래프 재구성: k-ring만 → (선택) 주기적 전체 k-NN

**현재**: merge 후 `update_graph_after_merge`로 엣지만 재인덱싱. `graph_rebuild_every_merge_runs`마다 `knn_from_graph_ring(edges, xyz, k, ring=3)`으로 **기존 그래프의 3-ring 이웃 안에서만** 다시 k-NN을 구함. 전역 k-NN은 초기 1회만.

**한계**: merge가 많이 되면 연결 성분이 갈라지거나, ring 밖에 있는 “좋은 merge 후보”가 그래프에 아예 없을 수 있음.

**개선**:
- 주기적으로(예: rebuild 횟수가 5의 배수일 때) **전체 포인트에 대해 k-NN을 처음부터 다시 구하고** `connect_components`로 브릿지 추가. 비용은 N이 클수록 커지지만, merge 구간이 고정이라 자주 하지 않으면 부담 제한 가능.

---

## 4. Sibling / 방금 split된 쌍 merge 제외 (미구현)

**현재**: `train_merge`에서는 **densification/split 이력**을 추적하지 않음. 그래프의 모든 edge에 대해 cost만 보고 merge 후보를 뽑음.

**문서 상 아이디어** ([graph_merge_edge_collapse.md](graph_merge_edge_collapse.md), [train_merge_colmap_gsplat.md](train_merge_colmap_gsplat.md)): “같은 부모에서 나온 쌍(sibling)” 또는 “방금 densification된 가우시안”은 일정 iteration 동안 merge 후보에서 제외해, split으로 얻은 표현력을 잠시 유지.

**개선**:
- `GaussianModel`에 `parent_index` 또는 `created_at_iteration` 같은 메타데이터를 두고, `filter_edges_*` 단계에서 “같은 부모” 또는 “최근 N iter 이내 생성” 엣지를 제거. 단, GHAP/mini-splatting의 init PLY에는 부모 정보가 없으므로, **train_merge 구간에서 발생한 split/clone만** 추적해야 함 (densify_and_prune 호출 여부와 연동).

---

## 5. Merge 직후 짧은 fine-tune

**현재**: merge한 뒤 바로 다음 iteration부터 일반 학습(렌더 loss + 기존 prune 등)만 수행. “방금 합쳐진 가우시안”만 따로 몇 step 돌리는 구간은 없음.

**개선**:
- merge가 일어난 iteration 직후, 예를 들어 50~100 step은 **merge된 노드만** 또는 **merge로 변한 점 + 그 이웃**만 업데이트하도록 마스크를 두어 국소 fine-tune. 구현 복잡도와 품질 이득을 실험으로 확인 필요.

---

## 6. Cost / threshold

**현재**:  
- 단일 스칼라 `merge_dist_l2_thresh`로 L2² (volume-normalized + bloat + size-mismatch) cost를 자르고,  
- `greedy_edge_matching`은 2DGS용으로 7종 cost·threshold 인자를 받지만, 3DGS 경로(`train_merge.py`)에서는 **동일한 t를 7번 전달**해 단일 threshold만 사용. API와 실제 사용이 어긋나 있으므로, 3DGS 전용으로는 단일 cost·단일 threshold 호출로 정리하거나 doc/주석으로 명시하는 것이 좋음.

**개선**:
- **호출 단순화**: 3DGS에서 `greedy_edge_matching(edges, cost_3d, ...)` 대신 단일 cost·단일 thresh 래퍼를 두어 의도가 드러나게 함.
- **Iteration/스테이지별 스케줄**: 학습 초반에는 t를 작게, 후반에만 조금 완화하는 등 (예: warmup 비슷한 스케줄).  
- **Per-scene 또는 per-region threshold**: 동적 범위가 큰 씬에서는 지역별로 다른 t를 쓰는 것은 구현·튜닝 부담이 있음.

---

## 7. 그래프 degree 관리 (선택)

**현재**: merge된 노드는 “u의 이웃 ∪ v의 이웃 \ {u,v}”로 이웃이 합쳐져서 degree가 커질 수 있음. `knn_from_graph_ring` 재구성 시 각 노드당 최대 k개만 남기므로 주기적으로 정리됨.

**개선**: merge 직후 `update_graph_after_merge` 단계에서, **merged 노드의 엣지만** “거리 기준 최근접 k개”로 잘라서 degree 상한을 두면, 재구성 전에도 그래프 크기를 더 안정적으로 유지할 수 있음. (현재도 rebuild로 상한이 있으므로 우선순위는 낮음.)

---

## 요약 표

| 항목 | 현재 | 개선 방향 | 난이도 |
|------|------|-----------|--------|
| 매칭 | cost 오름차순 greedy | Min-cost max matching (Blossom 등) | 중 |
| Eigh | CPU 배치 | GPU + NaN fallback 또는 배치 cap | 하 |
| 그래프 | k-ring 재구성만 | 주기적 전체 k-NN | 하 |
| Sibling/신규 점 보호 | 없음 | parent/created_iter 필터 | 중 |
| Merge 후 fine-tune | 없음 | merge된 점만 N step | 중 |
| Threshold | 단일 고정 | iteration 스케줄 | 하 |

원하면 위 항목 중 하나를 골라서 구체적인 패치 포인트(함수/라인)와 API 변경안까지 적어 줄 수 있음.

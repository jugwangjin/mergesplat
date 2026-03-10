# Edge / Merge 관련 코드 업데이트 체크리스트

edge·cost·merge 알고리즘 또는 그래프 로직을 수정할 때, **함께 확인·수정해야 할 파일·위치**를 정리한 문서. 다른 채팅에서도 이 문서를 참고해 일관되게 반영할 수 있다.

---

## 1. 관련 파일 위치 (프로젝트별)

| 역할 | GHAP | gaussian-splatting (vanilla) | mini-splatting |
|------|------|-----------------------------|-----------------|
| **트레이닝 진입점** | `GHAP/train_merge.py` | `gaussian-splatting/train_merge.py` | `mini-splatting/train_merge.py` |
| **그래프** | `GHAP/graph_merge/graph.py` | `gaussian-splatting/graph_merge/graph.py` | (GHAP 또는 자체 graph_merge 참조) |
| **엣지 비용 (3DGS)** | `GHAP/graph_merge/edge_cost_3dgs.py` | `gaussian-splatting/graph_merge/edge_cost_3dgs.py` | `mini-splatting/graph_merge/edge_cost_3dgs.py` 등 |
| **매칭·merge 유틸** | `GHAP/graph_merge/merge.py` | `gaussian-splatting/graph_merge/merge.py` | 동일 |
| **3DGS merge 연산** | `GHAP/graph_merge/merge_3dgs.py` | `gaussian-splatting/graph_merge/merge_3dgs.py` | 동일 |
| **GaussianModel (merge 반영)** | `GHAP/scene/gaussian_model.py` (`replace_after_merge`) | `gaussian-splatting/scene/gaussian_model.py` (`replace_after_merge`) | 해당 씬 모델 |

- **graph_merge** 모듈은 GHAP·vanilla·mini 간에 **로직을 맞춰 둔 복사본**일 수 있으므로, 한 쪽에서 cost/merge 공식·그래프 규칙을 바꾸면 **다른 쪽 graph_merge + 해당 train_merge.py**도 동일하게 반영해야 한다.

---

## 2. Cost ↔ Merge 정렬

- **edge_cost_3dgs.py**에서 쓰는 **질량 정의**(예: `V_q = (V_u + V_v) / 2`)는 **merge_3dgs.py**의 opacity 공식  
  `α_new = (α_u·vol_u + α_v·vol_v) / (2·vol_new)`  
  와 **반드시 일치**시켜야 한다.  
  - Cost에서 `V_q`를 바꾸면 → merge에서 `α_new` 식이 그에 맞는지 확인.  
  - merge에서 opacity/질량 공식을 바꾸면 → cost의 `V_q` 및 관련 docstring·주석을 같이 수정.

---

## 3. Merge 결과 → 모델 반영 규칙

- **merge_3dgs**의 출력 키: `xyz`, `scaling`(log), `rotation`(quat 4), `opacity`(logit), `f_dc`, `f_rest`.  
- **replace_after_merge**에 넘기는 `new_tensors_dict`의 키는 **optimizer param_groups의 `"name"`**과 동일해야 함:  
  `xyz`, `f_dc`, `f_rest`, `opacity`, `scaling`, `rotation`.  
- **노드 순서 규칙** (graph와 공유):  
  `[기존에서 유지된 노드(원래 인덱스 오름차순)] + [merge된 노드 1개 per pair]`.  
  - **update_graph_after_merge**의 `old_to_new` 매핑과 **replace_after_merge**에서의 `kept_old_indices` + merged 블록 순서가 이 규칙과 같아야 함.  
- merge 공식이나 출력 키를 바꾸면 →  
  - **merge_3dgs.py** 반환 dict,  
  - **train_merge.py**의 `new_tensors` 구성,  
  - **GaussianModel.replace_after_merge**에서 기대하는 키와 shape  
  를 한꺼번에 점검.

---

## 4. Cost·Matching 인자

- **train_merge.py**에서 `compute_edge_costs_3dgs`에 넘기는 인자와,  
  `greedy_edge_matching` / `min_cost_max_matching`에 넘기는 **threshold**는 **하나의 cost 벡터**에 맞춰져 있다.  
  - cost 식을 바꾸거나 항을 추가/제거하면 →  
    - **edge_cost_3dgs**의 반환/의미,  
    - **train_merge**의 cost 호출부,  
    - **match_fn**에 넘기는 cost 개수·threshold 개수·순서  
    를 같이 수정.  
- 현재 3DGS는 **단일 cost**만 쓰므로 `cost_3d`를 7번 복제해 threshold `t` 7개와 맞추는 형태.  
  cost 종류를 늘리면 match 함수 시그니처·threshold 인자도 늘려야 함.

---

## 5. 그래프 갱신

- **Prune 후**: `train_merge.py`에서 `prune_points` 호출 뒤 **edges**를 `keep` 마스크로 재인덱싱 (update_graph_after_prune와 동일한 로직 또는 해당 함수 사용).  
- **Merge 후**: **update_graph_after_merge**(edges, pairs, N_before) 호출 필수.  
- **Rebuild**: `graph_rebuild_every_merge_runs`마다 **knn_from_graph_ring** + **connect_components** 호출.  
- `build_knn_graph` / `knn_from_graph_ring` / `connect_components`의 시그니처나 반환 형식을 바꾸면 → 이들을 호출하는 **train_merge.py**와, 필요 시 **update_graph_after_merge**의 인덱스 범위 가정을 함께 수정.

---

## 6. GaussianModel 쪽

- **replace_after_merge** 구현이 있는 모든 프로젝트(GHAP, gaussian-splatting, mini-splatting 등)에서:  
  - optimizer **param_groups**의 `name`과 **new_tensors_dict** 키가 일치하는지,  
  - **xyz_gradient_accum**, **denom**, **max_radii2D**, **tmp_radii**를 kept + merged 크기에 맞게 concat하는지  
  확인.  
- **prune_points**에서 제거하는 필드가 바뀌면, **replace_after_merge**에서도 같은 필드를 동일한 인덱스 규칙으로 갱신하는지 확인.

---

## 7. 빠른 체크리스트 (수정 시 한 번씩 확인)

- [ ] **merge_3dgs.py** opacity/질량 공식 수정 시 → **edge_cost_3dgs.py**의 `V_q` 및 docstring 반영  
- [ ] **edge_cost_3dgs.py** cost 식/인자 변경 시 → **train_merge.py**의 `compute_edge_costs_3dgs` 호출 인자 및 match threshold 개수/순서  
- [ ] **merge.py** (greedy/min_cost_max_matching) 시그니처 변경 시 → **train_merge.py**의 `match_fn` 호출부  
- [ ] **merge_3dgs.py** 반환 dict 키 변경 시 → **train_merge.py**의 `new_tensors` 구성 및 **replace_after_merge** 기대 키  
- [ ] **graph.py** (build_knn_graph, update_graph_after_merge, knn_from_graph_ring 등) 시그니처/규칙 변경 시 → **train_merge.py**의 그래프 빌드·갱신·rebuild 호출  
- [ ] **replace_after_merge**가 사용하는 optimizer state 키(예: exp_avg, exp_avg_sq, step)를 바꾼 경우 → 해당 프로젝트의 **GaussianModel**만이 아니라, **다른 복사본**(vanilla, mini 등)에도 동일 적용  
- [ ] 다른 저장소/브랜치에 **graph_merge** 복사본이 있으면 → 동일한 cost·merge·그래프 규칙으로 맞춰 둘 것  

- **확장성**: N이 수십만~수백만일 때는 현재 전역 K-NN + 전 엣지 cost + 전역 매칭이 메모리·시간 모두 병목이다. **Voxel 기반 지역 merge**, **엣지 캡**, **cost 배치 처리** 등은 [edge_merge_scalability.md](edge_merge_scalability.md) 참고.

이 문서는 `docs/impl/` 아래 구현 관리용이며, edge/merge 관련 수정 시 함께 갱신하는 것을 권장한다.

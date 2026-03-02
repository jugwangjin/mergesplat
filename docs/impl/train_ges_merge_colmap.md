# train_ges_merge: COLMAP + densify / prune / merge

> **참고**: 현재 레포의 메인 merge 트레이너는 **train_merge.py** (MapAnything dense init).  
> COLMAP 데이터셋을 쓰더라도 eval 시 `--dataset_dir`에 sparse/가 있으면 COLMAP test_images + Umeyama를 사용하며, merge·그래프는 MapAnything 출력 기반.  
> 아래 내용은 **SfM sparse init + merge** 파이프라인(train_ges_merge.py)이 있을 경우의 참조용.

## 목적

- **train_ges.py** 기반으로 COLMAP 데이터에 대해 2DGS(LaplacianModel) 학습.
- **Merge**를 densification·pruning과 함께 써서 # GS를 줄이기.
- 그래프는 **densification → pruning → merging** 세 가지 연산 모두에 맞춰 유지·갱신.

## 구성

- **스크립트**: `ges-splatting/train_ges_merge.py`
- **그래프**: 3D K-NN만 사용 (COLMAP에는 view_ids/pixel 없음). `gsplat/graph_merge/graph.py`의 `build_knn_graph`, `update_graph_after_merge` 사용.
- **Edge cost**: 학습 뷰의 이미지 + COLMAP에는 GT depth/normal이 없어 dummy(상수 depth, up normal) 사용. Primitive color/normal + RGB 비용만 의미 있음.

## 그래프 갱신 규칙

| 연산 | 시점 | 갱신 방식 |
|------|------|-----------|
| **초기** | merge 사용 시 학습 시작 전 | SfM 초기점 기준 **한 번만** `build_knn_graph(xyz, k)` |
| **Densification + prune** | `densification_interval`마다 `densify_and_prune` 후 | **증분**: clone된 점은 부모와 동일 connectivity + (부모, clone) 엣지 추가; split된 점은 부모 제거 후 자식 2개에 부모 이웃 + 형제 엣지 추가; 이후 opacity prune에 대해 `update_graph_after_prune`. (`update_graph_after_densify`) |
| **Size prune** | `shape_pruning_interval`마다 `size_prune` 후 | `update_graph_after_prune(edges, keep_mask)` 로 **재인덱싱만** (K-NN 재구성 없음) |
| **Merge** | `merge_every`마다 (merge_start_iter ~ merge_stop_iter) | `update_graph_after_merge(edges, pairs, N_before)` 로 **인덱스만 재매핑** |

- Densification마다 K-NN 전체 재구성은 하지 않음 (6M 등 대규모에서 비현실적). 초기 1회 + clone/split 시 부모 기준 연결만 추가.

## 사용 예

```bash
cd ges-splatting

# Merge 없이 (기존 train_ges와 동일)
python train_ges_merge.py -s <source_path> -m <model_path> --exp_set 00

# Merge 사용 (예: 50 iter마다 5k~35k 구간)
python train_ges_merge.py -s <source_path> -m <model_path> --exp_set 00 \
  --merge_every 50 --merge_start_iter 5000 --merge_stop_iter 35000 \
  --merge_prim_color_thresh 0.09 --merge_prim_normal_thresh 0.12 \
  --merge_rgb_thresh 0.08 --knn_k 8
```

## 의존성

- `train_merge.compute_merged_ges`, `apply_merge_to_model` (GES 2DGS merge 로직)
- `gsplat.graph_merge`: `build_knn_graph`, `update_graph_after_merge`, `compute_edge_costs`, `greedy_edge_matching`
- COLMAP Scene 로드 후 `get_edge_cost_data_from_scene(scene)`으로 이미지·dummy depth/normal·c2w·K 리스트 생성

## 2DGS

- LaplacianModel 사용. Optimizer step 이후 `_scaling[:, 2] = log(1e-6)` 로 normal 축 scale 고정 (train_merge와 동일).

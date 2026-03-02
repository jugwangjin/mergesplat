# 구현 문서 (impl)

| 문서 | 내용 |
|------|------|
| [ges_merge_training.md](ges_merge_training.md) | **train_merge.py** 메인: MapAnything dense init, 데이터 해상도, merge 파이프라인, CLI, 평가, run_experiments |
| [graph_merge_edge_collapse.md](graph_merge_edge_collapse.md) | Edge collapse **알고리즘 참조**: merge 공식, edge cost 종류, greedy matching, 그래프 구성 |
| [mapanything_only_dataset.md](mapanything_only_dataset.md) | MapAnything-only 데이터셋: FPS 분할, prepare_mapanything_dataset, mapanything_only.json |
| [mapanything_dataset_inference.md](mapanything_dataset_inference.md) | MapAnything 추론 스크립트 (map-anything 쪽) 입출력·사용법 |
| [train_ges_merge_colmap.md](train_ges_merge_colmap.md) | COLMAP + SfM init + merge (train_ges_merge.py) 참조용. 현재 메인은 train_merge.py |
| [superpixel_2dgs_framework.md](superpixel_2dgs_framework.md) | Superpixel 기반 feed-forward 2DGS (merge 없는 별도 파이프라인) |

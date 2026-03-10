# 구현 문서 (impl)

- **문서·구현 규칙**: [../project_rules.md](../project_rules.md) (글로벌 룰 요약).

| 문서 | 내용 |
|------|------|
| [trisplat_impl.md](trisplat_impl.md) | **trisplat**: mesh extraction(Poisson), TriangleModelVertex, nvdiffrast DepthPeeler, CWD 실행, 출력 경로, CLI |
| [trisplat_atlas_qem_o1.md](trisplat_atlas_qem_o1.md) | **trisplat (참고)**: QEM edge collapse 시 O(1) 동적 아틀라스 업데이트(타일 풀, Free List, 미분 가능 국소 베이킹). simplification/densification 구현 시 참고 |
| [ges_merge_training.md](ges_merge_training.md) | **train_merge.py** 메인: MapAnything dense init, 데이터 해상도, merge 파이프라인, CLI, 평가, run_experiments |
| [graph_merge_edge_collapse.md](graph_merge_edge_collapse.md) | Edge collapse **알고리즘 참조**: 2DGS/3DGS merge 공식, **3DGS edge cost (질량 보존·bloat penalty)**, edge cost 종류, greedy matching, 그래프 구성 |
| [mapanything_only_dataset.md](mapanything_only_dataset.md) | MapAnything-only 데이터셋: FPS 분할, prepare_mapanything_dataset, mapanything_only.json |
| [mapanything_dataset_inference.md](mapanything_dataset_inference.md) | MapAnything 추론 스크립트 (map-anything 쪽) 입출력·사용법 |
| [train_ges_merge_colmap.md](train_ges_merge_colmap.md) | COLMAP + SfM init + merge (train_ges_merge.py) 참조용. 현재 메인은 train_merge.py |
| [train_merge_ghap.md](train_merge_ghap.md) | **train_merge_ghap.py**: COLMAP + GHAP PLY, LaplacianModel 전용, resolve PLY·merge·shape 보정·질량 보존 cost |
| [superpixel_2dgs_framework.md](superpixel_2dgs_framework.md) | Superpixel 기반 feed-forward 2DGS (merge 없는 별도 파이프라인) |

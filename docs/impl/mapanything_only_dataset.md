# MapAnything-only 데이터셋 (COLMAP 없이 train_merge 실행)

## 개요

COLMAP을 돌리지 않고, **입력 디렉터리의 images/만**으로 MapAnything 추론 → depth/pose/normal 저장 → **FPS로 num_init개 학습 뷰** 선정 → 나머지를 test set으로 두고 `train_merge.py`를 실행하는 흐름.

- **sparse/ 있음**: 기존처럼 COLMAP 데이터셋으로 간주. `prepare_mapanything_dataset.py`는 아무것도 하지 않고 종료.
- **sparse/ 없음**: MapAnything-only. `prepare_mapanything_dataset.py`로 한 번 준비 후 `train_merge.py --dataset_dir <path>` 만으로 학습·평가.

## 사용 순서

### 1) MapAnything 추론 (map-anything에서 실행)

```bash
cd map-anything
python scripts/run_mapanything_inference_on_dataset.py \
  --images_dir /path/to/dataset/toilet/images \
  --output_dir /path/to/dataset/toilet/mapanything_output
```

- MapAnything 실행 스크립트는 **map-anything** 쪽에만 둠. ges-splatting에서는 호출하지 않음.

### 2) FPS 분할 + 플래그 (ges-splatting)

```bash
cd ges-splatting
python prepare_mapanything_dataset.py --dataset_dir ../dataset/toilet --num_init 20
```

- `dataset_dir/images/` 와 `dataset_dir/mapanything_output/` (depth/*.npy 등) 이 있어야 함.
- `dataset_dir/sparse/` 가 있으면 COLMAP 데이터셋으로 간주하고 스크립트는 그대로 종료.
- 없으면:
  1. 카메라 위치 + 방향으로 **FPS (position + direction)** 해서 `num_init` 개 뷰를 학습용으로 선택.
  2. 나머지 뷰 이미지를 `dataset_dir/test_images/` 에 심링크(또는 `--copy_test_images` 시 복사).
  3. `dataset_dir/mapanything_only.json` 에 `train_indices`, `test_indices`, `stems`, `num_init` 저장.

### 3) 학습

```bash
python train_merge.py --dataset_dir ../dataset/toilet --model_path outputs/merge
```

- `mapanything_only.json` 이 있으면:
  - `mapanything_dir` = `dataset_dir/mapanything_output`, `images_dir` = `dataset_dir/images` 로 자동 설정.
  - `train_indices` 뷰만으로 초기 모델 생성 및 학습.
  - 평가 시 **test_indices 뷰 전부** 사용 (1/3 서브셋 없음), MapAnything pose로 렌더링 후 PSNR 등 저장.

## 디렉터리 구조

- **MapAnything-only** (sparse 없음):
  - `dataset_dir/images/`          # 입력 이미지
  - `dataset_dir/mapanything_output/`  # depth/, pose/, normal/, intrinsics/, mask/
  - `dataset_dir/test_images/`    # 테스트 뷰 이미지 (심링크 또는 복사)
  - `dataset_dir/mapanything_only.json`  # train_indices, test_indices, stems, num_init

- **COLMAP** (기존):
  - `dataset_dir/images/`, `dataset_dir/sparse/`, `dataset_dir/test_images/`
  - MapAnything 출력은 별도 경로에 두고 `--mapanything_dir`, `--dataset_dir` 로 지정.

## FPS (position + direction)

- 각 뷰: 카메라 위치 `c2w[:3,3]`, 방향 `c2w[:3,:3] @ [0,0,1]` (OpenCV +Z forward).
- 위치는 zero-mean + scale 정규화, 방향은 단위 벡터로 6D 점으로 묶어서 **Farthest Point Sampling**.
- 선정된 `num_init` 개 인덱스 = 학습 뷰, 나머지 = 테스트 뷰.

## train_merge.py 변경 요약

- `--dataset_dir` 만 줄 수 있음. `mapanything_only.json` 이 있으면 `--mapanything_dir`/`--images_dir` 자동 설정.
- `create_model_from_mapanything(..., view_indices=train_indices)` 로 학습 뷰만 역투영.
- `cameras` = 학습 뷰에 해당하는 카메라만 사용.
- `eval_render`: MapAnything-only면 COLMAP/Umeyama 없이 MapAnything pose로 test_images 전부 렌더 후 메트릭 저장.

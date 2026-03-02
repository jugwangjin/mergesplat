# MapAnything 데이터셋 이미지 추론 스크립트

## 개요

`map-anything/scripts/run_mapanything_inference_on_dataset.py`는 `dataset/images`에 있는 이미지들에 대해 MapAnything으로 inference를 수행하고, 뷰별로 **depth**, **normal**, **pose**, **intrinsics** 등을 저장하는 스크립트이다.

## 사용법

map-anything 루트에서 실행:

```bash
cd map-anything
python scripts/run_mapanything_inference_on_dataset.py --images_dir ../dataset/images --output_dir ../dataset/mapanything_output
```

기본값이 `../dataset/images`, `../dataset/mapanything_output`이므로 위와 동일하게 실행하려면 인자 생략 가능:

```bash
cd map-anything
python scripts/run_mapanything_inference_on_dataset.py
```

Apache 2.0 모델 사용:

```bash
python scripts/run_mapanything_inference_on_dataset.py --apache
```

N장마다 1장만 사용 (stride):

```bash
python scripts/run_mapanything_inference_on_dataset.py --stride 2
```

## 입출력

- **입력**: `--images_dir` 폴더 내 `.jpg`, `.jpeg`, `.png` (및 `.heic`, `.heif` 지원 시) 이미지. `load_images()`와 동일한 정렬·stride로 로드.
- **출력**: `--output_dir` 아래 서브디렉터리별로 이미지 파일명 stem 기준으로 저장.
  - `depth/`     : `depth_z` (H,W) float32 `.npy`
  - `normal/`    : pts3d로부터 `points_to_normals()`로 계산한 법선 (H,W,3) float32 `.npy`
  - `pose/`      : cam2world 4x4 행렬 (OpenCV) float32 `.npy`
  - `intrinsics/`: 3x3 pinhole intrinsics float32 `.npy`
  - `confidence/`: per-pixel confidence float32 `.npy`
  - `mask/`      : 유효 픽셀 마스크 bool `.npy`

## 구현 요약

- MapAnything `model.infer(views, ...)` 사용 (image-only inference).
- 뷰 순서는 `get_image_filenames_same_order_as_load_images()`로 `load_images(images_dir, stride)`와 동일한 파일 목록·순서를 보장.
- Normal은 모델 출력에 없으므로 `mapanything.utils.geometry.points_to_normals(pts3d, mask=mask)`로 계산 후 저장.

## gsplat 데이터셋 연동

저장된 MapAnything 결과는 **gsplat**의 데이터셋으로 사용할 수 있다.

- **위치**: `gsplat/datasets/mapanything.py`
- **클래스**: `Parser`, `Dataset` (COLMAP 데이터셋과 동일한 반환 형식)
- **Parser**: `data_dir`에 MapAnything 출력 디렉터리(depth/, pose/, intrinsics/, mask/ 등), `images_dir`에 원본 이미지 폴더. depth unproject로 초기 포인트 클라우드(`points`, `points_rgb`) 생성 → `init_type="sfm"` 사용 가능.
- **Dataset**: `K`, `camtoworld`, `image`, `image_id`, `camera_idx`, `mask` 및 선택 시 `points`/`depths`(depth loss용), `normal` 반환.

### simple_trainer에서 사용

```bash
# MapAnything 출력으로 학습 (data_dir = mapanything_output 경로, 이미지는 기본값 ../images)
python simple_trainer.py default --data_type mapanything --data_dir ../dataset/kitchen/mapanything_output --result_dir results/kitchen

# 이미지 경로를 따로 지정할 때
python simple_trainer.py default --data_type mapanything --data_dir ../dataset/kitchen/mapanything_output --data_images_dir ../dataset/kitchen/images --result_dir results/kitchen
```

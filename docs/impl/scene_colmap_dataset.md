# COLMAP 데이터셋 및 Train/Test 분리

## 지원하는 디렉터리 구조

`scene/` 로더는 다음 구조를 기대합니다.

```
<source_path>/
  images/          # 이미지 폴더 (이름은 --images로 변경 가능, 기본 "images")
  sparse/          # 또는 sparse/0/ (COLMAP 기본 출력)
    cameras.bin    # 카메라 내부 파라미터 (또는 cameras.txt)
    images.bin     # 카메라 외부 파라미터·이미지 이름 (또는 images.txt)
    points3D.bin   # 3D 포인트 (또는 points3D.txt)
```

- **sparse 위치**: `sparse/` 아래에 직접 bin/txt가 있거나, **`sparse/0/`** 아래에 있어도 인식함. `_colmap_sparse_dir()`가 `sparse` vs `sparse/0` 중 존재하는 쪽을 사용.
- **Binary 우선**: 해당 sparse 폴더에서 `images.bin`, `cameras.bin` 있으면 binary 사용. 없으면 `images.txt`, `cameras.txt` 시도.
- **points3D**: `points3D.bin` 또는 `points3D.txt` → 첫 로드 시 같은 sparse 폴더에 `points3D.ply`로 변환해 두고 재사용.
- `sparse/0/` 아래의 `frames.bin`, `rigs.bin`, `project.ini` 등은 **사용하지 않음** (COLMAP 표준 재구성에는 `cameras.bin`, `images.bin`, `points3D.bin`만 필요).

따라서 `Auditorium/images`, `Auditorium/sparse/0/{cameras,images,points3D}.bin` 또는 `Auditorium/sparse/{...}.bin` 구조 모두 **그대로 사용 가능**합니다.

## 인자 (ModelParams)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--source_path` | "" | 씬 루트 경로 (예: `.../full_tat_colmap/Auditorium`). |
| `--model_path` | "" | 학습 결과 저장 경로 (output). |
| `--images` | "images" | 이미지가 들어 있는 **서브폴더 이름** (source_path 아래). |
| `--eval` | False | True면 train/test 분리 사용. False면 전부 train, test 비움. |

실행 예:

```bash
python train_ges.py -s /path/to/full_tat_colmap/Auditorium -m output/Auditorium --eval
```

이미지 디렉터리가 `images`가 아니면 `--images <폴더명>` 으로 지정.

## Train/Test 분리 위치

**파일**: `scene/dataset_readers.py` → `readColmapSceneInfo(path, images, eval, llffhold=8)`.

- **진입**: `scene/__init__.py`의 `Scene.__init__`에서  
  `source_path/sparse` 또는 `source_path/sparse/0` 아래에 `images.bin` 또는 `images.txt`가 있으면 Colmap으로 인식하고  
  `sceneLoadTypeCallbacks["Colmap"](...)` → `readColmapSceneInfo` 호출.  
  `readColmapSceneInfo` 내부에서 `_colmap_sparse_dir(path)`로 실제 사용할 sparse 서브디렉터리(`sparse` 또는 `sparse/0`)를 결정.

- **분리 로직** (같은 파일 145–152행 근처):
  1. 모든 카메라를 `image_name` 기준으로 **정렬**: `cam_infos = sorted(..., key=lambda x: x.image_name)`.
  2. **`eval == True`**:
     - **Train**: `idx % llffhold != 0` 인 카메라.
     - **Test**: `idx % llffhold == 0` 인 카메라.
     - 기본 `llffhold=8` → 대략 **1/8을 test**, 나머지 train (LLFF 스타일).
  3. **`eval == False`**:
     - **Train**: 전체 카메라.
     - **Test**: `[]`.

즉, **train/test split은 `dataset_readers.readColmapSceneInfo` 안에서, `eval` 플래그와 고정된 `llffhold=8`로만** 이루어집니다.  
`llffhold`는 현재 인자로 노출되어 있지 않고, 8로 하드코딩되어 있습니다.

## 카메라 모델

- **지원**: `PINHOLE`, `SIMPLE_PINHOLE` (undistorted).
- 그 외 모델은 `readColmapCameras`에서 `assert False`로 막혀 있음.

## ges-splatting vs GHAP dataset_readers 차이

- **colmap_loader.py**: 두 프로젝트 모두 **동일** (COLMAP bin/txt 파싱 로직).
- **dataset_readers.py** — **COLMAP test set은 동일** (train_merge_ghap vs train_and_prune 공정 비교용):
  - **ges-splatting**: `readColmapSceneInfo(path, images, eval, llffhold=8)` — eval 시 카메라를 `image_name`(stem) 기준 정렬 후 `idx % 8 == 0` = test.
  - **GHAP**: eval 시 **동일 로직** 적용 — `test.txt` 미사용, 정렬 키 `_image_name_stem`(basename without extension), `idx % LLFFHOLD == 0` = test (`LLFFHOLD=8`). train_merge_ghap 결과와 visualize_experiment_results.py에서 step0/final 비교 시 같은 test view 사용.
  - GHAP만의 추가: `depths`, `train_test_exp` 인자, `CameraInfo`에 `depth_params`, `depth_path`, `is_test` 필드.

## 요약

- 예시 Auditorium 구조(`database.db`, `images`, `sparse/0/*.bin`)는 **지원됨**.
- **Train/test 분리**는 `scene/dataset_readers.py`의 `readColmapSceneInfo`에서, **`--eval`** 이 켜져 있을 때만 적용되며, **이미지 이름 정렬 후 `idx % 8 == 0`을 test**로 쓰는 방식입니다.

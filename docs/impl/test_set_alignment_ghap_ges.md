# GHAP vs ges-splatting Test Set 동일성

아래 스크립트/렌더 경로는 **동일한 test set**을 쓰도록 맞춰 두었음.

## Test set 정의 (공통)

- **정렬**: COLMAP 카메라를 **image_name의 stem**(확장자 제외) 기준 정렬.
- **분리**: 정렬된 리스트에서 **idx % 8 == 0** 인 뷰 = test, 나머지 = train (llffhold=8).
- **이미지 누락**: 해당 이미지 파일이 없으면 그 카메라를 **스킵** (양쪽 모두 동일하게 적용 → 카메라 목록·인덱스 일치).

## 관련 파일별 출처

| 파일 | test set 출처 | 비고 |
|------|----------------|------|
| **GHAP/render.py** | `Scene(dataset, ...)` → `scene.getTestCameras()` | GHAP `scene/__init__.py` → `sceneLoadTypeCallbacks["Colmap"]` → **GHAP** `dataset_readers.readColmapSceneInfo`. `args.eval` 사용. |
| **GHAP/train_and_prune.py** | `Scene(dataset, ...)` → `scene.getTestCameras()` (training_report 등) | 위와 동일. `run_from_pointcloud.sh`에서 `--eval` 전달. |
| **ges-splatting/train_merge_ghap.py** | `sceneLoadTypeCallbacks["Colmap"](..., use_eval)` → `scene_info.test_cameras` | **ges-splatting** `scene/dataset_readers.readColmapSceneInfo`. `eval=True` 기본. test_cameras로 GT 저장·step0/final 렌더. |
| **ges-splatting/visualize_experiment_results.py** | 직접 정의 없음 | `merge_process/test_view*_gt.jpg`, `step*_test_view*` 등 **train_merge_ghap이 만든 파일**만 읽음 → train_merge_ghap의 test set과 동일. |

**ges-splatting/render.py**: ges-splatting 루트에는 별도 `render.py` 없음. test view 렌더는 **train_merge_ghap.py** 내부에서 수행됨.

## 양쪽 dataset_readers 정렬·분리 일치

- **GHAP** `scene/dataset_readers.py`  
  - 정렬: `_image_name_stem(cam)` = `os.path.basename(cam.image_name).split(".")[0]` (GHAP은 `image_name = extr.name` 이므로 stem과 동일).  
  - 분리: `idx % llffhold == 0` → test, `LLFFHOLD = 8`.  
  - 이미지 없으면 `continue` (스킵).

- **ges-splatting** `scene/dataset_readers.py`  
  - 정렬: `key=lambda x: x.image_name` (여기서 `image_name` = stem만 저장).  
  - 분리: `idx % llffhold == 0` (기본 8) → test.  
  - 이미지 없으면 `continue` (스킵).

같은 `source_path`, 같은 `images` 폴더면 **동일 카메라 목록·동일 test 인덱스**가 됨.

## mini-splatting

- **scene/dataset_readers.py**: `readColmapSceneInfo`에서 `sorted(..., key=lambda x: x.image_name)`, `idx % llffhold == 0` (기본 8) → test. 이미지 파일 없으면 해당 카메라 **스킵** (동일 정책 적용). 이미지는 numpy 배열로 로드해 파일 핸들 누수 방지.
- 따라서 같은 source_path·동일 images 폴더면 **동일 test set** (GHAP/ges-splatting과 동일).

## 요약

- **GHAP render.py / train_and_prune.py**: GHAP Scene → GHAP dataset_readers (stem 정렬, idx%8, 스킵).
- **train_merge_ghap.py**: ges-splatting dataset_readers (stem 정렬, idx%8, 스킵).
- **mini-splatting ms/ms_d**: mini-splatting dataset_readers (stem 정렬, idx%8, 스킵).
- **visualize_experiment_results.py**: train_merge_ghap이 저장한 test view 결과만 사용.

세 코드베이스의 정렬 키·llffhold·스킵 정책이 맞춰져 있으므로, **동일한 test set**을 사용한다고 보면 됨.

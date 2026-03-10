# train_merge.py vs train_and_prune.py: 이미지 로드·렌더·저장 차이

GHAP 내 `train_merge.py`(merge 학습)와 `train_and_prune.py`(GHAP 본학습) 간 **이미지 로드, 테스트 뷰 렌더, 저장** 메커니즘 차이 정리. color space, 배경, 노출, 저장 형식 등 비교.

---

## 1. 이미지 로드 (동일)

- **경로**: 둘 다 Colmap → `sceneLoadTypeCallbacks["Colmap"]` → `CameraInfo`(image_path) → `loadCam` → `cameraList_from_camInfos`.
- **픽셀**: `Image.open(cam_info.image_path)` → `PILtoTorch(image, resolution)` (`utils/general_utils.py`).
  - GHAP `PILtoTorch`: `pil_image.resize(resolution)` → `np.array(...) / 255.0` → `permute(2,0,1)` → **float [0,1], C×H×W**.
- **Camera 저장**: `Camera` 생성 시 `resized_image_rgb = PILtoTorch(...)`, `gt_image = resized_image_rgb[:3]`, `self.original_image = gt_image.clamp(0.0, 1.0).to(data_device)`.
- **결론**: 입력 이미지는 동일 경로·동일 **sRGB [0,1]** 해석. color space 차이 없음.

---

## 2. 배경 (background)

| 항목 | train_merge.py | train_and_prune.py |
|------|----------------|---------------------|
| 설정 | `bg_color = [0, 0, 0]` 고정 | `bg_color = [1,1,1]` if `dataset.white_background` else `[0,0,0]` |
| 렌더 | 항상 검정 배경 | 학습 시 `white_background`이면 흰 배경 |

- **영향**: GHAP을 `--white_background`으로 학습했으면 `test/`, `test_before_prune/` 렌더는 **흰 배경**; `train_merge`의 step0/final 렌더는 **검정 배경**. PSNR/시각 비교 시 배경 불일치 가능.
- **권장**: 비교 시 동일 배경 쓰려면 `train_merge`에 `--white_background` 옵션을 추가해 `train_and_prune`와 맞추거나, 둘 다 검정으로 통일.

---

## 3. render() 호출 · Exposure

| 항목 | train_merge.py | train_and_prune.py |
|------|----------------|---------------------|
| 호출 | `render(..., use_trained_exp=use_trained_exp)` | `render(..., use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)` |
| use_trained_exp | **PLY와 동일**: `load_ply(use_train_test_exp=True)` 후 `exposure.json` 있으면 로드 → `use_trained_exp=True`, 없으면 False | dataset 플래그 사용 (보통 True면 per-camera exposure 적용) |
| separate_sh | 미전달 → **False** | SparseAdam 있으면 True (DC/rest 분리 전달) |

- **train_merge**: init PLY 경로 기준으로 `exposure.json`을 찾아 있으면 `pretrained_exposures`로 로드하고, 모든 `render()`에 `use_trained_exp=True`를 넘김. 없으면 identity(기본)로 `use_trained_exp=False`.
- **use_trained_exp=True**일 때: `rendered_image = matmul(rendered_image, exposure[:3,:3]) + exposure[:3,3]` 로 per-camera exposure 적용.
- **separate_sh**: rasterizer에 DC/rest를 따로 넘길지 여부. 수치 차이는 있을 수 있으나 color space 자체는 동일.

---

## 4. 테스트 뷰 crop (train_test_exp)

- **train_and_prune** (`save_test_renders_to_disk`):  
  `train_test_exp==True` 이면  
  `rendering = rendering[..., rendering.shape[-1]//2:]`, `gt = gt[..., gt.shape[-1]//2:]` 로 **오른쪽 절반만** 저장.
- **train_merge**: crop 없음. **전체 해상도**로 저장 (`step00000_test_view*`, `step{iter}_test_view*`).
- **결과**: GHAP이 `train_test_exp`로 학습된 경우, `test/`·`test_before_prune/` 이미지는 **절반 너비**; merge 쪽 렌더는 **전체 너비**. 해상도·영역이 달라서 PSNR/SSIM을 그대로 비교하면 안 됨.
- **권장**: `train_test_exp` 사용 시 `train_merge`에서도 동일하게 오른쪽 절반만 crop 해서 저장하거나, 비교 시 둘 다 동일 영역만 사용해 메트릭 계산.

---

## 5. 저장 방식

| 항목 | train_merge.py | train_and_prune.py (save_test_renders_to_disk) |
|------|----------------|--------------------------------------------------|
| 렌더 | `(render.clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()` → `Image.fromarray(...).save(..., .png)` | `torchvision.utils.save_image(rendering, ..., .png)` |
| GT | `original_image.permute(1,2,0).clamp(0,1).cpu().numpy()*255` → `Image.fromarray(..., .jpg, quality=90)` | `torchvision.utils.save_image(gt, ..., .png)` |
| 값 범위 | 둘 다 ** [0,1] → 255 → uint8** 로 저장. | `save_image`는 [0,1] float 입력 가정, 내부에서 255 곱해 저장. |

- **color/값**: 둘 다 [0,1] 기준이므로 **저장 시 color space 차이 없음**. 포맷만 PNG vs JPG(GT) 차이.
- **파일명/레이아웃**: train_merge는 `renders/step*_test_view{i}_{cam_name}.png`, `merge_process/test_view{i}_gt.jpg`. train_and_prune은 `test/` 또는 `test_before_prune/` 아래 `last_folder/renders/{idx:05d}.png`, `.../gt/{idx:05d}.png`.

---

## 6. 요약 표

| 구분 | train_merge | train_and_prune |
|------|-------------|-----------------|
| 이미지 로드 | Colmap + PILtoTorch, [0,1] | 동일 |
| 배경 | 항상 [0,0,0] | white_background 시 [1,1,1] |
| use_trained_exp | False (미전달) | dataset.train_test_exp |
| separate_sh | False | SPARSE_ADAM_AVAILABLE |
| train_test_exp 시 crop | 없음 (전체 프레임) | 렌더/GT 오른쪽 절반만 저장 |
| 렌더 저장 | PIL PNG [0,1→255] | torchvision PNG [0,1] |
| GT 저장 | PIL JPG [0,1→255] | torchvision PNG [0,1] |

---

## 7. 비교 시 권장 정렬 사항

1. **배경**: `train_and_prune`에서 사용한 `white_background`와 동일하게 `train_merge`에서도 배경 설정 (옵션 추가 권장).
2. **노출**: exposure 학습 데이터셋이면 `train_merge`에서도 `use_trained_exp=True`로 렌더 (및 동일 camera list/이름 매핑).
3. **crop**: `train_test_exp` 사용 시 `train_merge`에서도 테스트 뷰 렌더/GT를 오른쪽 절반으로 crop 후 저장하거나, 메트릭 계산 시 동일 영역만 사용.
4. **메트릭**: 위가 맞춰진 상태에서 동일 해상도·동일 영역으로 PSNR/SSIM 계산.

이 문서는 `docs/impl/` 아래 구현 관리용으로 두었으며, 이후 코드 변경 시 함께 갱신하는 것이 좋다.

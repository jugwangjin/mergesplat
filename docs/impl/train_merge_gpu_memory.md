# train_merge.py GPU 메모리 사용 분석

## 30GB는 이미지만으로 설명되지 않음

뷰가 200장이어도, 풀해상도 GT 이미지 200장 = 200 × (H×W×3×4 bytes) ≈ 200 × 25MB ≈ **5GB** 수준이다.  
즉 **이미지만으로 30GB를 쓰는 것이 아니라**, 30GB의 대부분은 아래에서 나온다.

### 1. **Diff-Gaussian 래스터라이저 (주요 원인)**

- `render_laplacian` → `GaussianRasterizer`(diff-gaussian) 사용.  
  풀해상도(예: 1920×1080) × Gaussian 8만~10만 개일 때 **타일 버퍼·포인트 리스트·역전파용 중간 버퍼**가 수십 GB 단위로 쌓이는 경우가 많다.
- 3DGS 계열 학습에서 GPU 메모리 대부분을 차지하는 것이 보통 이 래스터라이저다.

### 2. **옵티마이저 + 역전파 버퍼**

- Gaussian 9만 개 × (xyz, features_dc/rest, scaling, rotation, opacity, shape) 파라미터와 Adam 2배 상태, gradient 누적 등으로 **1~3GB** 단위.

### 3. **GT 이미지 전 뷰 GPU 상주**

- `Camera.original_image`를 전 뷰 풀해상도로 GPU에 두면: 뷰 200장 × 약 25MB ≈ **5GB**.
- 30GB의 직접 원인이라기보다는, **그 5GB를 줄일 수 있는 부분**이다.

### 4. **Edge cost용 데이터**

- `ec_imgs_at_depth`, depth, normal을 전 뷰 depth 해상도로 GPU에 두는 부분. 뷰 200 × (H_d×W_d×…) → **수백 MB~1GB** 정도.

### 5. **기타**

- LPIPS(Alex) 로드 시 수백 MB, 뷰 변환 행렬 등은 수 MB 수준.

---

## 요약

- **30GB ≈ 래스터라이저(가장 큼) + 옵티마이저/역전파 + GT 이미지(~5GB) + edge cost + 기타.**
- 이미지 200장은 ~5GB이므로, 30GB의 “대부분”은 래스터라이저와 옵티마이저/역전파 쪽이다.

---

## 대응: GT 이미지만 CPU에 두기 (선택)

- 30GB를 근본적으로 줄이려면 **렌더 해상도 감소**나 **래스터라이저 쪽 최적화**가 필요하다.
- 그와 별개로, **GT 이미지 5GB는 불필요하게 전 뷰를 GPU에 둔 것**이므로, CPU에 두고 사용할 때만 올리면 그만큼 줄일 수 있다.

### 구현 요약

1. **`--camera_image_cpu`**  
   - `make_cameras(..., data_device="cpu")` 로 모든 `Camera.original_image`를 CPU에만 둠.
2. **학습 루프**  
   - `gt_image = cam.original_image.to(device)` 로 **현재 스텝에서 선택된 뷰 1장만** GPU로 복사해 사용.
3. **효과**  
   - (뷰 수 - 1) × 풀해상도 이미지 메모리 절감.  
   - 예: 200뷰 1080p 기준 **약 5GB** 절감 (30GB → 25GB 수준으로 감소).

---

## 사용 방법

**30GB 줄이기 (가장 효과 큼): 학습 해상도 낮추기**

```bash
python train_merge.py \
  --mapanything_dir ... \
  --dataset_dir ... \
  --model_path ... \
  --train_resolution_scale 0.5
```

- `--train_resolution_scale 0.5`: 렌더/GT를 절반 해상도로 학습 → **래스터라이저 메모리가 약 1/4 수준**으로 감소 (면적 기준).
- `0.25`면 1/16 수준으로 더 줄어듦. 풀해상도는 `1.0`(기본값).

**추가로 적용 가능**

- `--camera_image_cpu`: GT 이미지를 CPU에 두고 사용 시에만 GPU로 복사 (뷰 많을 때 수 GB 절감).

---

## 쓸모없이 사용되는 메모리 (조사 결과)

### 1. **Edge cost용 이미지 전 뷰 GPU 상주 (개선함)**

- `ec_imgs_at_depth`: **전 뷰** depth 해상도 RGB를 **학습 시작 시 전부 GPU에 올려 두고** 끝까지 유지한다.
- Edge cost는 **merge 구간(merge_every마다)** 에만 호출되고, 호출 시에도 뷰 루프에서 **한 뷰씩만** 사용한다.  
  → 나머지 시간에는 전 뷰 분량이 GPU에 쓸모없이 상주한다.
- **조치**: `ec_imgs_at_depth`·`ec_Ks_at_depth`는 CPU에만 두고, `compute_edge_costs` 루프 안에서 해당 뷰만 `.to(device)` 로 올린다.  
  → GPU에는 **한 뷰분**만 유지되므로 수백 MB~1GB 절감.

### 2. **get_tensors의 ec_imgs (사용처 없음)**

- `ec_imgs, ec_deps, ec_nrms, ec_c2ws, ec_Ks = ma.get_tensors(device)` 에서 **ec_imgs**는 **전혀 사용하지 않는다.**  
  Edge cost에는 `ec_imgs_at_depth`를 넘기고, 학습 loss에는 `Camera.original_image`를 쓴다.
- `get_tensors`가 반환하는 full-res `ec_imgs` 리스트는 CPU 텐서이므로 GPU 메모리는 쓰지 않지만, **CPU 메모리와 생성 비용만 낭비**한다.  
  필요하면 `get_tensors`에서 imgs를 빼거나, edge cost 전용 경로를 두어 imgs를 만들지 않게 할 수 있다.

### 3. **LPIPS가 eval 외 시간에도 GPU 상주**

- `_get_lpips()`로 처음 eval 시점에 LPIPS(Alex)를 로드해 `.cuda()`로 GPU에 두고, **그 뒤로 계속 GPU에 유지**한다.
- LPIPS는 **eval 시에만** 쓰이므로, 학습 루프 대부분 구간에서는 GPU에서 사용되지 않는다. (수백 MB)

### 4. **Graph build 시 K-NN (torch.cdist) — 기본으로 완화함**
- `build_hybrid_graph`에서 뷰별로 `torch.cdist(pts_v, pts_other)`를 쓰면 (예: 1만×10만) 뷰당 수 GB가 할당되어 **peak ~10GB, reserved 27GB**까지 올라갑니다.
- **조치**: 기본값으로 cross-view K-NN을 **CPU에서** 수행하도록 했습니다 (scipy cKDTree). `--no_graph_knn_on_cpu`를 주면 예전처럼 GPU cdist 사용 (속도는 빠르지만 메모리 높음).

### 5. **ec_c2ws, ec_Ks (전 뷰 GPU)**

- `get_tensors(device)`로 **c2w, K를 전 뷰 GPU**에 올린다. Edge cost는 merge 시에만 쓰고, 뷰 루프에서 한 뷰씩만 참조한다.
- 용량은 뷰당 4×4 + 3×3 float → 작지만, 원하면 CPU에 두고 루프 안에서 `.to(device)` 로 올릴 수 있다.

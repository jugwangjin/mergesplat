# Superpixel-based 2DGS Feed-Forward Scene Generation

## 개요

MapAnything 출력(RGB, depth, normal, pose, intrinsics)으로부터 **optimization 없이** feed-forward 방식으로 2D Gaussian Splatting scene을 생성하는 프레임워크.

핵심 설계 원칙:
- **SH 계수 미사용**: view-dependent color 대신 view-dependent opacity로 처리
- **Cross-view Gaussian merge 없음**: 각 view에서 생성된 Gaussian은 독립적으로 유지, view-dependent opacity가 렌더링 시 적절한 선택 수행
- **Adaptive primitive 크기**: 균일 영역은 소수의 큰 primitive, 디테일 영역은 다수의 작은 primitive

## 파이프라인

```
Input (per view)
  ├─ RGB image
  ├─ Depth map (MapAnything)
  ├─ Normal map (MapAnything)
  ├─ Camera pose (MapAnything)
  └─ Intrinsics (MapAnything)
      │
      ▼
[1] Adaptive Superpixel Segmentation (slic.py)
    ├─ 7D feature 구성: RGB + log-depth + normal (normalized)
    ├─ SLIC initial segmentation
    ├─ Merge: 유사 인접 cell 합치기 (RAG 기반)
    ├─ Split: 고분산 cell k-means 분할
    └─ Convex decomposition: non-convex cell binary split
      │
      ▼
[2] 2DGS Primitive 생성 (gaussian_init.py)
    ├─ means: cell centroid depth unproject → 3D world point
    ├─ quats: avg normal → surfel orientation quaternion
    ├─ scales: surfel plane PCA → 2D extent, 3rd ≈ 0
    ├─ colors: cell mean RGB (no SH)
    ├─ opacities: high initial value
    └─ source_cam_pos: 생성 view의 camera position
      │
      ▼
[3] Multi-view Concatenate (scene_generator.py)
    └─ 모든 view의 Gaussian을 단순 concatenate (merge 없음)
      │
      ▼
[4] Rendering (renderer.py)
    ├─ View-dependent opacity: per-Gaussian softmax across source views
    └─ gsplat rasterization_2dgs (sh_degree=None)
```

## 파일 구조

```
gsplat/
  superpixel_init/
    __init__.py              # 패키지 exports
    slic.py                  # SLIC + merge + split + convex decomp
    gaussian_init.py         # superpixel → 2DGS params
    scene_generator.py       # multi-view 처리 + concatenate
    renderer.py              # view-dependent opacity + render wrapper
  superpixel_trainer.py      # CLI entry point
  run_experiments.sh         # threshold 조합 실험 스크립트
```

## 핵심 파라미터

### Superpixel 제어

| 파라미터 | 기본값 | 효과 |
|---------|-------|------|
| `n_segments` | 800 | SLIC 초기 segment 수. 높을수록 세밀한 시작점 |
| `compactness` | 10.0 | SLIC spatial vs feature 균형. 높을수록 regular |
| `merge_threshold` | 0.05 | 인접 cell merge 기준 (7D feature distance). 높을수록 aggressive merge → fewer Gaussians |
| `split_threshold` | 0.02 | cell 분할 기준 (7D feature variance). 낮을수록 aggressive split → more Gaussians |
| `min_pixels` | 16 | 최소 cell 크기. 이보다 작으면 split/merge 중단 |

### num_Gaussians / quality 트레이드오프

- **최소 Gaussian (최대 압축)**: `merge_threshold=0.15, split_threshold=0.05, n_segments=400`
- **균형**: `merge_threshold=0.05, split_threshold=0.02, n_segments=800`
- **최대 품질**: `merge_threshold=0.02, split_threshold=0.005, n_segments=1200`

### View-dependent opacity

| 파라미터 | 기본값 | 효과 |
|---------|-------|------|
| `temperature` | 0.5 | softmax temperature. 낮을수록 sharp (best view만 보임), 높을수록 soft blending |

## View-Dependent Opacity 원리

SH coefficient 대신 각 Gaussian의 opacity를 렌더링 시점에 **source view 간 softmax**로 동적 조절.

**핵심: 절대 cosine falloff가 아닌 상대적 softmax이므로, 가장 잘 맞는 view의 Gaussian은 항상 opacity ≈ 1.**

```
# 각 Gaussian 위치 p에서 모든 source view에 대한 유사도 계산
for each source view s in V:
    source_dir_s = normalize(source_cam_s - p)
render_dir = normalize(render_cam - p)
cos_sim_s = dot(source_dir_s, render_dir)     # (V,)

# softmax across source views → 상대적 weight
weights = softmax(cos_sims / temperature)      # (V,), sum = 1

# 각 Gaussian은 자기 source view의 weight를 사용
effective_opacity = base_opacity * weights[my_view_id]
```

- `temperature` 낮음 (0.1): hard selection — best view의 Gaussian만 visible
- `temperature` 높음 (1.0): soft blending — 여러 view의 Gaussian이 혼합
- 모든 source view가 render camera에서 멀어도, softmax이므로 best view의 weight ≈ 1 보장
- cos_sim이 음수인 view (뒤에서 보는 경우)는 softmax 내에서 자연스럽게 weight ≈ 0

## 평가 (Evaluation)

COLMAP reconstruction의 test 카메라를 사용하여 평가:

1. `{dataset_dir}/sparse/0/`에서 COLMAP 카메라 (전체) 로드
2. `{dataset_dir}/test_images/`에 존재하는 이미지를 test set으로 선택
3. MapAnything 좌표계와 COLMAP 좌표계 정렬 (Umeyama/Procrustes 알고리즘)
   - 학습에 사용된 이미지들의 COLMAP pose ↔ MapAnything pose 대응점으로 similarity transform 계산
4. 변환된 COLMAP test 카메라로 렌더링하여 GT와 비교 (PSNR, SSIM, LPIPS)

`dataset_dir` 미지정 시 `data_dir`의 상위 디렉토리를 기본값으로 사용.

### 디렉토리 구조 요구사항

```
dataset/kitchen/
  images/               # 학습 이미지 (MapAnything 입력)
  test_images/           # 평가용 테스트 이미지
  sparse/0/              # COLMAP reconstruction (cameras.bin, images.bin, points3D.bin)
  mapanything_output/    # MapAnything 추론 결과
```

## 저장 형식

`.pt` checkpoint에 포함되는 데이터 (SH 계수 없음):

| Key | Shape | 설명 |
|-----|-------|------|
| `means` | (N, 3) | 3D world position |
| `quats` | (N, 4) | wxyz quaternion |
| `scales` | (N, 3) | surfel 2D extent + thickness |
| `colors` | (N, 3) | diffuse RGB [0,1] |
| `opacities` | (N,) | base opacity |
| `source_cam_pos` | (N, 3) | 생성 view camera position |
| `view_ids` | (N,) | 생성 view index |
| `camtoworlds` | (V, 4, 4) | 모든 view의 cam2world (MapAnything 좌표계) |
| `Ks` | (V, 3, 3) | 모든 view의 intrinsics |
| `stems` | List[str] | 각 view의 이미지 파일명 stem (좌표 정렬에 사용) |

기존 3DGS 대비 SH 계수 (Gaussian당 45 params) 제거로 메모리 ~80% 절감.

## 사용법

```bash
cd gsplat

# Scene 생성 + 저장
python superpixel_trainer.py --data_dir ../dataset/kitchen/mapanything_output

# Scene 생성 + COLMAP test set 평가
python superpixel_trainer.py --data_dir ../dataset/kitchen/mapanything_output --eval

# dataset_dir 명시적 지정
python superpixel_trainer.py --data_dir ../dataset/kitchen/mapanything_output --dataset_dir ../dataset/kitchen --eval

# 기존 checkpoint 로드 + 평가
python superpixel_trainer.py --ckpt results/superpixel/scene.pt --data_dir ../dataset/kitchen/mapanything_output --eval

# 궤적 렌더링
python superpixel_trainer.py --ckpt results/superpixel/scene.pt --data_dir ../dataset/kitchen/mapanything_output --render_traj

# 여러 조건 실험 (dataset_dir 지정)
bash run_experiments.sh ../dataset/kitchen/mapanything_output ../dataset/kitchen
```

## 의존성

- `scikit-image`: SLIC superpixel
- `scikit-learn`: KMeans (split 단계)
- `scipy`: ConvexHull (convexity 판정)
- `gsplat`: `rasterization_2dgs()` 렌더링
- COLMAP reconstruction (cameras.bin, images.bin) 직접 파싱 (pycolmap 불필요)
- `torchmetrics`: PSNR, SSIM, LPIPS 평가

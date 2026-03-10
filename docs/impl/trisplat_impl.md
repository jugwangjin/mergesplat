# Trisplat 구현 문서

nvdiffrast 기반 mesh(삼각형) 씬 표현 + DepthPeeler 렌더링. Colmap/Blender 데이터셋 유지, mesh extraction 후 per-vertex opacity/SH 학습.

---

## 1. 위치·실행 규칙

- **코드 위치**: `trisplat/` (패키지가 아님, CWD 기준 로컬 임포트).
- **실행**: 반드시 **CWD를 `trisplat/`로** 두고 실행.
  ```bash
  cd /mnt/d/mergesplat/trisplat
  python train.py --source_path <COLMAP_OR_BLENDER> --model_path <OUTPUT_DIR> [--resolution 4] [--eval]
  ```
- **임포트**: `trisplat.` 접두어 없음. `from scene import Scene`, `from utils.loss_utils import l1_loss` 등.

---

## 2. 디렉토리 구조

```
trisplat/
  train.py              # 진입점. training(), test_interval/final 렌더, 저장
  run_experiments_color_opacity.sh   # 4 scene × 2 model (color/opacity only)
  arguments/__init__.py # ModelParams, PipelineParams, OptimizationParams (set_weight 기본 0.5 등)
  scene/
    __init__.py         # Scene: 데이터 로드, mesh 추출, PLY 저장, TriangleModelVertex 초기화
    dataset_readers.py  # Colmap/Blender SceneInfo, CameraInfo, fetchPly, sceneLoadTypeCallbacks
    colmap_loader.py    # read_extrinsics_*, read_intrinsics_*, read_points3D_*, qvec2rotmat
    cameras.py          # Camera (world_view_transform, full_proj_transform, camera_center)
    mesh_extraction.py  # extract_mesh (Poisson + ball-pivoting fallback), save_mesh_ply
    triangle_model_vertex.py  # TriangleModelVertex: vertices, _triangle_indices, vertex_weight, _features_dc/_rest
  renderer/__init__.py  # render(): nvdiffrast DepthPeeler, barycentric 보간, transmittance 누적
  utils/
    camera_utils.py     # loadCam, cameraList_from_camInfos, camera_to_JSON (CameraInfo/Camera 둘 다 지원)
    general_utils.py    # inverse_sigmoid, PILtoTorch, get_expon_lr_func, safe_state
    graphics_utils.py   # BasicPointCloud, getWorld2View2, getProjectionMatrix, fov2focal/focal2fov
    sh_utils.py         # eval_sh, RGB2SH, SH2RGB
    loss_utils.py       # l1_loss, ssim
    image_utils.py      # psnr
    system_utils.py     # mkdir_p, searchForMaxIteration
```

---

## 3. 데이터·씬 흐름

- **입력**: Colmap(`sparse/0`, `images`, `points3D.ply`) 또는 Blender(`transforms_*.json`, `points3d.ply`). `sceneLoadTypeCallbacks["Colmap"]` / `["Blender"]` → `SceneInfo(point_cloud, train_cameras, test_cameras, nerf_normalization, ply_path)`.
- **초기화(신규)**:
  1. `model_path`에 `input.ply` 복사, `cameras.json` 저장 (CameraInfo 기준으로 `camera_to_JSON` 호출).
  2. `extract_mesh(points, normals, ...)`: 전처리(아웃라이어 제거, 노멀 정렬) 후 **(1) Delaunay 기반 alpha shape** → (2) Poisson depth 9→8→7 → (3) Ball Pivoting 순으로 시도 → `vertices [V,3]`, `faces [F,3]`. 모두 실패 시 RuntimeError.
  3. `save_mesh_ply(vertices, faces, model_path/mesh_extracted.ply)`.
  4. `TriangleModelVertex.create_from_mesh(vertices, faces, point_cloud, init_opacity=opt.set_weight)`: vertex 색은 pcd 최근접점 색 → RGB2SH, opacity는 `inverse_sigmoid(init_opacity)`.
- **체크포인트 로드**: `load_iteration` 지정 시 `point_cloud/iteration_<N>/point_cloud_state_dict.pt`에서 vertices, faces, opacity, SH 복원.

**Mesh 추출** (`scene/mesh_extraction.py`): 진입점 `extract_mesh()`. 전처리(통계적 아웃라이어 제거, 노멀 일관 정렬) 후 **(1) Delaunay 기반 alpha shape**(Open3D: tetrahedral Delaunay + alpha carving, alpha=mean_NN_dist*scale) → (2) Poisson depth 9→8→7 → (3) Ball Pivoting(radii from NN distance) 순으로 시도. 첫 번째로 비지 않은 메쉬를 반환.

---

## 4. 렌더링 (renderer/__init__.py)

- **nvdiffrast**: `RasterizeCudaContext`, `DepthPeeler(glctx, pos_clip, tri, resolution)`.
- **입력**: `viewpoint_camera` (Camera), `pc` (TriangleModelVertex), `pipe`, `bg_color`. vertex 위치를 `full_proj_transform`으로 clip space [1,V,4]로 변환.
- **레이어 루프**: `rasterize_next_layer()` → rast (u,v,z/w,triangle_id). triangle_id로 face index, barycentric (u,v,w)로 vertex attribute(opacity + SH coeff) 보간. `eval_sh`로 RGB, alpha로 transmittance 누적. 종료: 유효 픽셀 없음 또는 전 픽셀 transmittance ≤ 1e-3.
- **출력**: `{"render": [3,H,W]}`.

---

## 5. 학습 (train.py)

- **Phase**: `position_freeze_iters` 구간까지 position 고정(color/opacity/SH만), 이후 position도 학습. color/opacity만 쓰려면 `position_freeze_iters`를 iterations와 같게.
- **Loss**: L1 + λ_dssim*(1-SSIM). 배경은 `white_background` 여부에 따라 [1,1,1] 또는 [0,0,0].
- **테스트 렌더**: `--test_interval 3000`이면 매 3000 step마다 test set 렌더 → `model_path/renders/test/iteration_<N>/<image_name>.png` 저장, L1/PSNR 로그. 학습 종료 시 한 번 더 `renders/test/final/`에 저장.
- **저장**: `saving_iterations`에 포함된 step에서 `scene.save(iteration)` → `point_cloud/iteration_<N>/point_cloud_state_dict.pt`.
- **Densification**: 없음. mesh 토폴로지 고정.

---

## 6. 출력 디렉토리 (model_path)

| 경로 | 설명 |
|------|------|
| `cfg_args` | 실행 인자 스냅샷 |
| `input.ply` | 복사된 SfM point cloud |
| `cameras.json` | 카메라 메타 (camera_to_JSON) |
| `mesh_extracted.ply` | 추출 메시 (Poisson 또는 Ball Pivoting, 신규 초기화 시만) |
| `point_cloud/iteration_<N>/point_cloud_state_dict.pt` | 체크포인트 |
| `renders/test/iteration_<N>/*.png` | N step 시점 test 렌더 |
| `renders/test/final/*.png` | 학습 종료 시 test 렌더 |
| TensorBoard 로그 | `model_path`에 직접 |

---

## 7. 주요 인자 (arguments)

- **ModelParams**: `source_path`, `model_path`, `resolution`(-1 또는 1/2/4/8), `white_background`, `eval`, `sh_degree`(기본 3).
- **OptimizationParams**: `iterations`, `position_freeze_iters`, `set_weight`(초기 opacity, 기본 0.5), `feature_lr`, `weight_lr`, `lr_triangles_points_init`, `lambda_dssim`.
- **train.py 추가**: `--test_interval`(기본 3000, 0이면 중간 test 렌더 끔), `--test_iterations`, `--save_iterations`, `--start_checkpoint`.

---

## 8. 호환·주의

- **camera_to_JSON**: `CameraInfo`(width, height, FovY, FovX)와 `Camera`(image_width, image_height, FoVy, FoVx) 둘 다 받도록 `getattr`로 필드 통일.
- **의존성**: torch, numpy, PIL, plyfile, open3d, nvdiffrast, tqdm, tensorboard.

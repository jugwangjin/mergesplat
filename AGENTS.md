# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

MergeSplat combines two submodules into a unified 3D reconstruction + novel-view synthesis pipeline:
- **map-anything** (`/workspace/map-anything`): Meta's MapAnything — feed-forward metric 3D reconstruction (Python/PyTorch)
- **gsplat** (`/workspace/gsplat`): CUDA-accelerated Gaussian Splatting rasterization library
- **scripts/**: Custom pipeline scripts for superpixel-based 2DGS initialization

### Submodule setup

The repo references `gsplat` and `map-anything` as git submodules but **lacks a `.gitmodules` file**, so `git submodule init/update` will not work. The update script clones them from GitHub and checks out the pinned commits. If submodule directories already have content, the update script skips cloning.

### Running services

- **Lint (map-anything):** `ruff check mapanything/` and `ruff format --check mapanything/` from `/workspace/map-anything`
- **Tests (gsplat):** `pytest tests/` from `/workspace/gsplat` — all tests skip without a CUDA GPU
- **Full inference** requires: NVIDIA GPU with CUDA, and HuggingFace Hub network access to download pretrained weights (`facebook/map-anything` or `facebook/map-anything-apache`)
- gsplat's CUDA kernels (rasterization, projection) are GPU-only; Python-level modules (strategies, exporter) work on CPU

### Pipeline scripts

- `scripts/run_mapanything.py` — run MapAnything inference on images, saves depth/normals/poses/intrinsics/pts3d to `.npz`
- `scripts/superpixel_2dgs.py` — superpixel-based 2DGS initialization from MapAnything outputs (SLIC with depth+normal features, cross-view voxel dedup)
- `scripts/pipeline.py` — end-to-end orchestrator: inference → superpixel 2DGS init

Example usage (requires GPU + HuggingFace access):
```bash
python scripts/pipeline.py --image_dir dataset/kitchen/images --output_dir dataset/kitchen --n_segments 1000 --apache
```

The superpixel module itself runs on CPU and can be tested independently with `--skip_inference` if `mapanything_outputs.npz` already exists.

### Key gotchas

- `BUILD_NO_CUDA=1` must be set when installing gsplat on machines without CUDA toolkit, otherwise `pip install -e .` will fail at CUDA extension compilation.
- `$HOME/.local/bin` must be on `PATH` for `ruff`, `pytest`, `rerun`, and other pip-installed tools.
- HuggingFace Hub (`huggingface.co`) is blocked by network egress restrictions in Cloud Agent environments. Model weights cannot be downloaded. Add `huggingface.co` and `cdn-lfs.hf.co` to allowed domains in Network Access settings.
- gsplat v1.5.3's test suite is entirely CUDA-dependent; `pytest` collects and skips all 114 tests on CPU.
- The `dataset/` directory is in `.gitignore` — image data and outputs are local only.

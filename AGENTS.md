# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

MergeSplat combines two submodules into a unified 3D reconstruction + novel-view synthesis pipeline:
- **map-anything** (`/workspace/map-anything`): Meta's MapAnything — feed-forward metric 3D reconstruction (Python/PyTorch)
- **gsplat** (`/workspace/gsplat`): CUDA-accelerated Gaussian Splatting rasterization library

### Submodule setup

The repo references `gsplat` and `map-anything` as git submodules but **lacks a `.gitmodules` file**, so `git submodule init/update` will not work. The update script clones them from GitHub and checks out the pinned commits. If submodule directories already have content, the update script skips cloning.

### Running services

- **Lint (map-anything):** `ruff check mapanything/` and `ruff format --check mapanything/` from `/workspace/map-anything`
- **Tests (gsplat):** `pytest tests/` from `/workspace/gsplat` — all tests skip without a CUDA GPU
- **Full inference** requires: NVIDIA GPU with CUDA, and HuggingFace Hub network access to download pretrained weights (`facebook/map-anything` or `facebook/map-anything-apache`)
- gsplat's CUDA kernels (rasterization, projection) are GPU-only; Python-level modules (strategies, exporter) work on CPU

### Key gotchas

- `BUILD_NO_CUDA=1` must be set when installing gsplat on machines without CUDA toolkit, otherwise `pip install -e .` will fail at CUDA extension compilation.
- The `$HOME/.local/bin` directory must be on `PATH` for `ruff`, `pytest`, `rerun`, and other tools installed via pip with `--user`.
- HuggingFace Hub (`huggingface.co`) may be blocked by network egress restrictions in Cloud Agent environments. Model weights cannot be downloaded in that case.
- gsplat v1.5.3's test suite is entirely CUDA-dependent; `pytest` will collect and skip all 114 tests on CPU.

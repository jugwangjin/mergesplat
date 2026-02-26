"""
Superpixel-based 2DGS initialization from MapAnything outputs.

Groups pixels into superpixels using depth+normal features via SLIC,
then creates one 2D Gaussian Splat per superpixel — dramatically reducing
the number of primitives compared to per-pixel unprojection.

Usage:
    python scripts/superpixel_2dgs.py \
        --input dataset/kitchen/mapanything_outputs.npz \
        --output_dir dataset/kitchen/splat_init \
        --n_segments 1000 \
        --compactness 20
"""

import argparse
import math
import os

import numpy as np
import torch
from skimage.segmentation import slic


def normal_to_quaternion(normals: np.ndarray) -> np.ndarray:
    """Convert normal vectors to quaternions (wxyz convention).

    The quaternion represents a rotation that maps [0, 0, 1] to the target
    normal. For 2DGS, this defines the orientation of the flat disk.

    Args:
        normals: (N, 3) unit normal vectors.

    Returns:
        quats: (N, 4) quaternions in wxyz convention.
    """
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
    z_axis = np.array([0.0, 0.0, 1.0])

    quats = np.zeros((len(normals), 4), dtype=np.float32)

    for i, n in enumerate(normals):
        dot = np.clip(np.dot(z_axis, n), -1.0, 1.0)
        if dot > 0.9999:
            quats[i] = [1, 0, 0, 0]
        elif dot < -0.9999:
            quats[i] = [0, 1, 0, 0]
        else:
            axis = np.cross(z_axis, n)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            angle = np.arccos(dot)
            w = np.cos(angle / 2)
            s = np.sin(angle / 2)
            quats[i] = [w, axis[0] * s, axis[1] * s, axis[2] * s]

    return quats


def normal_to_quaternion_batch(normals: np.ndarray) -> np.ndarray:
    """Vectorized: convert normal vectors to quaternions (wxyz).

    Maps [0, 0, 1] to each target normal.
    """
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
    N = len(normals)
    z = np.array([0.0, 0.0, 1.0])

    dots = np.clip(normals @ z, -1.0, 1.0)  # (N,)

    axes = np.cross(np.tile(z, (N, 1)), normals)  # (N, 3)
    axes_norm = np.linalg.norm(axes, axis=-1, keepdims=True) + 1e-8
    axes = axes / axes_norm

    angles = np.arccos(dots)  # (N,)
    w = np.cos(angles / 2)
    s = np.sin(angles / 2)

    quats = np.stack([w, axes[:, 0] * s, axes[:, 1] * s, axes[:, 2] * s], axis=-1)

    # Handle degenerate cases
    aligned = dots > 0.9999
    quats[aligned] = [1, 0, 0, 0]
    anti_aligned = dots < -0.9999
    quats[anti_aligned] = [0, 1, 0, 0]

    return quats.astype(np.float32)


def compute_superpixel_features(
    color: np.ndarray,
    depth: np.ndarray,
    normals: np.ndarray,
    mask: np.ndarray,
    depth_weight: float = 0.5,
    normal_weight: float = 0.3,
) -> np.ndarray:
    """Build a multi-channel image for SLIC: [R, G, B, depth, nx, ny, nz].

    All channels are normalized to [0, 1] for balanced contribution.
    """
    H, W = depth.shape

    color_norm = color.astype(np.float32) / 255.0 if color.max() > 1 else color.astype(np.float32)

    depth_f = depth.astype(np.float32).copy()
    valid = mask & (depth_f > 0)
    if valid.any():
        d_min, d_max = depth_f[valid].min(), depth_f[valid].max()
        if d_max > d_min:
            depth_f = (depth_f - d_min) / (d_max - d_min)
        else:
            depth_f = np.zeros_like(depth_f)
    depth_f[~valid] = 0
    depth_ch = depth_f[..., None] * depth_weight

    normals_f = normals.astype(np.float32).copy()
    normals_f = (normals_f + 1) / 2  # [-1,1] → [0,1]
    normals_f[~mask] = 0
    normals_ch = normals_f * normal_weight

    features = np.concatenate([color_norm, depth_ch, normals_ch], axis=-1)
    return features


def run_superpixel_segmentation(
    features: np.ndarray,
    mask: np.ndarray,
    n_segments: int = 1000,
    compactness: float = 20.0,
) -> np.ndarray:
    """Run SLIC on the multi-channel feature image.

    Returns:
        labels: (H, W) integer label map. Masked pixels get label -1.
    """
    labels = slic(
        features,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        enforce_connectivity=True,
        mask=mask,
    )
    labels[~mask] = -1
    return labels


def aggregate_superpixels(
    labels: np.ndarray,
    pts3d: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    min_pixels: int = 4,
) -> dict:
    """Aggregate per-pixel data into per-superpixel primitives.

    For each superpixel:
    - position: mean 3D point
    - normal: mean normal (renormalized)
    - color: mean RGB
    - scale: estimated from spatial extent in 3D
    - pixel_count: number of pixels

    Returns dict with arrays of shape (S, ...) where S is the number of valid superpixels.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    positions = []
    sp_normals = []
    sp_colors = []
    scales = []
    pixel_counts = []

    H, W = labels.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]

    for label in unique_labels:
        lmask = (labels == label) & mask
        count = lmask.sum()
        if count < min_pixels:
            continue

        pts = pts3d[lmask]              # (K, 3)
        nrm = normals[lmask]            # (K, 3)
        col = colors[lmask]             # (K, 3)
        dep = depth[lmask]              # (K,)

        mean_pos = pts.mean(axis=0)
        mean_normal = nrm.mean(axis=0)
        nn = np.linalg.norm(mean_normal)
        if nn > 1e-6:
            mean_normal = mean_normal / nn
        else:
            mean_normal = np.array([0, 0, 1], dtype=np.float32)
        mean_color = col.mean(axis=0)

        # Estimate scale from 3D spatial extent of the superpixel
        if len(pts) > 1:
            centered = pts - mean_pos
            dists = np.linalg.norm(centered, axis=-1)
            scale_3d = np.percentile(dists, 90)
        else:
            mean_depth = dep.mean()
            scale_3d = mean_depth / min(fx, fy)

        scale_3d = max(scale_3d, 1e-5)

        positions.append(mean_pos)
        sp_normals.append(mean_normal)
        sp_colors.append(mean_color)
        scales.append(scale_3d)
        pixel_counts.append(count)

    return {
        "positions": np.array(positions, dtype=np.float32),
        "normals": np.array(sp_normals, dtype=np.float32),
        "colors": np.array(sp_colors, dtype=np.float32),
        "scales": np.array(scales, dtype=np.float32),
        "pixel_counts": np.array(pixel_counts, dtype=np.int32),
    }


def merge_multiview_superpixels(
    all_sp: list,
    voxel_size: float = 0.01,
) -> dict:
    """Merge superpixels from multiple views, deduplicating overlapping ones.

    Uses voxel-based deduplication: within each voxel, keeps the superpixel
    with the highest pixel count (best observed).
    """
    positions = np.concatenate([sp["positions"] for sp in all_sp], axis=0)
    normals = np.concatenate([sp["normals"] for sp in all_sp], axis=0)
    colors = np.concatenate([sp["colors"] for sp in all_sp], axis=0)
    scales = np.concatenate([sp["scales"] for sp in all_sp], axis=0)
    pixel_counts = np.concatenate([sp["pixel_counts"] for sp in all_sp], axis=0)

    if len(positions) == 0:
        return {
            "positions": np.zeros((0, 3), dtype=np.float32),
            "normals": np.zeros((0, 3), dtype=np.float32),
            "colors": np.zeros((0, 3), dtype=np.float32),
            "scales": np.zeros((0,), dtype=np.float32),
        }

    # Voxel-based deduplication
    voxel_coords = np.floor(positions / voxel_size).astype(np.int64)
    voxel_keys = (
        voxel_coords[:, 0] * 1000003
        + voxel_coords[:, 1] * 1000033
        + voxel_coords[:, 2] * 1000037
    )

    unique_keys, inverse = np.unique(voxel_keys, return_inverse=True)
    n_unique = len(unique_keys)

    merged_positions = np.zeros((n_unique, 3), dtype=np.float32)
    merged_normals = np.zeros((n_unique, 3), dtype=np.float32)
    merged_colors = np.zeros((n_unique, 3), dtype=np.float32)
    merged_scales = np.zeros(n_unique, dtype=np.float32)
    merged_weights = np.zeros(n_unique, dtype=np.float32)

    for i in range(len(positions)):
        idx = inverse[i]
        w = float(pixel_counts[i])
        merged_positions[idx] += positions[i] * w
        merged_normals[idx] += normals[i] * w
        merged_colors[idx] += colors[i] * w
        merged_scales[idx] += scales[i] * w
        merged_weights[idx] += w

    valid = merged_weights > 0
    merged_positions[valid] /= merged_weights[valid, None]
    merged_normals[valid] /= merged_weights[valid, None]
    nn = np.linalg.norm(merged_normals, axis=-1, keepdims=True) + 1e-8
    merged_normals /= nn
    merged_colors[valid] /= merged_weights[valid, None]
    merged_scales[valid] /= merged_weights[valid]

    return {
        "positions": merged_positions[valid],
        "normals": merged_normals[valid],
        "colors": merged_colors[valid],
        "scales": merged_scales[valid],
    }


def superpixels_to_2dgs_params(sp: dict, init_opacity: float = 0.5) -> dict:
    """Convert superpixel aggregates to 2DGS parameter format.

    Returns a dict compatible with gsplat's 2DGS trainer:
        means: (N, 3)
        quats: (N, 4) — wxyz, derived from normal direction
        scales: (N, 3) — [s_u, s_v, ~0] for flat disks
        opacities: (N,)
        colors_rgb: (N, 3) — raw RGB, to be converted to SH
    """
    positions = sp["positions"]
    normals_arr = sp["normals"]
    colors = sp["colors"]
    scale_vals = sp["scales"]
    N = len(positions)

    quats = normal_to_quaternion_batch(normals_arr)

    # 2DGS: two tangential scales, third near-zero for a flat disk
    log_scales = np.log(np.clip(scale_vals, 1e-6, None))
    scales_3 = np.stack([
        log_scales,
        log_scales,
        np.full(N, np.log(1e-6)),  # near-zero thickness
    ], axis=-1).astype(np.float32)

    opacities = np.full(N, np.log(init_opacity / (1.0 - init_opacity)),
                        dtype=np.float32)  # logit

    colors_rgb = np.clip(colors, 0, 1).astype(np.float32)

    return {
        "means": positions,
        "quats": quats,
        "scales": scales_3,
        "opacities": opacities,
        "colors_rgb": colors_rgb,
    }


def save_2dgs_init(params: dict, output_path: str):
    """Save 2DGS initialization as a .pt file compatible with gsplat."""
    data = {k: torch.from_numpy(v) for k, v in params.items()}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved 2DGS init ({data['means'].shape[0]} primitives) → {output_path}")


def save_point_cloud_ply(params: dict, output_path: str):
    """Save as a PLY point cloud for visualization."""
    from plyfile import PlyData, PlyElement

    positions = params["means"]
    colors = (np.clip(params["colors_rgb"], 0, 1) * 255).astype(np.uint8)
    N = len(positions)

    vertices = np.zeros(N, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertices['x'] = positions[:, 0]
    vertices['y'] = positions[:, 1]
    vertices['z'] = positions[:, 2]
    vertices['red'] = colors[:, 0]
    vertices['green'] = colors[:, 1]
    vertices['blue'] = colors[:, 2]

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Saved PLY point cloud ({N} points) → {output_path}")


def process_scene(
    npz_path: str,
    output_dir: str,
    n_segments: int = 1000,
    compactness: float = 20.0,
    depth_weight: float = 0.5,
    normal_weight: float = 0.3,
    voxel_size: float = 0.01,
    min_pixels: int = 4,
    init_opacity: float = 0.5,
):
    """Full pipeline: load MapAnything outputs → superpixel 2DGS init."""
    print(f"Loading MapAnything outputs from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    depths = data["depth"]          # (V, H, W)
    normals = data["normals"]       # (V, H, W, 3)
    poses = data["poses"]           # (V, 4, 4)
    intrinsics = data["intrinsics"] # (V, 3, 3)
    colors = data["colors"]         # (V, H, W, 3)
    masks = data["masks"]           # (V, H, W)
    pts3d = data["pts3d"]           # (V, H, W, 3)

    n_views = len(depths)
    print(f"Processing {n_views} views, target {n_segments} superpixels/view")

    all_superpixels = []

    for v in range(n_views):
        features = compute_superpixel_features(
            colors[v], depths[v], normals[v], masks[v],
            depth_weight=depth_weight,
            normal_weight=normal_weight,
        )

        labels = run_superpixel_segmentation(
            features, masks[v],
            n_segments=n_segments,
            compactness=compactness,
        )

        sp = aggregate_superpixels(
            labels, pts3d[v], normals[v], colors[v], depths[v],
            masks[v], intrinsics[v],
            min_pixels=min_pixels,
        )

        n_sp = len(sp["positions"])
        n_px = masks[v].sum()
        ratio = n_px / max(n_sp, 1)
        print(f"  View {v}: {n_sp} superpixels from {n_px} valid pixels "
              f"(reduction ratio: {ratio:.1f}x)")

        all_superpixels.append(sp)

    # Merge across views with deduplication
    print(f"Merging superpixels across {n_views} views (voxel_size={voxel_size})...")
    merged = merge_multiview_superpixels(all_superpixels, voxel_size=voxel_size)
    total_before = sum(len(sp["positions"]) for sp in all_superpixels)
    total_after = len(merged["positions"])
    print(f"  Before dedup: {total_before}, after: {total_after} "
          f"(removed {total_before - total_after} duplicates)")

    # Convert to 2DGS parameters
    params = superpixels_to_2dgs_params(merged, init_opacity=init_opacity)

    os.makedirs(output_dir, exist_ok=True)

    save_2dgs_init(params, os.path.join(output_dir, "splat_init.pt"))
    save_point_cloud_ply(params, os.path.join(output_dir, "splat_init.ply"))

    # Save per-view segmentation info for debugging
    np.savez_compressed(
        os.path.join(output_dir, "superpixel_info.npz"),
        n_superpixels_per_view=[len(sp["positions"]) for sp in all_superpixels],
        n_merged=total_after,
        n_segments_requested=n_segments,
        compactness=compactness,
        voxel_size=voxel_size,
    )

    print(f"\nDone! {total_after} 2DGS primitives initialized.")
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Superpixel-based 2DGS initialization from MapAnything"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to mapanything_outputs.npz")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for 2DGS init files")
    parser.add_argument("--n_segments", type=int, default=1000,
                        help="Target superpixels per view")
    parser.add_argument("--compactness", type=float, default=20.0,
                        help="SLIC compactness (higher = more regular shape)")
    parser.add_argument("--depth_weight", type=float, default=0.5,
                        help="Weight for depth channel in SLIC features")
    parser.add_argument("--normal_weight", type=float, default=0.3,
                        help="Weight for normal channels in SLIC features")
    parser.add_argument("--voxel_size", type=float, default=0.01,
                        help="Voxel size for cross-view deduplication")
    parser.add_argument("--min_pixels", type=int, default=4,
                        help="Minimum pixels per superpixel")
    parser.add_argument("--init_opacity", type=float, default=0.5,
                        help="Initial opacity for 2DGS primitives")
    args = parser.parse_args()

    process_scene(
        args.input, args.output_dir,
        n_segments=args.n_segments,
        compactness=args.compactness,
        depth_weight=args.depth_weight,
        normal_weight=args.normal_weight,
        voxel_size=args.voxel_size,
        min_pixels=args.min_pixels,
        init_opacity=args.init_opacity,
    )


if __name__ == "__main__":
    main()

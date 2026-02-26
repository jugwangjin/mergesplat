"""
Run MapAnything inference on a set of images.
Saves per-view depth, normals, poses, intrinsics, and colors.

Usage:
    python scripts/run_mapanything.py \
        --image_dir dataset/kitchen/images \
        --output_dir dataset/kitchen
"""

import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.geometry import (
    depthmap_to_camera_frame,
    depthmap_to_world_frame,
    points_to_normals,
)
from mapanything.utils.image import load_images


def compute_normals_from_points(pts3d_np, mask_np):
    """Compute per-pixel normals from a 3D point map."""
    normals = points_to_normals(pts3d_np, mask=mask_np)
    return normals


def main():
    parser = argparse.ArgumentParser(description="Run MapAnything inference")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--apache", action="store_true",
                        help="Use Apache 2.0 model")
    parser.add_argument("--minibatch_size", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    print(f"Loading model: {model_name}")
    model = MapAnything.from_pretrained(model_name).to(device)

    print(f"Loading images from: {args.image_dir}")
    views = load_images(args.image_dir)
    print(f"Loaded {len(views)} views")

    print("Running inference...")
    predictions = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=args.minibatch_size,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
    )
    print("Inference complete.")

    os.makedirs(args.output_dir, exist_ok=True)

    all_data = {
        "depth": [],
        "normals": [],
        "poses": [],
        "intrinsics": [],
        "colors": [],
        "masks": [],
        "pts3d": [],
        "image_names": [],
    }

    for view_idx, pred in enumerate(predictions):
        depth_z = pred["depth_z"][0].squeeze(-1).cpu()         # (H, W)
        intrinsics = pred["intrinsics"][0].cpu()                # (3, 3)
        camera_pose = pred["camera_poses"][0].cpu()             # (4, 4)
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)
        img = pred["img_no_norm"][0].cpu().numpy()              # (H, W, 3)
        conf = pred["conf"][0].cpu().numpy()                    # (H, W)

        pts3d_world, valid_mask = depthmap_to_world_frame(
            depth_z, intrinsics, camera_pose
        )
        pts3d_np = pts3d_world.numpy()
        combined_mask = mask & valid_mask.numpy()

        normals = compute_normals_from_points(pts3d_np, combined_mask)

        all_data["depth"].append(depth_z.numpy())
        all_data["normals"].append(normals)
        all_data["poses"].append(camera_pose.numpy())
        all_data["intrinsics"].append(intrinsics.numpy())
        all_data["colors"].append(img)
        all_data["masks"].append(combined_mask)
        all_data["pts3d"].append(pts3d_np)

        instance = views[view_idx].get("instance", f"view_{view_idx:04d}")
        all_data["image_names"].append(str(instance))

        print(f"  View {view_idx}: depth {depth_z.shape}, "
              f"normals {normals.shape}, "
              f"valid pixels: {combined_mask.sum()}/{combined_mask.size}")

    out_path = os.path.join(args.output_dir, "mapanything_outputs.npz")
    np.savez_compressed(
        out_path,
        depth=np.array(all_data["depth"]),
        normals=np.array(all_data["normals"]),
        poses=np.array(all_data["poses"]),
        intrinsics=np.array(all_data["intrinsics"]),
        colors=np.array(all_data["colors"]),
        masks=np.array(all_data["masks"]),
        pts3d=np.array(all_data["pts3d"]),
        image_names=np.array(all_data["image_names"]),
    )
    print(f"Saved outputs to: {out_path}")


if __name__ == "__main__":
    main()

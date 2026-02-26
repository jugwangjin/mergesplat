"""
End-to-end pipeline: MapAnything inference → superpixel-based 2DGS initialization.

Usage:
    python scripts/pipeline.py \
        --image_dir dataset/kitchen/images \
        --output_dir dataset/kitchen \
        --n_segments 1000
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="MapAnything → Superpixel 2DGS pipeline"
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--apache", action="store_true")
    parser.add_argument("--n_segments", type=int, default=1000)
    parser.add_argument("--compactness", type=float, default=20.0)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip MapAnything inference (use existing outputs)")
    args = parser.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(args.output_dir, "mapanything_outputs.npz")

    # Step 1: MapAnything inference
    if not args.skip_inference:
        print("=" * 60)
        print("Step 1: Running MapAnything inference")
        print("=" * 60)
        cmd = [
            sys.executable, os.path.join(scripts_dir, "run_mapanything.py"),
            "--image_dir", args.image_dir,
            "--output_dir", args.output_dir,
        ]
        if args.apache:
            cmd.append("--apache")
        subprocess.run(cmd, check=True)
    else:
        print(f"Skipping inference, using existing: {npz_path}")

    # Step 2: Superpixel-based 2DGS initialization
    print()
    print("=" * 60)
    print("Step 2: Superpixel-based 2DGS initialization")
    print("=" * 60)
    cmd = [
        sys.executable, os.path.join(scripts_dir, "superpixel_2dgs.py"),
        "--input", npz_path,
        "--output_dir", os.path.join(args.output_dir, "splat_init"),
        "--n_segments", str(args.n_segments),
        "--compactness", str(args.compactness),
        "--voxel_size", str(args.voxel_size),
    ]
    subprocess.run(cmd, check=True)

    print()
    print("=" * 60)
    print("Pipeline complete!")
    print(f"  MapAnything outputs: {npz_path}")
    print(f"  2DGS init: {os.path.join(args.output_dir, 'splat_init', 'splat_init.pt')}")
    print(f"  Point cloud: {os.path.join(args.output_dir, 'splat_init', 'splat_init.ply')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

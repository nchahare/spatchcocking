"""
preprocess_tiff.py

Post-process a 3D lumen segmentation mask from 3D Slicer and export a
triangulated surface mesh.

This script performs:
  1. Load a binary TIFF mask (lumen = 1, background = 0)
  2. Fill internal holes
  3. Remove small isolated objects
  4. Apply Gaussian smoothing to reduce voxel staircase artifacts
  5. Extract the inner lumen surface via marching cubes
  6. Export the mesh as .ply (and optionally .stl)

Usage:
    python preprocess_tiff.py --input path/to/mask.tiff --output path/to/output.ply

Requirements:
    pip install tifffile scipy scikit-image trimesh numpy
"""

import argparse
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure, morphology, filters
import trimesh
import os


def load_mask(path):
    mask = tifffile.imread(path)
    if mask.ndim == 4:
        mask = mask[0] if mask.shape[0] < mask.shape[-1] else mask[:, :, :, 0]
    return (mask > 0).astype(np.uint8)


def postprocess_mask(mask, min_size=500, sigma=1.0):
    # Fill holes in each z-slice, then globally
    filled = ndimage.binary_fill_holes(mask)

    # Remove objects smaller than min_size voxels
    cleaned = morphology.remove_small_objects(filled.astype(bool), min_size=min_size)

    # Smooth to reduce staircase artifacts before meshing
    smoothed = filters.gaussian(cleaned.astype(float), sigma=sigma)

    return smoothed, cleaned.astype(np.uint8)


def extract_mesh(smoothed_mask, level=0.5, voxel_size=(1.0, 1.0, 1.0)):
    verts, faces, normals, _ = measure.marching_cubes(
        smoothed_mask,
        level=level,
        spacing=voxel_size,
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh


def save_mesh(mesh, output_path):
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".ply":
        mesh.export(output_path, file_type="ply")
    elif ext == ".stl":
        mesh.export(output_path, file_type="stl")
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .ply or .stl")
    print(f"Mesh saved: {output_path}  ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")


def main():
    parser = argparse.ArgumentParser(description="Post-process lumen mask and export surface mesh")
    parser.add_argument("--input", required=True, help="Path to binary TIFF mask")
    parser.add_argument("--output", required=True, help="Output mesh path (.ply or .stl)")
    parser.add_argument("--min-size", type=int, default=500, help="Minimum object size in voxels (default: 500)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma (default: 1.0)")
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=("Z", "Y", "X"),
                        help="Voxel size in µm, ZYX order (default: 1.0 1.0 1.0)")
    args = parser.parse_args()

    print(f"Loading mask: {args.input}")
    mask = load_mask(args.input)
    print(f"  Mask shape: {mask.shape}, non-zero voxels: {mask.sum()}")

    print("Post-processing...")
    smoothed, cleaned = postprocess_mask(mask, min_size=args.min_size, sigma=args.sigma)
    print(f"  After cleaning: {cleaned.sum()} voxels")

    print("Extracting surface mesh via marching cubes...")
    mesh = extract_mesh(smoothed, voxel_size=tuple(args.voxel_size))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_mesh(mesh, args.output)


if __name__ == "__main__":
    main()

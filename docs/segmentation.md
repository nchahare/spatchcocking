# Segmentation workflow

This document describes the image segmentation protocol used to generate the inner (lumen) and outer (basal) surface meshes from confocal image stacks. These meshes are the input to all downstream analyses (notebooks 01–05).

---

## Software

| Step | Software |
|---|---|
| Image format conversion & preprocessing | [Fiji/ImageJ](https://fiji.sc/) |
| Volumetric segmentation | [3D Slicer](https://www.slicer.org/) 5.x |
| Post-processing & mesh export | Python — `tifffile`, `scipy`, `scikit-image`, `trimesh` |
| Visualisation (optional) | [napari](https://napari.org/) |

```bash
pip install tifffile scipy scikit-image trimesh numpy
```

The `spatchcocking` pip package does not include napari or 3D Slicer; these must be installed separately.

---

## Protocol

### 1. Image preprocessing (Fiji)

Raw confocal `.czi` files were converted to `.tiff` and downsampled four-fold in XY using the **Scale** function in Fiji/ImageJ (Schindelin et al., 2012):

- **Image → Scale**: X scale = 0.25, Y scale = 0.25, Z scale = 1.0, interpolation = Bilinear
- The DAPI channel was isolated and gamma-corrected (γ = 0.5) to enhance contrast at tissue boundaries: **Process → Math → Gamma**

The resulting single-channel `.tiff` stacks (~256 × 256 pixels per slice, 400–500 slices) were used as input to segmentation.

---

### 2. Volumetric segmentation (3D Slicer)

Two segments were generated per embryo: `Inner` (lumen boundary) and `Outer` (basal surface of the neuroepithelium). Segmentation was performed in [3D Slicer](https://www.slicer.org/) using the **Segment Editor** module.

1. Import the `.tiff` stack: **File → Add Data**
2. Open **Segment Editor** and add two segments: `Inner` and `Outer`
3. Generate initial masks using the **Local Threshold** effect:
   - The neuroepithelial wall is densely packed with DAPI-stained nuclei and appears
     brighter than the surrounding mesenchyme; local thresholding reliably detects it
     as a continuous bright band
   - Set the neighbourhood size to ~20 voxels to capture local intensity variation
   - Draw inside the neuroepithelial wall for the `Inner` segment and just outside
     the wall for the `Outer` segment
4. Refine with the **Paint** and **Erase** tools as needed:
   - Close any gaps at the rostral and caudal ends of the tube
   - Remove signal from the floor plate or spinal cord if present
5. Apply **Fill holes** to close interior voids
6. Export both segments: **Segmentations → Export to file → .tiff**, preserving the original voxel spacing

---

### 3. Mask post-processing and mesh export (Python)

Run `preprocess_tiff.py` (below) on each exported mask. The script fills holes, removes small objects, smooths voxel staircase artefacts with a Gaussian filter, and extracts a triangulated surface via marching cubes.

```bash
# Inner (lumen) surface
python preprocess_tiff.py \
    --input  Inner_Mask.tiff \
    --output lumen.ply \
    --voxel-size 4.5 5.535 5.535

# Outer (basal) surface
python preprocess_tiff.py \
    --input  Outer_Mask.tiff \
    --output basal.ply \
    --voxel-size 4.5 5.535 5.535
```

`--voxel-size` takes Z Y X spacings in µm. The values above match the example dataset (HH20, 4.5 µm z-step, 5.535 µm/pixel xy); adjust for your microscope calibration.

```python
"""
preprocess_tiff.py

Post-process a 3D segmentation mask and export a triangulated surface mesh.

Steps:
  1. Load binary TIFF mask (tissue = 1, background = 0)
  2. Fill internal holes
  3. Remove small isolated objects
  4. Gaussian smoothing to reduce voxel staircase artefacts
  5. Extract surface via marching cubes
  6. Export as .ply (or .stl)

Usage:
    python preprocess_tiff.py --input mask.tiff --output surface.ply
    python preprocess_tiff.py --input mask.tiff --output surface.ply \
        --voxel-size 4.5 5.535 5.535 --sigma 1.5 --min-size 1000

Requirements:
    pip install tifffile scipy scikit-image trimesh numpy
"""

import argparse
import os
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure, morphology, filters
import trimesh


def load_mask(path):
    mask = tifffile.imread(path)
    if mask.ndim == 4:
        mask = mask[0] if mask.shape[0] < mask.shape[-1] else mask[:, :, :, 0]
    return (mask > 0).astype(np.uint8)


def postprocess_mask(mask, min_size=500, sigma=1.0):
    filled  = ndimage.binary_fill_holes(mask)
    cleaned = morphology.remove_small_objects(filled.astype(bool), min_size=min_size)
    smoothed = filters.gaussian(cleaned.astype(float), sigma=sigma)
    return smoothed, cleaned.astype(np.uint8)


def extract_mesh(smoothed_mask, level=0.5, voxel_size=(1.0, 1.0, 1.0)):
    verts, faces, normals, _ = measure.marching_cubes(
        smoothed_mask, level=level, spacing=voxel_size   # (Z, Y, X) in µm
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)


def save_mesh(mesh, output_path):
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in (".ply", ".stl"):
        raise ValueError(f"Unsupported format: {ext}. Use .ply or .stl")
    mesh.export(output_path, file_type=ext[1:])
    print(f"Saved: {output_path}  ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")


def main():
    parser = argparse.ArgumentParser(description="Post-process mask and export surface mesh")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--min-size",   type=int,   default=500)
    parser.add_argument("--sigma",      type=float, default=1.0)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=("Z", "Y", "X"))
    args = parser.parse_args()

    mask = load_mask(args.input)
    print(f"Loaded: {args.input}  shape={mask.shape}  non-zero={mask.sum()}")

    smoothed, cleaned = postprocess_mask(mask, min_size=args.min_size, sigma=args.sigma)
    print(f"After cleaning: {cleaned.sum()} voxels")

    mesh = extract_mesh(smoothed, voxel_size=tuple(args.voxel_size))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_mesh(mesh, args.output)


if __name__ == "__main__":
    main()
```

---

## Notes

### When local thresholding fails

Local thresholding works well where the neuroepithelium is clearly brighter than the surrounding tissue. In regions where the boundary is ambiguous (e.g., near the floor plate, at the caudal end, or where signal is low), manual correction with the **Paint** tool in 3D Slicer is necessary. Inspect every 10–20 slices and correct before running **Fill holes**.

### Sparse manual annotation in napari

For very large volumes where 3D Slicer is slow, an alternative is to annotate a sparse subset of slices in [napari](https://napari.org/) and then interpolate the gaps using the [napari-label-interpolator](https://github.com/haesleinhuepf/napari-label-interpolator) plugin:

1. Transpose the volume from `(Z, Y, X)` to `(Y, Z, X)` so that the rostral-caudal axis becomes the slicing axis
2. Extract ~50 evenly spaced Y-slices; save the index array (`y_indices.npy`) for reconstruction
3. Annotate `Inner` and `Outer` contours on each sparse slice using the **Paint** or **Polygon** tool
4. Place annotated slices back into the full-size volume at their original Y positions
5. Run **Plugins → napari-label-interpolator → Interpolate Labels** (one mask at a time)
6. Transpose the interpolated mask back to `(Z, Y, X)` before passing to `preprocess_tiff.py`

---

## Output

Each embryo produces two mesh files passed to notebook 01:

- `lumen.ply` — inner lumen surface (fluid–neuroepithelium interface)
- `basal.ply` — outer basal surface (neuroepithelium–mesenchyme interface)

---

## References

Schindelin, J., Arganda-Carreras, I., Frise, E., et al. (2012). Fiji: an open-source platform for biological-image analysis. *Nat. Methods* 9:676–682.

Fedorov, A., Beichel, R., Kalpathy-Cramer, J., et al. (2012). 3D Slicer as an image computing platform for the Quantitative Imaging Network. *Magn. Reson. Imaging* 30:1323–1341.

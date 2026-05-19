# Segmentation workflow

This document describes the image segmentation pipeline used to generate 3D
surface meshes from confocal image stacks.  The pipeline has two parts:

- **Interactive volumetric segmentation** — performed in
  [3D Slicer](https://www.slicer.org/) and
  [napari](https://napari.org/); not automated.
- **Mask post-processing and mesh export** — scripted with standard Python
  packages (tifffile, scipy, scikit-image, trimesh); code is provided below.

The output of this pipeline is the `.ply` mesh files used as input to all
downstream spatchcocking analyses (notebooks 01–05).

---

## Software requirements

| Step | Software | Version used |
|---|---|---|
| Image format conversion & preprocessing | [Fiji/ImageJ](https://fiji.sc/) | 2.x |
| 3D visualization (optional) | [napari](https://napari.org/) | 0.5.x |
| Volumetric segmentation | [3D Slicer](https://www.slicer.org/) | 5.x |
| Post-processing & mesh export | Python (see script below) | — |

**The `spatchcocking` pip package does NOT include napari or 3D Slicer.**
Those are installed separately and used interactively.

Python dependencies for the post-processing script:

```
pip install tifffile scipy scikit-image trimesh numpy
```

---

## Step-by-step protocol

### 1. Raw image preprocessing (Fiji)

Raw confocal `.czi` files were converted to `.tiff` format and downsampled
four-fold using the **Scale** function in Fiji/ImageJ (Schindelin et al., 2012):

- **Analyze → Scale**: X scale = 0.25, Y scale = 0.25, Z scale = 1.0,
  interpolation = Bilinear
- DAPI channel was isolated and gamma-corrected (γ = 0.5) to enhance
  structural contrast at tissue boundaries:
  **Process → Math → Gamma**

The resulting single-channel `.tiff` stacks (~256 × 256 pixels per slice,
400–500 slices) are the input to segmentation.

---

### 2. 3D Slicer segmentation

Volumetric segmentation was performed in [3D Slicer](https://www.slicer.org/)
using the **Segment Editor** module:

1. Import the `.tiff` stack: **File → Add Data** → select the `.tiff` file
2. Open **Segment Editor**
3. Add two segments: `lumen` and `neuroepithelium`
4. Generate an initial mask using the **Threshold** effect:
   - Set threshold range to capture lumen pixels (bright DAPI nuclei define
     the outer boundary; the fluid-filled lumen is darker)
   - Use **Local thresholding** (neighbourhood size ~20 voxels) rather than
     global threshold
5. Refine manually using the **Paint** and **Erase** tools to:
   - Close gaps at the caudal end of the neural tube (posterior to the hindbrain)
   - Remove extraembryonic staining outside the neural tube
   - Separate lumen from spinal cord when needed
6. Apply **Fill holes** to close any interior voids in the lumen mask
7. Export the lumen segment:
   **Segmentations → Export to file** → select `.tiff`, same voxel spacing
   as input

---

### 3. Sparse annotation in napari (alternative to 3D Slicer)

For large volumes where 3D Slicer is slow, an efficient alternative is to
annotate a sparse subset of slices in napari and then interpolate the gaps
automatically using the
[napari-label-interpolator](https://github.com/haesleinhuepf/napari-label-interpolator)
plugin.

**Install the plugin:**
```bash
pip install napari-label-interpolator
```

**Workflow:**

1. **Reslice the volume** — transpose from `(Z, Y, X)` to `(Y, Z, X)` so
   that the rostral-caudal axis (Y) becomes the primary slicing axis:
   ```python
   import tifffile, numpy as np
   vol = tifffile.imread("HH20_DAPI.tiff")          # shape (Z, Y, X)
   vol_t = np.transpose(vol, (1, 0, 2))             # shape (Y, Z, X)
   tifffile.imwrite("HH20_DAPI_transposed.tiff", vol_t)
   ```

2. **Select sparse slice indices** — choose ~50 evenly spaced Y-positions
   to annotate; save the index array so you can reconstruct later:
   ```python
   n_slices = 50
   y_indices = np.linspace(0, vol_t.shape[0] - 1, n_slices, dtype=int)
   np.save("y_indices.npy", y_indices)
   sparse_vol = vol_t[y_indices]                    # shape (50, Z, X)
   tifffile.imwrite("HH20_DAPI_sparse.tiff", sparse_vol)
   ```

3. **Annotate in napari** — open `HH20_DAPI_sparse.tiff`, add a Labels
   layer, and draw the `Inner` (lumen) and `Outer` (basal) contours on
   each slice using the **Paint** or **Polygon** tool.
   - Save frequently — export labels as TIFF after every few slices.
   - Label `0` is the eraser.
   - Annotate `Inner` and `Outer` as separate label values (e.g., 1 and 2),
     or as separate label layers exported individually.

4. **Reconstruct full-resolution label volume** — place the annotated slices
   back into their original Y positions:
   ```python
   labels_sparse = tifffile.imread("HH20_labels_sparse.tiff")  # shape (50, Z, X)
   labels_full = np.zeros(vol_t.shape, dtype=np.uint8)
   for i, y in enumerate(y_indices):
       labels_full[y] = labels_sparse[i]
   tifffile.imwrite("HH20_labels_full.tiff", labels_full)
   ```

5. **Interpolate gaps** — open `HH20_labels_full.tiff` in napari and run
   **Plugins → napari-label-interpolator → Interpolate Labels**.
   - Interpolate one mask (Inner or Outer) at a time to avoid memory issues.
   - Close unused layers before interpolating.
   - Save the result as `HH20_lumen_mask.tiff` and `HH20_basal_mask.tiff`.

6. **Transpose back** to `(Z, Y, X)` before passing to the post-processing
   script:
   ```python
   mask = tifffile.imread("HH20_lumen_mask.tiff")   # shape (Y, Z, X)
   mask_zyx = np.transpose(mask, (1, 0, 2))         # shape (Z, Y, X)
   tifffile.imwrite("HH20_lumen_mask_zyx.tiff", mask_zyx)
   ```

7. **Validate** by toggling the mask layer over the raw image in napari to
   confirm alignment and biological accuracy before proceeding to meshing.

---

### 4. TIFF mask post-processing and mesh export (Python)

The script below loads the segmentation mask from 3D Slicer, cleans it, and
exports a triangulated surface mesh.  Save it as `preprocess_tiff.py` and run:

```bash
python preprocess_tiff.py \
    --input  path/to/lumen_mask.tiff \
    --output data/example/lumen.ply \
    --voxel-size 2.0 0.65 0.65
```

`--voxel-size` takes Z Y X spacings in µm (match your microscope calibration).

```python
"""
preprocess_tiff.py

Post-process a 3D lumen segmentation mask from 3D Slicer and export a
triangulated surface mesh.

Steps:
  1. Load binary TIFF mask (lumen = 1, background = 0)
  2. Fill internal holes
  3. Remove small isolated objects
  4. Gaussian smoothing to reduce voxel staircase artefacts
  5. Extract inner lumen surface via marching cubes
  6. Export as .ply (or .stl)

Usage:
    python preprocess_tiff.py --input mask.tiff --output lumen.ply
    python preprocess_tiff.py --input mask.tiff --output lumen.ply \
        --voxel-size 2.0 0.65 0.65 --sigma 1.5 --min-size 1000

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
    # Fill holes globally across all three axes
    filled = ndimage.binary_fill_holes(mask)

    # Remove isolated objects smaller than min_size voxels
    cleaned = morphology.remove_small_objects(
        filled.astype(bool), min_size=min_size
    )

    # Gaussian smoothing to reduce staircase artefacts before meshing
    smoothed = filters.gaussian(cleaned.astype(float), sigma=sigma)

    return smoothed, cleaned.astype(np.uint8)


def extract_mesh(smoothed_mask, level=0.5, voxel_size=(1.0, 1.0, 1.0)):
    verts, faces, normals, _ = measure.marching_cubes(
        smoothed_mask,
        level=level,
        spacing=voxel_size,   # (Z, Y, X) in µm
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
    print(f"Saved: {output_path}  ({len(mesh.vertices)} vertices, "
          f"{len(mesh.faces)} faces)")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process lumen mask and export surface mesh"
    )
    parser.add_argument("--input",  required=True,
                        help="Path to binary TIFF mask")
    parser.add_argument("--output", required=True,
                        help="Output mesh path (.ply or .stl)")
    parser.add_argument("--min-size", type=int, default=500,
                        help="Minimum object size in voxels (default: 500)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma (default: 1.0)")
    parser.add_argument("--voxel-size", type=float, nargs=3,
                        default=[1.0, 1.0, 1.0],
                        metavar=("Z", "Y", "X"),
                        help="Voxel size in µm, ZYX order (default: 1.0 1.0 1.0)")
    args = parser.parse_args()

    print(f"Loading mask: {args.input}")
    mask = load_mask(args.input)
    print(f"  Shape: {mask.shape},  non-zero voxels: {mask.sum()}")

    print("Post-processing...")
    smoothed, cleaned = postprocess_mask(
        mask, min_size=args.min_size, sigma=args.sigma
    )
    print(f"  After cleaning: {cleaned.sum()} voxels")

    print("Extracting surface via marching cubes...")
    mesh = extract_mesh(smoothed, voxel_size=tuple(args.voxel_size))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_mesh(mesh, args.output)


if __name__ == "__main__":
    main()
```

---

### 5. Visualisation in napari (optional)

[napari](https://napari.org/) was used during manual refinement to verify the
3D segmentation alongside the raw image:

```python
import napari
import tifffile
import numpy as np

viewer = napari.Viewer()
image = tifffile.imread("HH17_embryo1_DAPI.tiff")
mask  = tifffile.imread("HH17_embryo1_lumen_mask.tiff")
viewer.add_image(image, name="DAPI", colormap="gray")
viewer.add_labels(mask.astype(np.uint8), name="lumen mask")
napari.run()
```

---

## Output

The expected output for each embryo is:

- `lumen.ply` — inner lumen surface mesh (triangulated, ~10k–50k vertices
  after decimation)

This file is the input to notebook 01 (`01_mesh_generation.py`) and all
downstream analyses.

---

## References

Schindelin, J., Arganda-Carreras, I., Frise, E., et al. (2012). Fiji: an
open-source platform for biological-image analysis. *Nat. Methods* 9:676–682.

Fedorov, A., Beichel, R., Kalpathy-Cramer, J., et al. (2012). 3D Slicer as
an image computing platform for the Quantitative Imaging Network. *Magn.
Reson. Imaging* 30:1323–1341.

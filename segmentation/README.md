# Segmentation workflow

This folder documents the image segmentation pipeline used to generate the 3D surface meshes from confocal image stacks. The pipeline has two parts:

- **Preprocessing and mesh export** (`preprocess_tiff.py`) — runnable with standard Python packages (tifffile, scipy, trimesh).
- **Volumetric segmentation** — performed interactively in [3D Slicer](https://www.slicer.org/) and [napari](https://napari.org/); not automated.

The output of this pipeline is the `.ply` mesh files in `data/meshes/`.

---

## Software requirements

| Step | Software | Version used |
|---|---|---|
| Image format conversion & preprocessing | [Fiji/ImageJ](https://fiji.sc/) | 2.x |
| 3D visualization (optional) | [napari](https://napari.org/) | 0.5.x |
| Volumetric segmentation | [3D Slicer](https://www.slicer.org/) | 5.x |
| Post-processing & mesh export | Python (`preprocess_tiff.py`) | see requirements |

**The `spatchcocking` pip package does NOT include napari or 3D Slicer.** Those are installed separately and used interactively.

---

## Step-by-step protocol

### 1. Raw image preprocessing (Fiji)

Raw confocal `.czi` files were converted to `.tiff` format and downsampled four-fold using the **Scale** function in Fiji/ImageJ (Schindelin et al., 2012):

- **Analyze → Scale**: X scale = 0.25, Y scale = 0.25, Z scale = 1.0, interpolation = Bilinear
- DAPI channel was isolated and gamma-corrected (γ = 0.5) to enhance structural contrast at tissue boundaries: **Process → Math → Gamma**

The resulting single-channel `.tiff` stacks (~256 × 256 pixels per slice, 400–500 slices) are the input to segmentation.

### 2. 3D Slicer segmentation

Volumetric segmentation was performed in [3D Slicer](https://www.slicer.org/) using the **Segment Editor** module:

1. Import the `.tiff` stack: **File → Add Data** → select the `.tiff` file
2. Open **Segment Editor**
3. Add two segments: `lumen` and `neuroepithelium`
4. Generate an initial mask using **Threshold** effect:
   - Set threshold range to capture lumen pixels (bright DAPI nuclei define the outer boundary; the fluid-filled lumen is darker)
   - Use **Local thresholding** (neighborhood size ~20 voxels) rather than global threshold
5. Refine manually using the **Paint** and **Erase** tools to:
   - Close gaps at the caudal end of the neural tube (posterior to the hindbrain)
   - Remove extraembryonic staining outside the neural tube
   - Separate lumen from spinal cord when needed
6. Apply **Fill holes** to close any interior voids in the lumen mask
7. Export the lumen segment: **Segmentations → Export to file** → select `.tiff`, same voxel spacing as input

### 3. TIFF mask post-processing and mesh export (Python)

Run `preprocess_tiff.py` to:
- Load the segmentation mask from 3D Slicer
- Fill remaining holes and remove small isolated objects
- Apply smoothing to reduce voxel staircase artifacts
- Extract the inner lumen surface using marching cubes
- Export as `.ply` and/or `.stl`

```bash
python segmentation/preprocess_tiff.py --input path/to/lumen_mask.tiff --output data/meshes/HH17/HH17_embryo1_lumen.ply
```

### 4. Visualization in napari (optional)

[napari](https://napari.org/) was used during manual refinement to verify the 3D segmentation:

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

The expected output from this pipeline for each embryo is:
- `{stage}_embryo{n}_lumen.ply` — inner lumen surface mesh (triangulated, ~10k–50k vertices after decimation)

These files are the input to all downstream spatchcocking analyses in `notebooks/01` onwards.

---

## Reference

Schindelin, J., Arganda-Carreras, I., Frise, E., Kaynig, V., Longair, T., Pietzsch, T., et al. (2012). Fiji: an open-source platform for biological-image analysis. *Nat. Methods* 9:676–682.

Fedorov, A., Beichel, R., Kalpathy-Cramer, J., Finet, J.-C., Fillion-Robin, J.-C., Pujol, S., et al. (2012). 3D Slicer as an image computing platform for the Quantitative Imaging Network. *Magn. Reson. Imaging* 30:1323–1341.

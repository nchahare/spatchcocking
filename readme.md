# spatchcocking

**Supplementary data and code for Chahare, Imamura, and Nerurkar (2026)**

This repository contains the Python scripts, example data, and pre-computed meshes cited as supplementary materials in:

> Chahare N., Imamura C., and Nerurkar N.L. (2026). *Morphometric analysis reveals that the chick cranial neural tube expands as an active shell.* bioRxiv. https://doi.org/10.64898/2026.05.18.726048

---

## What this repository contains

The cranial neural tube is a highly curved, three-dimensional structure whose local tissue properties (curvature, thickness, mitotic activity) vary continuously across its surface. Comparing these properties across embryos or developmental stages requires a common coordinate system independent of specimen size and posture.

**Spatchcocking** is a 3D-to-2D geometric projection that maps the neural tube surface onto a standardized cylindrical coordinate frame — analogous to how a chicken is split along the spine and flattened for even cooking. The transformation:

1. Extracts the longitudinal medial axis via iterative Moving Least Squares smoothing
2. Defines local cross-sectional planes perpendicular to that axis
3. Uses anatomical dorsal landmarks to orient the azimuthal coordinate
4. Warps the 3D mesh into a straight cylindrical geometry (thin-plate spline)
5. Unwraps the cylinder into a 2D map, normalising the rostral-caudal length to [0, 1]

This enables direct spatial comparison of Gaussian curvature, Mean curvature, tissue thickness, and mitotic cell density across embryos and stages.

The `spatchcocking` Python package takes its name from this transformation, but it also bundles the broader set of vedo-based mesh utilities used throughout the analysis in this paper — mesh generation, curvature fitting, thickness mapping, and density interpolation.

```
  Raw confocal image
        ↓  (Fiji + 3D Slicer)              [docs/segmentation.md]
  3D binary mask
        ↓  (marching cubes)                [notebook 01]
   Surface mesh (.ply)
        ├──────────────────┬─────────────────────────┐
        ↓                  ↓                         ↓
  Curvature mapping  Thickness mapping    pHH3 density mapping
  [notebook 02]      [notebook 03]        [notebook 04]
        │                  │                         │
        └──────────────────┴─────────────────────────┘
                           ↓
               Spatchcocking: 3D → 2D               [notebook 05]
                           ↓
               2D heatmaps + statistics              [notebook 06]
```

---

## Repository map

| Folder / File | Contents | Paper figures |
|---|---|---|
| `src/spatchcocking/` | Installable Python package | All |
| [00 — Mesh viewer](https://jexpnimesh.com/spatchcocking/notebooks/00_mesh_viewer/) | 3D viewer for meshes | — |
| [01 — Mesh generation](https://jexpnimesh.com/spatchcocking/notebooks/01_mesh_generation/) | TIFF mask → surface mesh | Fig. 1c–d |
| [02 — Curvature mapping](https://jexpnimesh.com/spatchcocking/notebooks/02_curvature_mapping/) | Gaussian & Mean curvature | Figs. 3, 4 |
| [03 — Thickness mapping](https://jexpnimesh.com/spatchcocking/notebooks/03_thickness_mapping/) | Apico-basal thickness | Fig. 5 |
| [04 — pHH3 density mapping](https://jexpnimesh.com/spatchcocking/notebooks/04_pHH3_density_mapping/) | Mitotic density field | Fig. 6 |
| [05 — Spatchcocking](https://jexpnimesh.com/spatchcocking/notebooks/05_spatchcocking/) | 3D→2D projection | Fig. 2 |
| [06 — Figure plots](https://jexpnimesh.com/spatchcocking/notebooks/06_figure_plots/) | Statistical plots from CSV data | Figs. 1, 3–6 |
| [FEA notebook](https://jexpnimesh.com/spatchcocking/notebooks/notebook-fem/) | 2D plane-stress FEA | Fig. S9 |
| `data/meshes/` | 6 pre-computed lumen meshes (3×HH17, 3×HH20) | Figs. 1, 3–6 |
| `data/csv/` | Per-embryo measurements for figure reproduction | Figs. 1, 3–6 |
| `data/example/` | Single-embryo trial dataset (masks, meshes, npys, CSV) | Methods |
| `docs/segmentation.md` | Segmentation workflow (Fiji → 3D Slicer → napari) | Fig. 1, Methods |
| `finite_element/` | 2D plane-stress FEA scripts (separate environment) | Fig. S9 |

---

## Installation

```bash
pip install git+https://github.com/nchahare/spatchcocking
```

Requires Python ≥ 3.10.

**Note for Jupyter:** vedo requires an explicit backend setting at the top of each notebook:
```python
from vedo import settings
settings.default_backend = "vtk"
```

---

## Quick usage

```python
from vedo import settings
settings.default_backend = "vtk"

from spatchcocking import *   # re-exports vedo, numpy, os + all pipeline functions

# Load a pre-computed mesh and inspect scalar fields
mesh = Mesh("data/meshes/2025-09-18-13-02-HH17.vtk")
print(mesh.pointdata.keys())  # Mean_Curvature, Gauss_Curvature, K1, K2, thickness, phh3

# Spatchcock a basal mesh to 2D
basal   = Mesh("data/example/basal.ply")
axispts = np.load("data/example/2025-10-22-12-30-axis.npy")
endpts  = np.load("data/example/2025-10-22-12-30-endpts.npy")

axis_info     = getPlanes(basal, axispts, endpts)
deformed_mesh = getDeformedmesh2(basal, axis_info, namefile="2025-10-22-12-30")
radius, angle, height, dm = get_flatdata2(deformed_mesh)
norm_height, angle_degrees = normalize_values2(height, angle, shift_deg=-90)
```

See [`notebooks/00_mesh_viewer.ipynb`](notebooks/00_mesh_viewer.ipynb) for an interactive walkthrough of the pre-computed meshes, and notebooks 01–05 for the full processing pipeline.

---

## Data

**Example dataset** (`data/example/`): Binary TIFF masks, surface meshes, per-vertex measurement arrays, and the final spatchcocked CSV for one HH20 embryo. This is the trial dataset used throughout notebooks 01–05.

**Research meshes** (`data/meshes/`): 6 pre-computed VTK lumen meshes (3×HH17, 3×HH20) with all vertex arrays embedded (curvature, thickness, pHH3 density). See [`data/meshes/README.md`](data/meshes/README.md).

**Measurement CSV files** (`data/csv/`): Per-embryo measurements used to reproduce the paper figures directly via notebook 06, without re-running the pipeline. See [`data/csv/README.md`](data/csv/README.md).

**Full raw imaging data** are available from the corresponding author upon request.

---

## Segmentation workflow

Image segmentation (Fiji preprocessing → 3D Slicer volumetric segmentation → napari QC) requires software outside this Python package. See [`docs/segmentation.md`](docs/segmentation.md) for the full protocol.

---

## Finite element analysis

The 2D plane-stress FEA (Fig. S9) requires [SolidsPy](https://github.com/AppliedMechanics-EAFIT/SolidsPy) and runs in a separate conda environment. The folder contains `fem_plane_stress.py`, an interactive `notebook-fem.ipynb`, and the pre-computed input files (`mater.txt`, `nodes.txt`, `eles.txt`, `loads.txt`) so the model can be inspected and re-run without regenerating inputs from scratch. See [`finite_element/README.md`](finite_element/README.md).

---

## Generative AI declaration

Portions of the code and documentation in this repository were developed with the assistance of AI tools (Google Gemini and Anthropic Claude). All code was written, reviewed, and executed by N. Chahare, who verified that it produces the correct results. Claude additionally assisted with writing and editing the documentation files.

---

## Citation

If you use this code, please cite:

```
Chahare N., Imamura C., and Nerurkar N.L. (2026).
Morphometric analysis reveals that the chick cranial neural tube expands as an active shell.
bioRxiv. https://doi.org/10.64898/2026.05.18.726048
```

---

## Acknowledgements

This work relies on the following open-source packages:

- [vedo](https://vedo.embl.es/) — 3D mesh processing, curvature estimation, volumetric interpolation, and visualisation
- [NumPy](https://numpy.org/) — numerical arrays and computations
- [SciPy](https://scipy.org/) — scattered-data interpolation (`griddata`) and statistics
- [pandas](https://pandas.pydata.org/) — tabular data handling and CSV I/O
- [matplotlib](https://matplotlib.org/) — 2D plotting and figure export
- [seaborn](https://seaborn.pydata.org/) — heatmaps and statistical visualisation
- [trimesh](https://trimesh.org/) — marching cubes mesh extraction from binary masks

---

## License

MIT — see [LICENSE](LICENSE).

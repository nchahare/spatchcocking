# spatchcocking

**3D-to-2D morphometric mapping of curved biological tubes**

Companion code repository for:

> Chahare N., Imamura C., and Nerurkar N.L. (2026). *Morphometric analysis reveals that the chick cranial neural tube expands as an active shell.* [Journal, DOI — to be updated upon publication]

---

## Overview

The cranial neural tube is a highly curved, three-dimensional structure whose local tissue properties (curvature, thickness, mitotic activity) vary continuously across its surface. Comparing these properties across individual embryos or developmental stages requires a common coordinate system that is independent of specimen size and posture.

**Spatchcocking** is a 3D-to-2D geometric projection that maps the neural tube surface onto a standardized cylindrical coordinate frame — analogous to how a chicken is split along the spine and flattened for even cooking. The transformation:

1. Extracts the longitudinal medial axis via iterative Moving Least Squares smoothing
2. Defines local cross-sectional planes perpendicular to that axis
3. Uses anatomical dorsal landmarks to orient the azimuthal coordinate
4. Warps the 3D mesh into a straight cylindrical geometry (thin-plate spline)
5. Unwraps the cylinder into a 2D heatmap, normalizing the rostral-caudal length to [0, 1]

This enables direct node-by-node spatial comparison of Gaussian curvature, Mean curvature, tissue thickness, and mitotic cell density across embryos and stages.

```
Raw confocal image
        ↓  (Fiji + 3D Slicer)         [segmentation/]
  3D binary mask
        ↓  (trimesh marching cubes)   [notebook 01]
   Surface mesh (.ply)
        ↓  (spatchcocking package)
  ┌─────────────────────────────┐
  │  Curvature mapping          │  [notebook 02]
  │  Thickness mapping          │  [notebook 03]
  │  pHH3 density mapping       │  [notebook 04]
  │  Spatchcocking projection   │  [notebook 05]
  └─────────────────────────────┘
        ↓
  2D heatmaps + statistics       [notebook 06]
```

---

## Repository map

| Folder / File | Contents | Paper figures |
|---|---|---|
| `src/spatchcocking/` | Installable Python package | All |
| `notebooks/00_quickstart.ipynb` | End-to-end demo | — |
| `notebooks/01_mesh_generation.ipynb` | TIFF → surface mesh | Fig. 1c–d |
| `notebooks/02_curvature_mapping.ipynb` | Gaussian & Mean curvature | Figs. 3, 4 |
| `notebooks/03_thickness_mapping.ipynb` | Apico-basal thickness | Fig. 5 |
| `notebooks/04_pHH3_density_mapping.ipynb` | Mitotic density field | Fig. 6 |
| `notebooks/05_spatchcocking.ipynb` | 3D→2D projection | Fig. 2 |
| `notebooks/06_figure_plots.ipynb` | Statistical plots from CSV data | Figs. 1, 3–6 |
| `data/meshes/` | 3 example lumen meshes per stage (6 total) | Figs. 1, 3–6 |
| `data/csv/` | Per-embryo measurements for figure reproduction | Figs. 1, 3–6 |
| `segmentation/` | Segmentation workflow documentation | Fig. 1, Methods |
| `finite_element/` | 2D plane-stress FEA (separate environment) | Fig. S9 |

---

## Installation

```bash
pip install git+https://github.com/nchahare/spatchcocking
```

Requires Python ≥ 3.10. The `vedo` dependency is installed from the GitHub source to track the latest version.

**Note for Jupyter:** vedo requires an explicit backend setting in notebooks:
```python
from vedo import settings
settings.default_backend = "vtk"
```

---

## Quick usage

```python
import spatchcocking as sp
from vedo import settings
settings.default_backend = "vtk"

# Load a mesh and compute Gaussian curvature
mesh = sp.get_mesh("data/meshes/HH17/HH17_embryo1_lumen.ply")
mesh = sp.compute_and_save_curvatures(mesh)

# Extract medial axis and spatchcock to 2D
axis_points = sp.getAxis(mesh)
planes = sp.getPlanes(axis_points)
flat_mesh = sp.getDeformedmesh(mesh, planes, axis_points)
sp.visualize_flatmesh(flat_mesh, property="Gauss_curvature")
```

See [`notebooks/00_quickstart.ipynb`](notebooks/00_quickstart.ipynb) for a full walkthrough.

---

## Data

**Example meshes** (`data/meshes/`): 3 lumen surface meshes per developmental stage (HH17 and HH20). These are a representative subset of the 7 embryos per stage analyzed in the paper. Meshes are PLY files of the inner lumen boundary, generated from 3D confocal image stacks (4.5 µm z-step, ~5.5 µm/pixel xy). See [`data/meshes/README.md`](data/meshes/README.md).

**Measurement CSV files** (`data/csv/`): Per-embryo measurements (lumen geometry, curvature, thickness, pHH3 density) used to generate the paper figures. See [`data/csv/README.md`](data/csv/README.md).

**Full raw imaging data** are available from the corresponding author upon request.

---

## Segmentation workflow

Image segmentation (Fiji preprocessing → 3D Slicer volumetric segmentation → napari visualization) requires software outside this Python package. See [`segmentation/README.md`](segmentation/README.md) for the full protocol.

---

## Finite element analysis

The 2D plane-stress FEA (Fig. S9) requires [SolidsPy](https://github.com/AppliedMechanics-EAFIT/SolidsPy) and runs in a separate environment. See [`finite_element/README.md`](finite_element/README.md).

---

## Citation

If you use this code, please cite:

```
Chahare N., Imamura C., and Nerurkar N.L. (2026).
Morphometric analysis reveals that the chick cranial neural tube expands as an active shell.
[Journal]. DOI: [to be updated]
```

---

## License

MIT — see [LICENSE](LICENSE).

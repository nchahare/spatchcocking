# spatchcocking

**Supplementary data and code for Chahare, Imamura, and Nerurkar (2026)**

This repository contains the Python scripts, example data, and pre-computed meshes cited as supplementary materials in:

> Chahare N., Imamura C., and Nerurkar N.L. (2026). *Morphometric analysis reveals that the chick cranial neural tube expands as an active shell.* [Journal, DOI — to be updated upon publication]

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
        ↓  (Fiji + 3D Slicer)              [Segmentation]
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
| `notebooks/00_mesh_viewer.ipynb` | 3D viewer for meshes | — |
| `notebooks/01_mesh_generation.ipynb` | TIFF mask → surface mesh | Fig. 1c–d |
| `notebooks/02_curvature_mapping.ipynb` | Gaussian & Mean curvature | Figs. 3, 4 |
| `notebooks/03_thickness_mapping.ipynb` | Apico-basal thickness | Fig. 5 |
| `notebooks/04_pHH3_density_mapping.ipynb` | Mitotic density field | Fig. 6 |
| `notebooks/05_spatchcocking.ipynb` | 3D→2D projection | Fig. 2 |
| `notebooks/06_figure_plots.ipynb` | Statistical plots from CSV data | Figs. 1, 3–6 |
| `data/meshes/` | 6 pre-computed lumen meshes (3×HH17, 3×HH20) | Figs. 1, 3–6 |
| `data/csv/` | Per-embryo measurements for figure reproduction | Figs. 1, 3–6 |
| `data/example/` | Single-embryo trial dataset (masks, meshes, npys, CSV) | Methods |
| `docs/segmentation.md` | Segmentation workflow (Fiji → 3D Slicer → napari) | Fig. 1, Methods |
| `finite_element/` | 2D plane-stress FEA + notebook (separate environment) | Fig. S9 |

---

## Installation

```bash
pip install git+https://github.com/nchahare/spatchcocking
```

Requires Python ≥ 3.10.

!!! note "Jupyter backend"
    vedo requires an explicit backend setting at the top of each notebook:
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

See [notebooks/00_mesh_viewer.ipynb](https://github.com/nchahare/spatchcocking/blob/main/notebooks/00_mesh_viewer.ipynb) for an interactive walkthrough of the pre-computed meshes, and notebooks 01–05 for the full processing pipeline.

---

## Generative AI declaration

Portions of the code and documentation in this repository were developed with the assistance of AI tools (Google Gemini and Anthropic Claude). All code was written, reviewed, and executed by N. Chahare, who verified that it produces the correct results. Claude additionally assisted with writing and editing the documentation files.

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

## Citation

If you use this code, please cite:

```
Chahare N., Imamura C., and Nerurkar N.L. (2026).
Morphometric analysis reveals that the chick cranial neural tube expands as an active shell.
[Journal]. DOI: [to be updated]
```

---

## License

MIT — see [LICENSE](https://github.com/nchahare/spatchcocking/blob/main/LICENSE).

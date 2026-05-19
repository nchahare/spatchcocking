# Example meshes

This folder contains triangulated surface meshes of the cranial neural tube lumen for 3 representative embryos at each developmental stage (6 meshes total). These are a subset of the 7 embryos per stage analyzed in the paper.

## What the meshes represent

Each mesh is the **inner lumen surface** of the cranial neural tube — the fluid-filled cavity bounded by the neuroepithelium. This surface is used for:

- Curvature mapping (Gaussian and Mean curvature)
- Medial axis extraction
- Spatchcocking 3D→2D projection

The **basal (outer) surface** of the neuroepithelium is used for thickness measurements; those meshes are not included here but follow the same format.

## File naming

Files are named by acquisition date and developmental stage:

```
{YYYY-MM-DD-HH-MM}-{stage}.vtk
```

| File | Stage |
|---|---|
| `2025-09-18-13-02-HH17.vtk` | HH17 |
| `2025-09-18-14-26-HH17.vtk` | HH17 |
| `2025-09-18-16-46-HH17.vtk` | HH17 |
| `2025-09-23-15-48-HH20.vtk` | HH20 |
| `2025-10-22-12-30-HH20.vtk` | HH20 |
| `2025-10-23-13-06-HH20.vtk` | HH20 |

## Pre-computed vertex properties

Each VTK file contains all scalar fields pre-computed as vertex point arrays:

| Array name | Units | Description |
|---|---|---|
| `Mean_Curvature` | µm⁻¹ | Mean curvature H = (k1 + k2) / 2 |
| `Gauss_Curvature` | µm⁻² | Gaussian curvature K = k1 × k2 |
| `K1` | µm⁻¹ | Maximum principal curvature |
| `K2` | µm⁻¹ | Minimum principal curvature |
| `thickness` | µm | Apico-basal wall thickness (lumen to basal surface) |
| `phh3` | cell count | pHH3+ mitotic cell density within R = 100 µm |

## Imaging and segmentation parameters

| Parameter | Value |
|---|---|
| Microscope | Zeiss LSM880, 10× air objective (NA 0.45) |
| Z-step size | 4.5 µm |
| XY pixel size | ~5.5 µm (after 4× downsampling in Fiji) |
| Typical stack depth | 400–500 slices |
| Stain | DAPI (nuclear counterstain) |
| Clearing | 3DISCO protocol |
| Segmentation | 3D Slicer (local thresholding + manual refinement) |
| Mesh generation | Marching cubes via trimesh; decimated to 200 vertices, subdivided ×3, Windowed Sinc smoothed |

## Format

Files are in [VTK Legacy](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf) format (binary polydata). They can be opened with:

- **Python (vedo)**: `from vedo import Mesh; mesh = Mesh("2025-09-18-13-02-HH17.vtk")`
- **Python (pyvista)**: `import pyvista as pv; mesh = pv.read("2025-09-18-13-02-HH17.vtk")`
- **3D software**: ParaView, MeshLab, Blender (with VTK plugin)

## Units

Vertex coordinates are in **micrometres (µm)**.

## Loading in spatchcocking

```python
from vedo import Mesh

mesh = Mesh("../data/meshes/2025-09-18-13-02-HH17.vtk")
print(mesh.npoints, "vertices")
print("Arrays:", list(mesh.pointdata.keys()))

# Colour by mean curvature
mesh.cmap("PiYG", mesh.pointdata["Mean_Curvature"])
```

See `notebooks/00_quickstart.py` for a 2×3 overview of all 6 meshes.

## Full dataset

The complete dataset (all 7 embryos per stage, both lumen and basal surfaces, with raw confocal images) is available from the corresponding author upon request.

# Example meshes

This folder contains triangulated surface meshes of the cranial neural tube lumen for 3 representative embryos at each developmental stage (6 meshes total). These are a subset of the 7 embryos per stage analyzed in the paper.

## What the meshes represent

Each mesh is the **inner lumen surface** of the cranial neural tube — the fluid-filled cavity bounded by the neuroepithelium. This surface is used for:

- Curvature mapping (Gaussian and Mean curvature)
- Medial axis extraction
- Spatchcocking 3D→2D projection

The **basal (outer) surface** of the neuroepithelium is used for thickness measurements; those meshes are not included here but follow the same format.

## File naming

```
{stage}_embryo{n}_lumen.ply
```

For example: `HH17/HH17_embryo1_lumen.ply`

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

Files are in [PLY](https://en.wikipedia.org/wiki/PLY_(file_format)) format (ASCII or binary). They can be opened with:

- **Python**: `import trimesh; mesh = trimesh.load("HH17_embryo1_lumen.ply")`
- **3D software**: MeshLab, Blender, ParaView, napari

## Units

Vertex coordinates are in **micrometres (µm)**.

## Loading in spatchcocking

```python
import spatchcocking as sp
mesh = sp.get_mesh("data/meshes/HH17/HH17_embryo1_lumen.ply")
```

## Full dataset

The complete dataset (all 7 embryos per stage, both lumen and basal surfaces, with raw confocal images) is available from the corresponding author upon request.

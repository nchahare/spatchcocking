# CLAUDE.md — Project context for AI assistants

This file stores persistent context for Claude Code sessions working on this repository.

---

## Paper

**Title:** "Morphometric analysis reveals that the chick cranial neural tube expands as an active shell"  
**Authors:** Chahare, Imamura, Nerurkar (2026)  
**Repo:** https://github.com/nchahare/spatchcocking  
**Purpose:** Supplementary materials — custom Python scripts and example meshes cited in the paper.

---

## Working directory

```
C:\Users\nimes\WorkWorkWork\Playground\2026-03-05-git-spatchcocking\spatchcocking\
```

---

## Architecture decisions

- **Only `src/spatchcocking/` is pip-installable** (`pip install git+https://github.com/nchahare/spatchcocking`)
- **Segmentation** (napari, 3D Slicer) — documented only in `segmentation/README.md`, not integrated into the package
- **FEA** (SolidsPy) — separate conda environment (`finite_element/`), not part of the main package
- All notebooks use `from vedo import settings; settings.default_backend = "vtk"` at the top

## Workflow division

- **User** creates `.ipynb` Jupyter notebooks
- **Assistant** creates `.py` percent-format scripts (same content, `# %%` code cells, `# %% [markdown]` docs)

---

## Repository map

```
src/spatchcocking/
    spatchcocking_utils.py   — core module, 1663 lines, all pipeline functions
    sectioning_utils.py      — placeholder, refers to segmentation/README.md
    __init__.py

notebooks/
    00_quickstart.py/.ipynb        — end-to-end demo (1 mesh)
    01_mesh_generation.py/.ipynb   — TIFF mask → PLY mesh
    02_curvature_mapping.py/.ipynb — Gaussian & Mean curvature
    03_thickness_mapping.py/.ipynb — apico-basal tissue thickness
    04_pHH3_density_mapping.py/.ipynb — mitotic cell density
    05_spatchcocking.py/.ipynb     — 3D→2D projection
    06_figure_plots.py/.ipynb      — reproduce all paper figures from CSV

data/
    meshes/HH17/   — 3 example lumen PLY meshes (user to add)
    meshes/HH20/   — 3 example lumen PLY meshes (user to add)
    csv/           — 4 pre-computed CSV files (see below)

segmentation/
    README.md            — Fiji → 3D Slicer → napari pipeline docs
    preprocess_tiff.py   — TIFF mask post-processing (tifffile + scipy)

finite_element/
    fem_plane_stress.py   — SolidsPy 2D FEA (Fig S9, Table S1)
    requirements_fem.txt  — solidspy==1.1.0.post1 etc.
    environment_fem.txt   — full conda spec (win-64, Python 3.10)
    README.md             — setup + run instructions + mater.txt note
```

---

## CSV files (`data/csv/`)

| File | Key columns | Paper figures |
|---|---|---|
| `spatchcocked_measurements.csv` | stage, timepoint, norm_height, angle_degrees, Gauss_Curvature, Mean_Curvature, K1, K2, thickness, phh3, radius | Figs 3–6, Supp Fig 1 |
| `cross_section_area.csv` | z_grid, area, stage, type (individual/mean), sample_id | Fig 1j |
| `compartment_lengths.csv` | stage, end, mhb (midbrain-hindbrain boundary µm), fmb (fore-midbrain boundary µm) | Fig 1e, Supp Fig 1 |
| `lumen_geometry.csv` | Sample_ID, Stage, Region (Forebrain/Midbrain/Hindbrain), Area (µm²), Volume (µm³) | Figs 1f–i, Supp Fig 1 |

**Compartment assignment from `norm_height`:**
- HH17: Forebrain > 0.80, Midbrain 0.60–0.80, Hindbrain < 0.60
- HH20: Forebrain > 0.75, Midbrain 0.55–0.75, Hindbrain < 0.55

---

## Key functions in `spatchcocking_utils.py`

| Function | Purpose |
|---|---|
| `load_tiff_stack(path)` | Load 3D TIFF mask |
| `preprocess_mask(mask)` | Threshold, fill holes, smooth |
| `extract_mesh_marching_cubes(mask, spacing)` | Mask → vertices/faces |
| `tiff_stack_to_mesh(tiff_path, ...)` | Combined convenience function |
| `get_mesh(mask_path, ...)` | Load/build mesh, return vedo Mesh |
| `save_mesh(mesh, output_path)` | Write PLY/STL |
| `compute_and_save_curvatures(msh, depth, degree, ...)` | K, H, k1, k2 as vertex arrays |
| `getTightercmap(values, sigma)` | Percentile-clipped colormap limits |
| `getAxis(mesh, ...)` | Medial axis control points |
| `getPlanes(mesh, axispts, ...)` | Cross-sectional planes |
| `find_closest_dorsal_points(axispts, dorsalpts)` | Anatomical orientation |
| `selectPointsonMesh(mesh, ...)` | Interactive point picker |
| `getDeformedmesh(mesh, axis_info, ...)` | Spatchcocking: 3D→2D TPS warp |
| `get_flatdata(deformed_mesh, ...)` | Extract 2D coords + scalar values |
| `normalize_values(height, angle, curvature, ...)` | Normalise to [0,1] RC, degrees DV |
| `visualize_flatmesh(height, angle, curvature, ...)` | 2D heatmap plot |
| `transfer_points_to_mesh(mesh, pts_data, scalar_name, ...)` | Map pHH3 spots → mesh density |
| `getProperCurvature(msh, depth, ...)` | Curvature with smoothing |

---

## FEA notes

- Script: `finite_element/fem_plane_stress.py`
- **Must create `mater.txt` before running:** `echo "870.0  0.45" > mater.txt`
- Run: `cd finite_element && python fem_plane_stress.py`
- Writes `nodes.txt`, `eles.txt`, `loads.txt` for SolidsPy
- Material: µ=300 Pa, ν=0.45 → E≈870 Pa (neo-Hookean linearisation)
- Loading: P=15 Pa internal pressure on half-annulus
- Outputs: hoop stress σ_hoop and membrane stress σ_mem per geometry condition (Fig S9, Table S1)

---

## Mesh files (`data/meshes/`) ✅

Flat directory, VTK format, date-stamped names. All properties pre-computed inside each file.

| File | Stage |
|---|---|
| `2025-09-18-13-02-HH17.vtk` | HH17 |
| `2025-09-18-14-26-HH17.vtk` | HH17 |
| `2025-09-18-16-46-HH17.vtk` | HH17 |
| `2025-09-23-15-48-HH20.vtk` | HH20 |
| `2025-10-22-12-30-HH20.vtk` | HH20 |
| `2025-10-23-13-06-HH20.vtk` | HH20 |

Vertex arrays: `Mean_Curvature`, `Gauss_Curvature`, `K1`, `K2`, `thickness`, `phh3`

Load with: `from vedo import Mesh; mesh = Mesh("../data/meshes/2025-09-18-13-02-HH17.vtk")`

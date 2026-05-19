# API Reference — `spatchcocking`

All functions are importable from the top-level package:

```python
from spatchcocking import *     # imports everything
import spatchcocking as sp      # use sp.function_name()
```

The source is `src/spatchcocking/spatchcocking_utils.py`.

---

## 1. Mesh generation

### `get_mesh(mask_path, px2umz, px2umxy, check=None)`

Convert a binary TIFF lumen mask into a processed surface mesh.

| Parameter | Type | Description |
|---|---|---|
| `mask_path` | str | Path to binary TIFF stack (lumen=1, background=0) |
| `px2umz` | float | Z-step size in µm (e.g. 4.5) |
| `px2umxy` | float | XY pixel size in µm (e.g. 5.5) |
| `check` | bool | If True, opens a before/after comparison window |

**Returns:** `vedo.Mesh` — surface mesh with ~1000 vertices, coordinates in µm.

**Side effects:** Writes `<name>.ply` and `<name>-mesh.stl` to the current directory.

```python
mesh = get_mesh("../data/example/mask.tif", px2umz=4.5, px2umxy=5.5)
```

---

### `tiff_stack_to_mesh(tiff_path, output_path, spacing, method, preprocess)`

Lower-level pipeline: load TIFF → preprocess mask → marching cubes → save.

Returns `(vertices, faces)` as numpy arrays (not a vedo Mesh). Use `get_mesh` for the full pipeline including smoothing.

---

### `load_tiff_stack(file_path)`

Load a 3D TIFF file. Handles 2D, 3D, and 4-channel TIFFs.

**Returns:** `numpy.ndarray` shape `(Z, Y, X)`.

---

### `preprocess_mask(mask, threshold=0.5, remove_small_objects=True, min_size=100, smooth=True, fill_holes=True)`

Binarise, clean, and smooth a 3D mask before mesh extraction.

**Returns:** binary `numpy.ndarray`.

---

### `save_mesh(vertices, faces, output_path, format='ply')`

Save a mesh (vertices + faces arrays) to PLY, OBJ, or STL.

```python
save_mesh(vertices, faces, "../data/example/lumen.ply")
```

---

## 2. Curvature

### `compute_and_save_curvatures(msh, depth=5, degree=2, save_name=None, check=False)`

Fit local quadric patches to each vertex and compute principal curvatures.

| Parameter | Type | Description |
|---|---|---|
| `msh` | vedo.Mesh | Input mesh |
| `depth` | int | Neighbourhood ring depth for patch fitting (default 5) |
| `degree` | int | Polynomial degree for patch (default 2) |
| `save_name` | str | If set, saves `<save_name>-curvatures.npy` |
| `check` | bool | If True, opens a 4-panel interactive visualisation |

**Returns:** `vedo.Mesh` with four new vertex arrays:

| Array name | Units | Formula |
|---|---|---|
| `Gauss_Curvature` | µm⁻² | K = k₁ × k₂ |
| `Mean_Curvature` | µm⁻¹ | H = (k₁ + k₂) / 2 |
| `K1` | µm⁻¹ | max principal curvature |
| `K2` | µm⁻¹ | min principal curvature |

```python
mesh = compute_and_save_curvatures(mesh, check=True)
K = mesh.pointdata["Gauss_Curvature"]
H = mesh.pointdata["Mean_Curvature"]
```

---

### `getTightercmap(values, sigma=3)`

Compute sigma-clipped colour limits for visualisation.

**Returns:** `(vmin, vmax)` — median ± sigma × std.

```python
vmin, vmax = getTightercmap(mesh.pointdata["Mean_Curvature"], sigma=2)
mesh.cmap("PiYG", "Mean_Curvature", vmin=vmin, vmax=vmax)
```

---

### `getProperCurvature(msh, depth, namefile=None, type="Gaussian", check=None)`

Older single-type curvature computation. Prefer `compute_and_save_curvatures` for new code.

---

## 3. Medial axis

### `selectPointsonMesh(mesh, namefile=None)`

Open an interactive VTK window to click points on the dorsal midline.

- Click to place red spheres at selected points
- Press **`c`** to clear all selections
- Close the window when done

**Returns:** `numpy.ndarray` shape `(N, 3)` — selected 3D coordinates.  
**Side effect:** Saves `<namefile>-endpts.npy`.

```python
endpts = selectPointsonMesh(mesh)   # interactive
# or load previously saved points:
endpts = np.load("mask-endpts.npy")
```

---

### `getAxis(mesh, endpts, namefile=None, num_points=12, N=15, check=None)`

Extract the 1D medial axis skeleton by MLS smoothing.

| Parameter | Type | Description |
|---|---|---|
| `mesh` | vedo.Mesh | Surface mesh |
| `endpts` | array (M, 3) | Dorsal endpoint coordinates from `selectPointsonMesh` |
| `num_points` | int | Number of output axis control points (default 12) |
| `N` | int | MLS smoothing iterations (10–30; more = smoother) |
| `check` | bool | Show each smoothing iteration |

**Returns:** `numpy.ndarray` shape `(num_points, 3)`.  
**Side effect:** Saves `<namefile>-axis.npy`.

```python
axis_pts = getAxis(mesh, endpts, num_points=12, N=15, check=True)
```

---

### `getPlanes(mesh, axispts, endpts, skip_index=np.array([]), check=None)`

Compute cross-sectional planes and dorsal normals along the axis.

| Parameter | Type | Description |
|---|---|---|
| `mesh` | vedo.Mesh | Surface mesh |
| `axispts` | array (N, 3) | Axis control points from `getAxis` |
| `endpts` | array (M, 3) | Dorsal points (defines θ=0 reference) |
| `skip_index` | array | Plane indices to skip (use if planes intersect) |
| `check` | bool | Show planes on mesh interactively |

**Returns:** `axis_info` dict:
```python
{
    'axis':           axispts,       # (N, 3)  axis control points
    'axis normals':   normals,       # (N, 3)  tangent unit vectors
    'dorsal normals': dnormals       # (N, 3)  dorsal-direction unit vectors
}
```

```python
axis_info = getPlanes(mesh, axis_pts, endpts, check=True)
```

---

### `find_closest_dorsal_points(axispts, dorsalpts)`

For each axis point, find the nearest point in `dorsalpts`.

**Returns:** `(closest_indices, closest_points)`.

---

## 4. Spatchcocking

### `getDeformedmesh2(mesh, axis_info, namefile=None, skip_index=np.array([]), dists_threshold=300, check=None)`

Straighten and cylindrically unwrap the mesh (core spatchcocking step).

| Parameter | Type | Description |
|---|---|---|
| `mesh` | vedo.Mesh | Mesh (with vertex arrays) |
| `axis_info` | dict | Output of `getPlanes` |
| `skip_index` | array | Cross-sections to skip |
| `dists_threshold` | float | Max distance (µm) from axis to include a slice point (default 300) |
| `check` | bool | Show original vs straightened axis and warped points |

**Returns:** `vedo.Mesh` — the spatchcocked mesh with vertex coordinates encoding (arc length, angle).

```python
flat_mesh = getDeformedmesh2(mesh, axis_info, check=True)
```

---

### `get_flatdata2(deformed_mesh, namefile=None, check=None)`

Convert the spatchcocked mesh to polar coordinates.

**Returns:** `(radius, angle, height, flat_mesh2)`

| Return | Units | Description |
|---|---|---|
| `radius` | µm | Distance from tube axis |
| `angle` | rad | Azimuthal angle (0 = dorsal, ±π = ventral) |
| `height` | µm | Position along the tube axis |
| `flat_mesh2` | vedo.Mesh | Decimated + smoothed mesh used for projection |

```python
radius, angle, height, flat_mesh2 = get_flatdata2(flat_mesh)
```

---

### `normalize_values2(height, angle, shift_deg=0)`

Normalise coordinates for plotting.

| Parameter | Description |
|---|---|
| `height` | Raw arc-length values (µm) |
| `angle` | Raw azimuthal angles (rad) |
| `shift_deg` | Rotate the DV angle by this many degrees (if dorsal ≠ 0°) |

**Returns:** `(norm_height, angle_degrees)`  
- `norm_height`: 0 (rostral) → 1 (caudal)  
- `angle_degrees`: −180° (ventral) → 0° (dorsal) → +180° (ventral)

```python
s, theta = normalize_values2(height, angle)
```

---

### `visualize_flatmesh2(height, angle, data, sigma=3, namefile=None, colorstr='RdBu')`

Interpolate scattered vertex data onto a 500×500 grid and display as a 2D heatmap.

| Parameter | Description |
|---|---|
| `height` | Normalised arc-length (0–1) |
| `angle` | DV angle in degrees |
| `data` | Scalar values to plot (any vertex array) |
| `sigma` | Sigma clipping for colour limits |
| `colorstr` | Matplotlib or vedo colormap name |

```python
visualize_flatmesh2(s, theta, flat_mesh2.pointdata["Gauss_Curvature"],
                    colorstr="PiYG")
```

---

### `transfer_points_to_mesh(mesh, pts_data, scalar_name, show_result=True)`

Interpolate a scalar field from a point cloud onto the mesh surface.

| Parameter | Type | Description |
|---|---|---|
| `mesh` | vedo.Mesh | Target mesh |
| `pts_data` | ndarray (N, 4) | Columns: X, Y, Z, scalar value |
| `scalar_name` | str | Name for the new vertex array |
| `show_result` | bool | Open a visualisation window if True |

**Returns:** `vedo.Mesh` with the new scalar array.

Used for mapping pHH3+ cell density (from Imaris spot detection) onto the lumen surface:

```python
import numpy as np
import pandas as pd

spots_df = pd.read_csv("../data/example/pHH3_spots.csv")
coords = spots_df[["Position X", "Position Y", "Position Z"]].values
density = spots_df["n_spots_per_area"].values   # pre-computed density column
pts_data = np.column_stack([coords, density])

mesh = transfer_points_to_mesh(mesh, pts_data, scalar_name="phh3")
```

---

## 5. Utility functions

### `getDefaultname(namefile)`

If `namefile` is None, infer the base name from any `.tif` file in the current directory.

---

### `nearest_neighbor_order(points, start=0)`

Greedy nearest-neighbour ordering of a point cloud (used internally by `getAxis` to sort the medial axis).

---

### `transform_to_radial(points, center, normal, dnormal)`

Convert 3D Cartesian points to cylindrical (r, θ, s) coordinates in a local frame defined by `normal` (axis) and `dnormal` (dorsal reference).

---

### `transform_to_cartesian(radial_coords, center, normal, dnormal)`

Inverse of `transform_to_radial`.

---

### `fix_angles(angles, shift_deg=0)`

Shift and wrap angles to [−180°, +180°].

---

## Legacy functions

The following functions are kept for backward compatibility. Prefer the `*2` variants for new code.

| Legacy | Preferred replacement |
|---|---|
| `getDeformedmesh()` | `getDeformedmesh2()` |
| `get_flatdata()` | `get_flatdata2()` |
| `normalize_values()` | `normalize_values2()` |
| `visualize_flatmesh()` | `visualize_flatmesh2()` |
| `getProperCurvature()` | `compute_and_save_curvatures()` |

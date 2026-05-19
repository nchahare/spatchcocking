# The Spatchcocking Method

## Motivation

A chick cranial neural tube is a curved, hollow tube — difficult to analyse as a 3D object because dorso-ventral and rostro-caudal patterns are hard to visualise simultaneously. The **spatchcocking** transformation (named after the culinary technique of flattening a bird) unfolds the tube into a flat 2D projection, mapping every surface vertex to a standardised coordinate pair.

This makes it possible to compare spatial patterns across embryos, developmental stages, and scalar quantities (curvature, thickness, mitotic density) on a single plot.

---

## The 2D coordinate system

After spatchcocking, each vertex has two coordinates:

| Symbol | Name | Range | Direction |
|---|---|---|---|
| **s** | Normalised arc length | 0 → 1 | Rostral (0) → Caudal (1) |
| **θ** | Azimuthal (DV) angle | −180° → +180° | Dorsal (0°) ← → Ventral (±180°) |

The resulting heatmap has arc length on the x-axis and DV angle on the y-axis.

---

## Pipeline overview

```
Binary TIFF mask
      │
      ▼  get_mesh()
  Surface mesh (vedo Mesh, ~1000 vertices)
      │
      ▼  compute_and_save_curvatures()
  Mesh + vertex arrays: Gauss_Curvature, Mean_Curvature, K1, K2
      │
      ▼  selectPointsonMesh()  ←  interactive click
  Dorsal endpoint array  (defines θ = 0)
      │
      ▼  getAxis()
  Medial axis  (1D skeleton, ~12 control points)
      │
      ▼  getPlanes()
  axis_info dict  (axis points + tangent normals + dorsal normals)
      │
      ▼  getDeformedmesh2()
  Straightened ("spatchcocked") mesh
      │
      ▼  get_flatdata2()
  (radius, angle, height, flat_mesh)
      │
      ▼  normalize_values2()
  (norm_height s ∈ [0,1],  angle_degrees θ ∈ [−180°,+180°])
      │
      ▼  visualize_flatmesh2()
  2D heatmap
```

See `notebooks/01` through `05` for step-by-step illustrations of each stage.

---

## Step 1 — Mesh generation

The input is a binary 3D TIFF lumen mask (lumen = 1, background = 0) produced by 3D Slicer segmentation (see `segmentation/README.md`).

`get_mesh()` chains three operations:
1. **Load** — `tifffile.imread` reads the TIFF stack
2. **Marching cubes** — `skimage.measure.marching_cubes` extracts an isosurface at level = 0.5
3. **Post-process** — vedo: `decimate(n=1000)`, `subdivide(3)`, `smooth()`, `compute_normals()`

Vertex coordinates are in **micrometres (µm)** because the voxel spacing (`px2umz`, `px2umxy`) is passed as the marching-cubes grid spacing.

---

## Step 2 — Surface curvature

`compute_and_save_curvatures()` fits a degree-2 polynomial surface patch to each vertex's local neighbourhood (default: depth = 5 rings of adjacent vertices). The principal curvatures k₁ (max) and k₂ (min) are extracted from the second fundamental form of this patch:

- **Gaussian curvature** K = k₁ × k₂  (µm⁻²)
- **Mean curvature** H = (k₁ + k₂) / 2  (µm⁻¹)

Results are stored as vertex point arrays directly on the vedo Mesh object: `Gauss_Curvature`, `Mean_Curvature`, `K1`, `K2`.

---

## Step 3 — Medial axis extraction

`selectPointsonMesh()` opens an interactive VTK window. Click 2–3 points along the **dorsal midline** from posterior to anterior end. These define both the endpoints of the tube and the θ = 0° reference direction.

`getAxis()` then:
1. Subsamples the mesh point cloud
2. Runs N iterations of 1D MLS (moving least-squares) smoothing (`smooth_mls_1d`)
3. Orders the smoothed points by nearest-neighbour traversal
4. Fits a spline and samples ~12 evenly-spaced control points

`getPlanes()` computes at each control point:
- **Tangent normal** — local tube direction (gradient of axis positions)
- **Dorsal normal** — direction from axis point toward the dorsal midline (projected onto the cross-sectional plane)

These are returned as the `axis_info` dictionary used by `getDeformedmesh2`.

---

## Step 4 — Spatchcocking (3D → 2D)

`getDeformedmesh2()` performs the core geometric transformation:

For each cross-sectional plane along the axis:
1. **Slice** the mesh with the plane (`mesh.intersect_with_plane`)
2. **Convert** slice points to cylindrical coordinates (r, θ, s) centred on the axis point, using the tangent and dorsal normals as the local frame
3. **Reconstruct** in a straightened frame where the axis runs along Z and dorsal points along +Y

The mesh is then **warped** from the original to the transformed point cloud using a thin-plate spline (`mesh.warp()`). This continuously deforms the entire surface, not just the slice points.

---

## Step 5 — Flat projection and normalisation

`get_flatdata2()` converts the straightened mesh to polar coordinates:
- **radius** r — distance from the tube axis
- **angle** θ — azimuthal angle (radians, relative to +Y = dorsal)
- **height** s — position along the tube axis (µm)

`normalize_values2()` maps s to [0, 1] and converts θ to degrees.

`visualize_flatmesh2()` interpolates the scattered (s, θ) vertex data onto a 500×500 grid using `scipy.interpolate.griddata` (method = 'linear') and displays it as a pseudocolour image.

---

## Notes and limitations

- The MLS smoothing in `getAxis` can produce artefacts if the tube has sharp bends (e.g. the midbrain–hindbrain boundary). The `check=True` flag in `getAxis` and `getPlanes` opens interactive windows to inspect each step.
- Intersecting cross-sectional planes (visible when tubes curve sharply) can be skipped using the `skip_index` argument in `getDeformedmesh2`.
- The TPS warp (`sigma=1`) introduces some smoothing; vertices far from any slice point may be poorly constrained.
- All scalar arrays (curvature, thickness, pHH3) are carried through the warp because vedo preserves `pointdata` when cloning and warping meshes.

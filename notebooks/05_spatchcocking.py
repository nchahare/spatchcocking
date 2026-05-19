# %% [markdown]
# # Notebook 05: Spatchcocking — 3D to 2D projection
#
# **Paper section:** Methods — Spatchcocking transformation
# **Paper figures:** Fig. 2 (coordinate system schematic),
# Figs. 3b/c, 4b/e, 5b/e, 6b/e (heatmaps)
#
# This script performs the full spatchcocking pipeline on a mesh that already
# has scalar properties (curvature, thickness, pHH3 density) stored as
# vertex arrays.
#
# **The pipeline:**
# 1. **Medial axis** — extract the 1D skeleton of the tube (`getAxis`)
# 2. **Cross-sectional planes** — perpendicular planes along the axis (`getPlanes`)
# 3. **Dorsal orientation** — define θ = 0 (dorsal midline) via selected points
# 4. **Straighten and unwrap** — thin-plate spline warp to cylindrical geometry
#    (`getDeformedmesh`)
# 5. **Extract 2D coordinates** — normalised arc length s ∈ [0,1] and
#    azimuthal angle θ ∈ [−180°, 180°] (`get_flatdata`, `normalize_values`)
# 6. **Visualize** — 2D heatmap of any vertex scalar (`visualize_flatmesh`)
#
# **Prerequisites:** Notebooks 02–04 must have been run to generate
# vertex arrays for curvature, thickness, and pHH3 density.

# %%
from vedo import settings
settings.default_backend = "vtk"

import spatchcocking as sp
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load mesh with all properties
#
# Load a mesh that has been processed through notebooks 02–04
# (curvature + thickness + pHH3 density stored as vertex arrays).

# %%
mesh = sp.get_mesh("../data/meshes/HH17/HH17_embryo1_lumen.ply")
mesh = sp.compute_and_save_curvatures(mesh)

print("Available vertex arrays:", list(mesh.pointdata.keys()))

# %% [markdown]
# ## Step 1: Extract medial axis
#
# `getAxis` traces the tube from one end to the other by computing
# the centroid of each cross-sectional slice and fitting a smooth
# 1D curve through these centroids using a moving-least-squares approach.

# %%
# getAxis fits a 1D MLS smooth curve through the tube centroids
axis_points = sp.getAxis(mesh)
print(f"Axis: {len(axis_points)} control points")

# %% [markdown]
# ## Step 2: Generate cross-sectional planes
#
# `getPlanes` returns one perpendicular plane per axis control point.
# Each plane has a centre (the axis point) and a normal (the local
# tangent direction).  These planes are used to slice the mesh and
# define the local cylindrical coordinate system.

# %%
planes = sp.getPlanes(mesh, axis_points, axis_points[[0, -1]])
print(f"{len(planes)} planes generated along the medial axis")

# %% [markdown]
# ## Step 3: Select dorsal points (anatomical orientation)
#
# Dorsal points define θ = 0 (dorsal midline).  In the spatchcocked
# projection, dorsal is placed at the centre of the x-axis
# (angle = 0°), with ventral at the left and right edges (±180°).
#
# Points can be selected interactively in a VTK window, or loaded
# from a previously saved `.npy` file.

# %%
# Interactive selection — run this cell in a VTK window
# dorsal_points = sp.selectPointsonMesh(mesh)  # click dorsal midline points

# Or load pre-saved dorsal points
# dorsal_points = np.load("HH17_embryo1_dorsal_pts.npy")

# For this demo, find dorsal points automatically from the axis + planes
dorsal_points = sp.find_closest_dorsal_points(axis_points, axis_points)

# %% [markdown]
# ## Step 4: Straighten and unwrap (spatchcock)
#
# `getDeformedmesh` performs the core geometric transformation:
# 1. Each mesh vertex is projected onto its nearest cross-sectional plane
# 2. The radial distance from the axis and the azimuthal angle from the
#    dorsal reference define cylindrical coordinates (r, θ, s)
# 3. A thin-plate spline (TPS) warp maps these coordinates to 2D (s, θ)
#
# The output is a new mesh whose x/y positions encode (arc length, angle).

# %%
# Thin-plate spline warping into cylindrical geometry
flat_mesh = sp.getDeformedmesh(mesh, planes, axis_points, dorsal_points)

# %% [markdown]
# ## Step 5: Extract 2D coordinates and normalize
#
# `get_flatdata` reads the 2D positions and the chosen vertex scalar.
# `normalize_values` maps arc length to [0, 1] (0 = rostral, 1 = caudal)
# and converts the azimuthal coordinate to degrees (0° = dorsal, ±180° = ventral).

# %%
properties = ["Gauss_curvature", "Mean_curvature"]

for prop in properties:
    height, angles, values = sp.get_flatdata(flat_mesh, property=prop)
    height_norm, angles_deg, values_norm = sp.normalize_values(height, angles, values)

    fig, ax = plt.subplots(figsize=(4, 7))
    sp.visualize_flatmesh(height_norm, angles_deg, values_norm,
                          title=f"{prop} — HH17 E1", ax=ax)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Averaging across multiple embryos
#
# Because all embryos are projected onto the same (s, θ) grid, data can be
# pooled by binning into equal-width 2D bins and computing the mean value
# per bin.  This is how the *Spatial Average* panels (Figs. 3c, 3f, etc.)
# were generated.

# %%
# Pseudo-code for multi-embryo pooling:

# all_heights, all_angles, all_values = [], [], []
# for mesh_path in mesh_paths:
#     mesh = sp.get_mesh(mesh_path)
#     mesh = sp.compute_and_save_curvatures(mesh)
#     axis_points = sp.getAxis(mesh)
#     planes = sp.getPlanes(mesh, axis_points, axis_points[[0, -1]])
#     dorsal_points = sp.find_closest_dorsal_points(axis_points, axis_points)
#     flat_mesh = sp.getDeformedmesh(mesh, planes, axis_points, dorsal_points)
#     h, a, v = sp.get_flatdata(flat_mesh, property="Gauss_curvature")
#     h, a, v = sp.normalize_values(h, a, v)
#     all_heights.append(h)
#     all_angles.append(a)
#     all_values.append(v)
#
# heights = np.concatenate(all_heights)
# angles  = np.concatenate(all_angles)
# values  = np.concatenate(all_values)
# sp.visualize_flatmesh(heights, angles, values, title="Mean K — HH17 (n=3)")

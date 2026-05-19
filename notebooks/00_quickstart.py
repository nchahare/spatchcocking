# %% [markdown]
# # Quickstart: spatchcocking a neural tube mesh
#
# This script demonstrates the full pipeline end-to-end using one example mesh:
#
# 1. Load a surface mesh
# 2. Compute Gaussian and Mean curvature
# 3. Extract the medial axis
# 4. Run the spatchcocking 3D→2D projection
# 5. Visualize the result as a 2D heatmap
#
# **Paper:** Chahare, Imamura, Nerurkar (2026)
# **Installation:**
# ```bash
# pip install git+https://github.com/nchahare/spatchcocking
# ```
#
# Run this file interactively in VS Code (Python Interactive) or Spyder using
# the `# %%` cell delimiters.

# %%
# vedo requires an explicit backend outside of Jupyter
from vedo import settings
settings.default_backend = "vtk"

# %%
import spatchcocking as sp
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Load mesh
#
# The example mesh is a triangulated surface of the lumen (inner wall) of
# a chick cranial neural tube at stage HH17.  It was generated from a
# binary TIFF segmentation mask using marching cubes (see notebook 01).

# %%
mesh_path = "../data/meshes/HH17/HH17_embryo1_lumen.ply"
mesh = sp.get_mesh(mesh_path)
print(f"Mesh: {mesh.npoints} vertices, {mesh.ncells} faces")

# %% [markdown]
# ## 2. Compute surface curvature
#
# `compute_and_save_curvatures` fits local quadric patches to each
# vertex neighbourhood (depth = 5 rings, degree = 2 polynomial) and
# returns the mesh with four new vertex arrays:
# - `Gauss_curvature` (K, µm⁻²): product of principal curvatures
# - `Mean_curvature` (H, µm⁻¹): average of principal curvatures
# - `k1`, `k2`: maximum and minimum principal curvatures

# %%
mesh = sp.compute_and_save_curvatures(mesh)
# Vertex arrays now available: 'Gauss_curvature', 'Mean_curvature', 'k1', 'k2'

# %% [markdown]
# ## 3. Extract medial axis
#
# `getAxis` moves through the mesh cross-sections and finds the centroid
# of each slice, producing a 1D skeleton of the tube.
# `getPlanes` returns the perpendicular cross-sectional planes at each
# axis control point — used in the spatchcocking step.

# %%
# Select dorsal points interactively (or load pre-saved coordinates)
# dorsal_pts = sp.selectPointsonMesh(mesh)  # interactive

axis_points = sp.getAxis(mesh)
planes = sp.getPlanes(mesh, axis_points, axis_points[[0, -1]])

# %% [markdown]
# ## 4. Spatchcocking: 3D → 2D projection
#
# `getDeformedmesh` unwraps the tube by:
# 1. Aligning each cross-section to the medial axis tangent
# 2. Converting to cylindrical coordinates (arc length s, azimuthal angle θ)
# 3. Applying a thin-plate spline warp to flatten the surface into 2D
#
# The result is a planar mesh where:
# - x-axis = normalised rostral-caudal position (0 = rostral, 1 = caudal)
# - y-axis = dorso-ventral angle in degrees (0° = dorsal midline, ±180° = ventral)

# %%
flat_mesh = sp.getDeformedmesh(mesh, planes, axis_points)
height, angles, values = sp.get_flatdata(flat_mesh, property="Gauss_curvature")
height_norm, angles_deg, values_norm = sp.normalize_values(height, angles, values)

# %% [markdown]
# ## 5. Visualize as 2D heatmap
#
# The spatchcocked heatmap shows the Gaussian curvature K across the
# unfolded tube surface.  Positive K (red) indicates locally saddle-free,
# dome-like regions; negative K (blue) indicates saddle points.

# %%
fig, ax = plt.subplots(figsize=(4, 6))
sp.visualize_flatmesh(height_norm, angles_deg, values_norm,
                      title="Gaussian curvature — HH17 embryo 1",
                      ax=ax)
plt.tight_layout()
plt.show()

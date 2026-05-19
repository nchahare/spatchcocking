# %% [markdown]
# # Notebook 02: Surface curvature mapping
#
# **Paper section:** Results — Surface curvature mapping
# **Paper figures:** Figs. 3 (Gaussian curvature), 4 (Mean curvature)
#
# This script computes the local Gaussian and Mean curvature at every vertex
# of the neural tube lumen mesh.
#
# **Method:** For each vertex, a polynomial quadric surface (degree 2) is
# fitted to the local neighbourhood (depth = 5 rings).  The principal
# curvatures k1 (maximum) and k2 (minimum) are extracted from this
# quadric, giving:
# - Gaussian curvature K = k1 × k2  (µm⁻²)
# - Mean curvature H = (k1 + k2) / 2  (µm⁻¹)
#
# These scalar fields are stored as vertex point arrays on the mesh and
# saved to the PLY file for downstream analysis.

# %%
from vedo import settings
settings.default_backend = "vtk"

import spatchcocking as sp
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load mesh

# %%
mesh = sp.get_mesh("../data/meshes/HH17/HH17_embryo1_lumen.ply")

# %% [markdown]
# ## Compute curvatures
#
# `compute_and_save_curvatures` fits local quadric patches to each vertex
# neighbourhood and adds four vertex arrays to the mesh:
# `Gauss_curvature`, `Mean_curvature`, `k1`, `k2`.

# %%
# Computes k1, k2, Gauss_curvature (K), Mean_curvature (H)
# and stores them as vertex point arrays on the mesh
mesh = sp.compute_and_save_curvatures(mesh)

K = mesh.pointdata["Gauss_curvature"]
H = mesh.pointdata["Mean_curvature"]
print(f"K: min={K.min():.2e}, max={K.max():.2e}, mean={K.mean():.2e}")
print(f"H: min={H.min():.2e}, max={H.max():.2e}, mean={H.mean():.2e}")

# %% [markdown]
# ## Visualize on 3D mesh
#
# `getTightercmap` clips the colour range at ±3 standard deviations
# to prevent outlier vertices from dominating the palette.

# %%
from vedo import Plotter

# Gaussian curvature
mesh.pointdata.select("Gauss_curvature")
vmin, vmax = sp.getTightercmap(K)
mesh.cmap("PiYG", vmin=vmin, vmax=vmax)

p = Plotter(offscreen=True)
p.show(mesh, axes=0)
p.screenshot("gaussian_curvature_3d.png")
from IPython.display import Image
Image("gaussian_curvature_3d.png")

# %%
# Mean curvature
mesh.pointdata.select("Mean_curvature")
vmin, vmax = sp.getTightercmap(H)
mesh.cmap("RdBu_r", vmin=vmin, vmax=vmax)

p = Plotter(offscreen=True)
p.show(mesh, axes=0)
p.screenshot("mean_curvature_3d.png")
Image("mean_curvature_3d.png")

# %% [markdown]
# ## Save mesh with curvature data
#
# The vertex arrays are embedded in the PLY file and will be available
# in all downstream notebooks.

# %%
sp.save_mesh(mesh, "../data/meshes/HH17/HH17_embryo1_lumen_curvature.ply")

# %% [markdown]
# ## Distribution of curvature values
#
# Histograms of K and H highlight the overall shape of the tube:
# - K > 0 vertices are locally dome-shaped (elliptic)
# - K < 0 vertices are locally saddle-shaped (hyperbolic)
# - H > 0 across most of the surface indicates net outward bending

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(K, bins=80, color="teal", edgecolor="none")
axes[0].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[0].set_xlabel("Gaussian curvature K (µm⁻²)")
axes[0].set_ylabel("Vertex count")
axes[0].set_title("Gaussian curvature — HH17 embryo 1")

axes[1].hist(H, bins=80, color="salmon", edgecolor="none")
axes[1].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[1].set_xlabel("Mean curvature H (µm⁻¹)")
axes[1].set_ylabel("Vertex count")
axes[1].set_title("Mean curvature — HH17 embryo 1")

plt.tight_layout()
plt.show()

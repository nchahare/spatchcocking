# %% [markdown]
# # Notebook 02: Surface curvature mapping
#
# **Paper section:** Results — Surface curvature mapping
# **Paper figures:** Figs. 3 (Gaussian curvature), 4 (Mean curvature)
#
# This script computes the local Gaussian and Mean curvature at every vertex
# of the neural tube surface meshes (lumen and basal).
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

from vedo import Mesh, Plotter
import spatchcocking as sp
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Lumen (inner surface) curvature

# %% [markdown]
# ### Load lumen mesh

# %%
lumen = Mesh("../data/example/lumen.ply")

# %% [markdown]
# ### Compute curvatures
#
# `compute_and_save_curvatures` fits local quadric patches to each vertex
# neighbourhood and adds four vertex arrays to the mesh:
# `Gauss_Curvature`, `Mean_Curvature`, `K1`, `K2`.

# %%
lumen = sp.compute_and_save_curvatures(lumen)

K_lumen = lumen.pointdata["Gauss_Curvature"]
H_lumen = lumen.pointdata["Mean_Curvature"]
print(f"Lumen  K: min={K_lumen.min():.2e}, max={K_lumen.max():.2e}, mean={K_lumen.mean():.2e}")
print(f"Lumen  H: min={H_lumen.min():.2e}, max={H_lumen.max():.2e}, mean={H_lumen.mean():.2e}")

# %% [markdown]
# ### Visualize on 3D mesh
#
# `getTightercmap` clips the colour range at ±3 standard deviations
# to prevent outlier vertices from dominating the palette.

# %%
# Gaussian curvature — lumen
lumen.pointdata.select("Gauss_Curvature")
vmin, vmax = sp.getTightercmap(K_lumen)
lumen.cmap("PiYG", vmin=vmin, vmax=vmax)

p = Plotter(offscreen=True)
p.show(lumen, axes=0)
p.screenshot("lumen_gaussian_curvature.png")
from IPython.display import Image
Image("lumen_gaussian_curvature.png")

# %%
# Mean curvature — lumen
lumen.pointdata.select("Mean_Curvature")
vmin, vmax = sp.getTightercmap(H_lumen)
lumen.cmap("RdBu_r", vmin=vmin, vmax=vmax)

p = Plotter(offscreen=True)
p.show(lumen, axes=0)
p.screenshot("lumen_mean_curvature.png")
Image("lumen_mean_curvature.png")

# %% [markdown]
# ### Save lumen mesh with curvature data

# %%
sp.save_mesh(lumen, "../data/example/lumen_curvature.ply")

# %% [markdown]
# ---
# ## Basal (outer surface) curvature

# %% [markdown]
# ### Load basal mesh

# %%
basal = Mesh("../data/example/basal.ply")

# %% [markdown]
# ### Compute curvatures

# %%
basal = sp.compute_and_save_curvatures(basal)

K_basal = basal.pointdata["Gauss_Curvature"]
H_basal = basal.pointdata["Mean_Curvature"]
print(f"Basal  K: min={K_basal.min():.2e}, max={K_basal.max():.2e}, mean={K_basal.mean():.2e}")
print(f"Basal  H: min={H_basal.min():.2e}, max={H_basal.max():.2e}, mean={H_basal.mean():.2e}")

# %% [markdown]
# ### Visualize on 3D mesh

# %%
# Gaussian curvature — basal
basal.pointdata.select("Gauss_Curvature")
vmin, vmax = sp.getTightercmap(K_basal)
basal.cmap("PiYG", vmin=vmin, vmax=vmax)

p = Plotter(offscreen=True)
p.show(basal, axes=0)
p.screenshot("basal_gaussian_curvature.png")
Image("basal_gaussian_curvature.png")

# %%
# Mean curvature — basal
basal.pointdata.select("Mean_Curvature")
vmin, vmax = sp.getTightercmap(H_basal)
basal.cmap("RdBu_r", vmin=vmin, vmax=vmax)

p = Plotter(offscreen=True)
p.show(basal, axes=0)
p.screenshot("basal_mean_curvature.png")
Image("basal_mean_curvature.png")

# %% [markdown]
# ### Save basal mesh with curvature data

# %%
sp.save_mesh(basal, "../data/example/basal_curvature.ply")

# %% [markdown]
# ---
# ## Curvature distributions
#
# Histograms of K and H for both surfaces:
# - K > 0 vertices are locally dome-shaped (elliptic)
# - K < 0 vertices are locally saddle-shaped (hyperbolic)
# - H > 0 across most of the surface indicates net outward bending

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Lumen
axes[0, 0].hist(K_lumen, bins=80, color="teal", edgecolor="none")
axes[0, 0].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[0, 0].set_xlabel("Gaussian curvature K (µm⁻²)")
axes[0, 0].set_ylabel("Vertex count")
axes[0, 0].set_title("Lumen — Gaussian curvature K")

axes[0, 1].hist(H_lumen, bins=80, color="salmon", edgecolor="none")
axes[0, 1].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[0, 1].set_xlabel("Mean curvature H (µm⁻¹)")
axes[0, 1].set_ylabel("Vertex count")
axes[0, 1].set_title("Lumen — Mean curvature H")

# Basal
axes[1, 0].hist(K_basal, bins=80, color="steelblue", edgecolor="none")
axes[1, 0].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[1, 0].set_xlabel("Gaussian curvature K (µm⁻²)")
axes[1, 0].set_ylabel("Vertex count")
axes[1, 0].set_title("Basal — Gaussian curvature K")

axes[1, 1].hist(H_basal, bins=80, color="mediumpurple", edgecolor="none")
axes[1, 1].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[1, 1].set_xlabel("Mean curvature H (µm⁻¹)")
axes[1, 1].set_ylabel("Vertex count")
axes[1, 1].set_title("Basal — Mean curvature H")

plt.tight_layout()
plt.show()

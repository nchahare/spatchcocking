# %% [markdown]
# # Notebook 03: Tissue thickness mapping
#
# **Paper section:** Results — The cranial neural tube maintains a
# dorso-ventral thickness gradient
# **Paper figures:** Fig. 5
#
# This script quantifies local tissue thickness by measuring the Euclidean
# distance from each vertex on the **lumen (inner) surface** mesh to the
# nearest face of the **basal (outer) surface** mesh.
#
# The resulting vertex-wise scalar field (in µm) is stored as the `thickness`
# point array on the lumen mesh and saved to the PLY file.
#
# **Prerequisites:**
# - `HH17_embryo1_lumen.ply` — inner (lumen) surface mesh
# - `HH17_embryo1_basal.ply` — outer (basal lamina) surface mesh
#
# Both meshes must be in the same coordinate system (µm, same origin).
# The basal surface is generated from the neuroepithelium segmentation
# using the same marching-cubes workflow as the lumen surface (notebook 01).

# %%
from vedo import settings
settings.default_backend = "vtk"

import spatchcocking as sp
import numpy as np
import matplotlib.pyplot as plt
from vedo import Mesh

# %% [markdown]
# ## Load lumen and basal meshes

# %%
lumen_mesh = Mesh("../data/example/lumen.ply")

# The basal (outer) surface mesh must be generated separately
# from the neuroepithelium segmentation (same protocol as lumen)
basal_mesh = Mesh("../data/example/basal.ply")

# %% [markdown]
# ## Compute thickness
#
# For each vertex on the lumen mesh, we compute the distance to the nearest
# polygon of the basal mesh using vedo's `distance_to` function.
# `signed=False` returns the absolute (unsigned) distance, which equals the
# wall thickness between the two surfaces.

# %%
# distance_to computes point-to-mesh distance and stores it as a vertex array
lumen_mesh = lumen_mesh.distance_to(basal_mesh, signed=False)

thickness = lumen_mesh.pointdata["Distance"]
print(f"Thickness: min={thickness.min():.1f} µm, max={thickness.max():.1f} µm, mean={thickness.mean():.1f} µm")

# %% [markdown]
# ## Visualize thickness on 3D mesh
#
# The colour map shows the dorso-ventral gradient of wall thickness:
# the ventral wall is consistently thinner than the dorsal wall.

# %%
from vedo import Plotter

lumen_mesh.pointdata.select("Distance")
lumen_mesh.cmap("viridis", vmin=0, vmax=thickness.max())

p = Plotter(offscreen=True)
p.show(lumen_mesh, axes=0)
p.screenshot("thickness_3d.png")
from IPython.display import Image
Image("thickness_3d.png")

# %% [markdown]
# ## Save mesh with thickness data
#
# Rename the array from `Distance` to `thickness` for clarity, then save.

# %%
# Rename the array for downstream clarity
lumen_mesh.pointdata["thickness"] = lumen_mesh.pointdata["Distance"]
sp.save_mesh(lumen_mesh, "../data/example/lumen_thickness.ply")

# %% [markdown]
# ## Thickness distribution
#
# The histogram confirms the bimodal character (thin ventral / thick dorsal
# wall) visible in Fig. 5 of the paper.

# %%
fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(thickness, bins=60, color="steelblue", edgecolor="none")
ax.axvline(thickness.mean(), color="red", linestyle="--",
           label=f"mean = {thickness.mean():.1f} µm")
ax.set_xlabel("Tissue thickness (µm)")
ax.set_ylabel("Vertex count")
ax.set_title("Apico-basal tissue thickness — HH17 embryo 1")
ax.legend()
plt.tight_layout()
plt.show()

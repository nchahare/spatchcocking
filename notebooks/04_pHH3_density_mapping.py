# %% [markdown]
# # Notebook 04: pHH3 mitotic cell density mapping
#
# **Paper section:** Results — Spatial patterns of mitotic activity are
# preserved across global expansion
# **Paper figures:** Fig. 6
#
# This script maps the spatial distribution of mitotic cells (pHH3+) onto
# the neural tube surface.
#
# **Inputs:**
# - Lumen surface mesh (`.ply`)
# - 3D coordinates of pHH3+ cell spots (detected in Imaris, exported as CSV)
#
# **Method:** For each mesh vertex, the local pHH3+ cell density is computed
# as the number of spots within a sphere of radius R = 100 µm.  This radius
# was chosen to span the full tissue thickness at both stages (HH17 and HH20)
# and provides stable, comparable density estimates.
#
# The result is stored as the `pHH3_density` vertex array on the mesh.

# %%
from vedo import settings
settings.default_backend = "vtk"

import spatchcocking as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load mesh and cell coordinates
#
# Imaris exports detected spots as a CSV file with columns
# `Position X`, `Position Y`, `Position Z` (in µm).

# %%
mesh = Mesh("../data/example/lumen.ply")

# Load pHH3+ spot coordinates (exported from Imaris as CSV)
# Expected columns: 'Position X', 'Position Y', 'Position Z' (in µm)
spots_df = pd.read_csv("../data/example/pHH3_spots.csv")
spots = spots_df[["Position X", "Position Y", "Position Z"]].values
print(f"Loaded {len(spots)} pHH3+ cells")

# %% [markdown]
# ## Compute density field and transfer to mesh
#
# `transfer_points_to_mesh` iterates over mesh vertices and, for each vertex,
# counts the number of spot coordinates within radius R.  The result is
# divided by the sphere volume to give a density in cells per µm³, then
# stored as a vertex array.
#
# R = 100 µm was validated against Imaris-derived tissue-section density
# values (see Supplementary Fig. S pHH3 density validation).

# %%
# Transfer pHH3+ density onto the mesh via IDW interpolation
# R = 100 µm: chosen to span the full tissue thickness at both stages
mesh = sp.transfer_points_to_mesh(mesh, spots, scalar_name="pHH3_density")

density = mesh.pointdata["pHH3_density"]
print(f"Density: min={density.min():.1f}, max={density.max():.1f}, mean={density.mean():.1f} cells")

# %% [markdown]
# ## Visualize on 3D mesh
#
# Colour encodes local mitotic density.  High-density regions (yellow/white)
# correspond to active proliferative zones.

# %%
from vedo import Plotter

mesh.pointdata.select("pHH3_density")
mesh.cmap("hot_r", vmin=0, vmax=density.max())

p = Plotter(offscreen=True)
p.show(mesh, axes=0)
p.screenshot("pHH3_density_3d.png")
from IPython.display import Image
Image("pHH3_density_3d.png")

# %% [markdown]
# ## Save mesh with density data

# %%
sp.save_mesh(mesh, "../data/example/lumen_pHH3.ply")

# %% [markdown]
# ## Density distribution

# %%
fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(density, bins=60, color="darkorange", edgecolor="none")
ax.axvline(density.mean(), color="navy", linestyle="--",
           label=f"mean = {density.mean():.2f} cells")
ax.set_xlabel("pHH3+ cell density (cells per sphere, R=100 µm)")
ax.set_ylabel("Vertex count")
ax.set_title("pHH3 mitotic density — HH17 embryo 1")
ax.legend()
plt.tight_layout()
plt.show()

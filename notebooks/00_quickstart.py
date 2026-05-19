# %% [markdown]
# # Quickstart: visualising all example meshes
#
# This script loads all 6 example lumen meshes (3 × HH17, 3 × HH20) from
# `data/meshes/` and demonstrates three levels of visualization:
#
# 1. **Overview grid** — all 6 meshes by stage (flat colour)
# 2. **Stage comparison** — two meshes side-by-side coloured by Gaussian curvature
# 3. **Property explorer** — one mesh showing all 6 scalar fields in a 2×3 grid
#
# The VTK files already contain all pre-computed vertex properties so no
# processing is needed — just load and plot.
#
# **Available vertex properties in each mesh:**
# | Array name | Description | Colormap |
# |---|---|---|
# | `Mean_Curvature` | Mean curvature H (µm⁻¹) | PiYG |
# | `Gauss_Curvature` | Gaussian curvature K (µm⁻²) | PiYG |
# | `thickness` | Apico-basal wall thickness (µm) | GnBu |
# | `K1` | Max principal curvature (µm⁻¹) | Spectral_r |
# | `K2` | Min principal curvature (µm⁻¹) | Spectral_r |
# | `phh3` | pHH3+ mitotic cell density | viridis |
#
# **Paper:** Chahare, Imamura, Nerurkar (2026)
# **Installation:**
# ```bash
# pip install git+https://github.com/nchahare/spatchcocking
# ```

# %%
# use vedo
# turn on this for Jupyter notebook
from vedo import settings
settings.default_backend = "vtk"

from spatchcocking import *

# %% [markdown]
# ## Find all mesh files
#
# Meshes live in `../data/meshes/` as VTK files with date-stamped names:
# `{YYYY-MM-DD-HH-MM}-{stage}.vtk`

# %%
import pathlib
import vedo

# 1. Define the mesh directory and separate by stage
mesh_dir = pathlib.Path("../data/meshes")

hh17_files = sorted(mesh_dir.glob("*HH17.vtk"))
hh20_files = sorted(mesh_dir.glob("*HH20.vtk"))

# Combine: HH17 first, then HH20 — gives natural 2×3 row layout
all_vtk_files = hh17_files + hh20_files

print(f"HH17 ({len(hh17_files)} files):", [f.name for f in hh17_files])
print(f"HH20 ({len(hh20_files)} files):", [f.name for f in hh20_files])

# We need exactly 6 for a 2x3 grid
files_to_show = all_vtk_files[:6]

print("\nFiles to show:")
print(files_to_show)

# %% [markdown]
# ## Overview: all 6 meshes coloured by stage

# %%
plt = Plotter(N=6, axes=4, size=(1800, 1200), sharecam=False)

# Loop through files and assign to subplots
for i, file_path in enumerate(files_to_show):
    # Load and color the mesh
    msh = Mesh(str(file_path))

    # Simple color switch based on stage in filename
    color = "tomato" if "hh17" in str(file_path).lower() else "seagreen"
    msh.color(color)  # .lighting("glossy")

    # Assign mesh to the specific subplot index
    plt.at(i).show(msh, f"File {i}: {file_path.name}")

# Final render and interaction
print(f"Showing {len(files_to_show)} meshes in grid...")
plt.interactive().close()

# %%
print(msh)

# %% [markdown]
# ## Stage comparison: two meshes side-by-side
#
# Pick one representative embryo from each stage and compare their
# Gaussian curvature distributions.
#
# Change `target_indices` to select different embryos:
# - Indices 0–2 → HH17 (rostral to caudal order)
# - Indices 3–5 → HH20

# %%
from vedo import Plotter, Mesh

# Select one HH17 and one HH20 embryo
target_indices = [2, 5]
selected_files = [all_vtk_files[i] for i in target_indices]

# Initialize Plotter for 2 subplots (1×2 grid)
plt = Plotter(N=2, axes=4, size=(1600, 800), sharecam=False)

for i, file_path in enumerate(selected_files):
    msh = Mesh(str(file_path)).add_scalarbar()
    msh.cmap("PiYG", "Gauss_Curvature")
    original_idx = target_indices[i]
    plt.at(i).show(msh)

plt.interactive().close()

# %% [markdown]
# ## Property explorer: all 6 scalar fields on one mesh
#
# Visualize every pre-computed property on a single embryo.
# Change `file_index` (0–5) to switch embryo;
# `cmap_limit` controls the sigma clipping of the colour range.

# %%
file_index = 5
cmap_limit = 2

from vedo import Plotter, Mesh
import numpy as np

# Setup Plotter for 6 fields in a 2×3 grid
plt = Plotter(N=6, axes=4, sharecam=True, size=(1800, 1000))
mesh1 = Mesh(all_vtk_files[file_index])

# Define the fields, titles, and preferred colormaps
# Curvatures benefit from diverging maps (PiYG, RdBu);
# density/thickness use sequential maps (viridis, GnBu)
fields = [
    ("Gauss_Curvature", "Gaussian Curvature",   "PiYG"),
    ("Mean_Curvature",  "Mean Curvature",        "PiYG"),
    ("thickness",       "Thickness",             "GnBu"),
    ("K1",              "Max Principal (K1)",    "Spectral_r"),
    ("K2",              "Min Principal (K2)",    "Spectral_r"),
    ("phh3",            "PHH3 Intensity",        "viridis"),
]

for i, (name, title, colormap) in enumerate(fields):
    if name in mesh1.pointdata.keys():
        msh_alt = mesh1.clone()
        data_values = msh_alt.pointdata[name]

        # Calculate colour limits with sigma clipping
        vmin, vmax = getTightercmap(data_values, cmap_limit)

        # For curvatures: force symmetry around zero
        if any(key in name for key in ["Curvature", "K1", "K2"]):
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax

        # For physical dimensions / density: force 0 as floor
        elif name in ["thickness", "phh3"]:
            vmin = 0

        msh_alt.cmap(colormap, name, vmin=vmin, vmax=vmax)
        msh_alt.add_scalarbar(title=title, label_format=":6.1e")

        plt.at(i).show(msh_alt, f"Field: {name}")
    else:
        plt.at(i).show(f"Data '{name}' not found")

# Render everything
plt.interactive().close()

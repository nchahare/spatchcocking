# %% [markdown]
# # Quickstart: visualising all example meshes
#
# This script loads all 6 example lumen meshes (3 × HH17, 3 × HH20) and
# displays them in a 2×3 grid coloured by Mean Curvature.
#
# The VTK files already contain all pre-computed vertex properties
# (curvature, thickness, pHH3 density) so no processing is needed —
# just load and plot.
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
from vedo import settings
settings.default_backend = "vtk"

# %%
from glob import glob
from pathlib import Path
from vedo import Mesh, Plotter

# %% [markdown]
# ## Load meshes
#
# Meshes are in `data/meshes/` as VTK files named by acquisition date and stage.

# %%
hh17_files = sorted(glob("../data/meshes/*HH17.vtk"))
hh20_files = sorted(glob("../data/meshes/*HH20.vtk"))

print(f"HH17 meshes ({len(hh17_files)}):")
for f in hh17_files:
    print(f"  {Path(f).name}")

print(f"\nHH20 meshes ({len(hh20_files)}):")
for f in hh20_files:
    print(f"  {Path(f).name}")

# %% [markdown]
# ## 2×3 overview — Mean Curvature
#
# Row 1: HH17 (stage ~66 hr incubation)
# Row 2: HH20 (stage ~90 hr incubation)
#
# Colour encodes Mean Curvature H (µm⁻¹).
# - Green (positive H): locally dome-shaped, outward bending
# - Purple (negative H): locally invaginated / inward bending
#
# The three vesicles (forebrain, midbrain, hindbrain) are visible as
# distinct bulges along the rostro-caudal axis.

# %%
PROP   = "Mean_Curvature"
CMAP   = "PiYG"

plt = Plotter(shape=(2, 3), offscreen=True, size=(1800, 1000))

for col, path in enumerate(hh17_files):
    mesh = Mesh(path)
    mesh.cmap(CMAP, mesh.pointdata[PROP])
    label = f"HH17  {Path(path).stem[:10]}"
    plt.at(0, col).show(mesh, label, axes=0, resetcam=True)

for col, path in enumerate(hh20_files):
    mesh = Mesh(path)
    mesh.cmap(CMAP, mesh.pointdata[PROP])
    label = f"HH20  {Path(path).stem[:10]}"
    plt.at(1, col).show(mesh, label, axes=0, resetcam=True)

plt.screenshot("all_meshes_mean_curvature.png")
plt.close()

from IPython.display import Image
Image("all_meshes_mean_curvature.png")

# %% [markdown]
# ## Inspect a single mesh
#
# Load one mesh and list all available vertex arrays.

# %%
mesh = Mesh(hh17_files[0])
print(f"Mesh: {mesh.npoints} vertices, {mesh.ncells} faces")
print("Vertex arrays:", list(mesh.pointdata.keys()))

# %% [markdown]
# ## Visualize a different property
#
# Change `PROP` and `CMAP` to explore other scalar fields.
# Available combinations:
# - `("Gauss_Curvature", "PiYG")`
# - `("thickness", "GnBu")`
# - `("K1", "Spectral_r")`
# - `("K2", "Spectral_r")`
# - `("phh3", "viridis")`

# %%
PROP = "thickness"
CMAP = "GnBu"

plt = Plotter(shape=(2, 3), offscreen=True, size=(1800, 1000))

for col, path in enumerate(hh17_files):
    mesh = Mesh(path)
    mesh.cmap(CMAP, mesh.pointdata[PROP])
    plt.at(0, col).show(mesh, f"HH17  {Path(path).stem[:10]}", axes=0, resetcam=True)

for col, path in enumerate(hh20_files):
    mesh = Mesh(path)
    mesh.cmap(CMAP, mesh.pointdata[PROP])
    plt.at(1, col).show(mesh, f"HH20  {Path(path).stem[:10]}", axes=0, resetcam=True)

plt.screenshot("all_meshes_thickness.png")
plt.close()

Image("all_meshes_thickness.png")

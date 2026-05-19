# %% [markdown]
# # Notebook 01: Mesh generation from TIFF mask
#
# **Paper section:** Methods — Neural tube segmentation
# **Paper figures:** Fig. 1c–d (segmented lumen meshes)
#
# This script converts two binary 3D TIFF segmentation masks (inner lumen and
# outer basal surface) into smoothed, decimated triangulated surface meshes.
#
# **Pipeline for each mask:**
# 1. Load binary TIFF stack (lumen/tissue = 1, background = 0)
# 2. Extract raw mesh via marching cubes (`get_mesh`)
# 3. Smooth (100 iterations of Laplacian smoothing)
# 4. Decimate to 5 % of original face count
# 5. Final smooth pass (50 iterations, boundary-preserving)
# 6. Save as PLY — `lumen.ply` (inner) and `basal.ply` (outer)
#
# **Prerequisites:** TIFF masks must already be segmented.
# See `docs/segmentation.md` for the full segmentation workflow
# (Fiji gamma correction → 3D Slicer local thresholding → manual refinement).
#
# **Input files** (`data/example/`):
# - `2025-10-22-12-30-Inner_Mask.tif`
# - `2025-10-22-12-30-Outer_Mask.tif`
#
# **Output files** (`data/example/`):
# - `lumen.ply` — inner surface, used by notebooks 02–05
# - `basal.ply`  — outer surface, used by notebook 03 (thickness)

# %%
from vedo import settings
settings.default_backend = "vtk"

from spatchcocking import *
import os

# %% [markdown]
# ## Voxel scaling
#
# Set the physical voxel dimensions for your microscope acquisition.
# These values scale the mesh from voxel units into µm.
#
# - `px2umz`  — z-step size (µm per slice)
# - `px2umxy` — in-plane pixel size (µm per pixel)

# %%
scaling = [4.5, 5.535, 5.535]   # [z, xy, xy] in µm

px2umz  = scaling[0]   # z-step size
px2umxy = scaling[1]   # xy pixel size

# %% [markdown]
# ## Set working directory
#
# `get_mesh` saves intermediate files (`.ply`, `-mesh.stl`) to the current
# working directory.  We move into `data/example/` so all outputs land there.

# %%
data_dir = os.path.abspath("../data/example")
os.chdir(data_dir)
print("Working directory:", os.getcwd())

# %% [markdown]
# ## Input file paths

# %%
inner_mask_path = "2025-10-22-12-30-Inner_Mask.tif"
outer_mask_path = "2025-10-22-12-30-Outer_Mask.tif"

# %% [markdown]
# ## Process inner (lumen) mask
#
# `get_mesh` loads the TIFF, runs marching cubes, saves a raw `{namefile}.ply`,
# does a light decimate + subdivide + smooth pass, and returns the mesh.
# We then load the raw `.ply` and apply a heavier smoothing + decimation
# pipeline to produce a clean surface at manageable vertex count.

# %%
namefile = strippathname(inner_mask_path)
print(f"Processing: {namefile}")

mesh_stl = f"{namefile}-mesh.stl"

if os.path.exists(mesh_stl):
    print(f"Loading existing mesh: {mesh_stl}")
    inner_mesh = Mesh(mesh_stl)
else:
    # Generate raw mesh (also saves {namefile}.ply and {namefile}-mesh.stl)
    inner_mesh = get_mesh(inner_mask_path, px2umz, px2umxy)
    print("Raw mesh generated")

# %%
# Quick preview
show(inner_mesh).close()

# %% [markdown]
# ### Smooth and decimate inner mesh
#
# - `smooth(niter=100)` — Laplacian relaxation, 100 passes
# - `decimate(fraction=0.05)` — keep 5 % of original faces
# - `decimate(fraction=0.1).smooth(niter=50, boundary=True)` — final pass

# %%
# Load the raw PLY produced by get_mesh
mesh0 = Mesh(namefile + ".ply")

# Heavy smoothing
mesh1 = mesh0.clone().smooth(niter=100)

# Aggressive decimation
mesh2 = mesh1.clone().decimate(fraction=0.05)

# Final light decimate + boundary-preserving smooth
mesh2.decimate(fraction=0.1).smooth(niter=50, boundary=True)

# %%
# Save final inner mesh — used by all downstream notebooks
mesh2.write("lumen.ply")
print("Saved: lumen.ply")

# %% [markdown]
# ## Process outer (basal) mask
#
# Same pipeline as the inner mask.

# %%
namefile = strippathname(outer_mask_path)
print(f"Processing: {namefile}")

mesh_stl = f"{namefile}-mesh.stl"

if os.path.exists(mesh_stl):
    print(f"Loading existing mesh: {mesh_stl}")
    outer_mesh = Mesh(mesh_stl)
else:
    outer_mesh = get_mesh(outer_mask_path, px2umz, px2umxy)
    print("Raw mesh generated")

# %%
show(outer_mesh).close()

# %%
# Load the raw PLY, smooth and decimate
mesh0 = Mesh(namefile + ".ply")
mesh1 = mesh0.clone().smooth(niter=100)
mesh2 = mesh1.clone().decimate(fraction=0.05)
mesh2.decimate(fraction=0.1).smooth(niter=50, boundary=True)

# %%
# Save final outer mesh — used by notebook 03 (thickness)
mesh2.write("basal.ply")
print("Saved: basal.ply")

# %% [markdown]
# ## Preview both meshes together

# %%
lumen = Mesh("lumen.ply").c("lightblue").alpha(0.7)
basal  = Mesh("basal.ply").c("salmon").alpha(0.4)

show([lumen, basal], N=2, axes=1).close()

# %% [markdown]
# ## Output summary
#
# | File | Content | Used by |
# |---|---|---|
# | `lumen.ply` | Inner (lumen) surface mesh | Notebooks 02, 03, 04, 05 |
# | `basal.ply`  | Outer (basal lamina) mesh  | Notebook 03 (thickness)   |
#
# Pass these paths to the next notebook:
# ```python
# from vedo import Mesh
# lumen_mesh = Mesh("../data/example/lumen.ply")
# basal_mesh  = Mesh("../data/example/basal.ply")
# ```

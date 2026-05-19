# %% [markdown]
# # Notebook 01: Mesh generation from TIFF mask
#
# **Paper section:** Methods — Neural tube segmentation
# **Paper figures:** Fig. 1c–d (segmented lumen meshes)
#
# This script converts a binary 3D TIFF lumen mask into a triangulated
# surface mesh (PLY format).
#
# **Pipeline:**
# 1. Load binary TIFF stack (lumen = 1, background = 0)
# 2. Preprocess mask (fill holes, smooth, threshold)
# 3. Extract mesh via marching cubes
# 4. Smooth and decimate to a manageable vertex count
# 5. Save as PLY
#
# **Prerequisites:** The TIFF mask must already be segmented.
# See `segmentation/README.md` for the full segmentation workflow
# (Fiji gamma correction → 3D Slicer local thresholding → manual refinement).

# %%
from vedo import settings
settings.default_backend = "vtk"

import spatchcocking as sp
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load TIFF mask and extract mesh
#
# `tiff_stack_to_mesh` is the high-level convenience function that chains
# `load_tiff_stack` → `preprocess_mask` → `extract_mesh_marching_cubes`.
#
# Key parameters:
# - `spacing`: voxel size in µm (x, y, z).  For our confocal data: ~5.5 µm/px
#   in-plane, 4.5 µm z-step.
# - `smooth_iterations`: Laplacian smoothing passes on the extracted mesh
# - `decimate`: target fraction of original face count to keep

# %%
# Path to a binary TIFF lumen mask (lumen=1, background=0)
tiff_path = "../data/example/mask.tif"

# Extract mesh with default parameters:
# - decimation to 200 vertices
# - 3 rounds of surface smoothing
mesh = sp.tiff_stack_to_mesh(
    tiff_path=tiff_path,
    spacing=(5.5, 5.5, 4.5),   # µm/pixel
    smooth_iterations=3,
)
print(f"Extracted mesh: {mesh.npoints} vertices, {mesh.ncells} faces")

# %% [markdown]
# ## Visualize
#
# Render the mesh off-screen and display a screenshot in the notebook.

# %%
from vedo import Plotter
plt_3d = Plotter(offscreen=True)
plt_3d.show(mesh, axes=1)
plt_3d.screenshot("mesh_preview.png")
from IPython.display import Image
Image("mesh_preview.png")

# %% [markdown]
# ## Save mesh
#
# Meshes are stored as PLY files — a standard binary format that supports
# arbitrary vertex data arrays (used in later steps for curvature, thickness,
# and pHH3 density).

# %%
output_path = "../data/example/lumen.ply"
sp.save_mesh(mesh, output_path)
print(f"Saved: {output_path}")

# %% [markdown]
# ## Process all embryos (batch)
#
# To process all embryos, iterate over the TIFF files in a directory:

# %%
import os

# tiff_dir = "path/to/HH17_masks/"
# output_dir = "../data/example/"
# os.makedirs(output_dir, exist_ok=True)

# for fname in sorted(os.listdir(tiff_dir)):
#     if fname.endswith(".tiff") or fname.endswith(".tif"):
#         tiff_path = os.path.join(tiff_dir, fname)
#         embryo_id = fname.replace("_lumen_mask.tiff", "")
#         mesh = sp.tiff_stack_to_mesh(tiff_path=tiff_path, spacing=(5.5, 5.5, 4.5))
#         out = os.path.join(output_dir, f"{embryo_id}_lumen.ply")
#         sp.save_mesh(mesh, out)
#         print(f"Saved {out} ({mesh.npoints} pts)")

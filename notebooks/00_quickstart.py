# %%
# use vedo
# turn on this for Jupyter notebook
from vedo import settings
settings.default_backend = "vtk"

from spatchcocking import *

# %%
# find all vtk files in the meshes folder and load them

import pathlib
import vedo

mesh_dir = pathlib.Path("../data/meshes")
all_vtk_files = sorted(mesh_dir.glob("*HH17.vtk")) + sorted(mesh_dir.glob("*HH20.vtk"))

# We need exactly 6 for a 2x3 grid, or we can just cap it at 6
files_to_show = all_vtk_files[:6]

print(files_to_show)

# %%
file_index = 5
cmap_limit = 2


from vedo import Plotter, Mesh
import numpy as np

# 1. Setup Plotter for 6 fields (e.g., a 2x3 or 3x2 grid)
plt = Plotter(N=6, axes=4, sharecam=True, size=(1800, 1000))
mesh1 = Mesh(all_vtk_files[file_index])

# Define the fields, titles, and preferred colormaps
# Curvatures usually benefit from Diverging maps (PiYG, RdBu),
# while density/thickness use Sequential (viridis, plasma)
fields = [
    ("Gauss_Curvature", "Gaussian Curvature", "PiYG"),
    ("Mean_Curvature",  "Mean Curvature",     "PiYG"),
    ("thickness",       "Thickness",   "GnBu"),
    ("K1",              "Max Principal (K1)", "Spectral_r"),
    ("K2",              "Min Principal (K2)", "Spectral_r"),
    ("phh3",            "PHH3 Intensity",     "viridis")
]
for i, (name, title, colormap) in enumerate(fields):
    if name in mesh1.pointdata.keys():
        msh_alt = mesh1.clone()
        data_values = msh_alt.pointdata[name]

        # Calculate initial limits based on your function
        vmin, vmax = getTightercmap(data_values, cmap_limit)

        # --- Applied Logic Correction ---
        # For Curvatures: Force symmetry around zero
        if any(key in name for key in ["Curvature", "K1", "K2"]):
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax

        # For Physical Intensity/Dimensions: Force 0 as floor
        elif name in ["thickness", "phh3"]:
            vmin = 0
        # --------------------------------

        msh_alt.cmap(colormap, name, vmin=vmin, vmax=vmax)
        msh_alt.add_scalarbar(title=title, label_format=':6.1e')

        plt.at(i).show(msh_alt, f"Field: {name}")
    else:
        plt.at(i).show(f"Data '{name}' not found")

# Render everything
plt.interactive().close()

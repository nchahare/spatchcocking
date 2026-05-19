# pHH3+ spot data: Imaris export to NumPy

This document describes how mitotic cell coordinates (pHH3+ immunostaining spots)
are extracted from Imaris, filtered to the neuroepithelium wall, and saved as a
`.npy` file for use in notebook 04.

---

## Overview

1. **Detect spots in Imaris** — automated spot detection on the pHH3 channel
2. **Export statistics** — Imaris writes one CSV per measurement type
3. **Load and filter in Python** — keep only spots inside the tissue wall
   (between the outer basal surface and the inner lumen surface)
4. **Save as `.npy`** — array of shape (N, 3): X, Y, Z in µm

---

## Step 1 — Imaris spot detection

In Imaris, create a **Spots** object on the pHH3 channel:

1. Open the `.ims` file
2. **Add new Spots** → enable *Detect spots close to surface* if needed
3. Set an appropriate estimated diameter (typically 5–10 µm for nuclei)
4. Apply intensity and quality thresholds to filter out background
5. Name the Spots object (e.g. `phh3-filter`)

---

## Step 2 — Export statistics

Export spot positions and intensities from Imaris:

1. Select the Spots object → **Edit → Export Statistics**
2. Choose **Position** and **Intensity Median** as the exported quantities
3. Imaris writes one CSV per quantity into a folder named `{object}_Statistics/`

Expected files after export:

```
2025-10-22-12-30-phh3-filter_Statistics/
    2025-10-22-12-30-phh3-filter_Position.csv
    2025-10-22-12-30-phh3-filter_Intensity_Median.csv   (optional)
    ...
```

The `_Position.csv` file has a 3-row header followed by rows of
`Position Z`, `Position Y`, `Position X` (in µm).

---

## Step 3 — Load spots in Python

```python
import pandas as pd
import numpy as np
import os
from vedo import Points

def getPHH3points(directory, basename, check=None):
    """
    Load pHH3+ spot positions from an Imaris statistics CSV export.

    Parameters
    ----------
    directory : str
        Path to the _Statistics folder produced by Imaris.
    basename : str
        Spots object name used in the export (e.g. '2025-10-22-12-30-phh3-filter').
    check : bool, optional
        If True, opens an interactive vedo window to preview the points.

    Returns
    -------
    points : vedo.Points
        Point cloud with XYZ coordinates and 'Intensity Median' scalar.
    """
    spots_file = os.path.join(directory, basename + "_Position.csv")
    df = pd.read_csv(spots_file, skiprows=3)   # first 3 rows are Imaris header

    positions = np.column_stack([
        df["Position Z"].values,
        df["Position Y"].values,
        df["Position X"].values,
    ])
    intensities = df["Intensity Median"].values if "Intensity Median" in df.columns else None

    points = Points(positions).c("white").ps(5)
    if intensities is not None:
        points.pointdata["Intensity Median"] = intensities
        points.cmap("hot", "Intensity Median",
                    vmin=intensities.min(), vmax=intensities.max())

    if check:
        show(points).close()

    return points
```

Usage:

```python
directory = r"path\to\2025-10-22-12-30-phh3-filter_Statistics"
basename  = "2025-10-22-12-30-phh3-filter"
points = getPHH3points(directory, basename, check=True)
```

---

## Step 4 — Filter spots to the tissue wall

Raw Imaris spots may include cells outside the neural tube.  We keep only
spots that lie **inside the outer (basal) surface** but **outside the inner
(lumen) surface** — i.e. within the neuroepithelium wall.

```python
from vedo import Mesh, Points, show
import numpy as np

# Load surface meshes (decimated, from notebook 01)
s_inner = Mesh("data/example/lumen.ply")   # inner (lumen) surface
s_outer = Mesh("data/example/basal.ply")   # outer (basal lamina) surface

coords = points.vertices   # (N, 3) array from getPHH3points

# Pass 1: keep points inside the outer surface
ids_outer = s_outer.inside_points(coords, return_ids=True)
coords_in_outer = coords[ids_outer]

# Pass 2: from that set, remove points inside the inner surface
ids_inner = s_inner.inside_points(coords_in_outer, return_ids=True)
coords_wall = np.delete(coords_in_outer, ids_inner, axis=0)

# Preview
pts_wall = Points(coords_wall).c("green").ps(5)
show(pts_wall, s_inner.alpha(0.2)).close()

print(f"Total spots: {len(coords)}  →  in wall: {len(coords_wall)}")
```

> **Coordinate alignment note:** If the spot cloud appears offset from the
> mesh, the Imaris and segmentation coordinate systems may differ.  Check
> alignment with `show(points, s_inner)` before filtering.  A common fix is
> subtracting the minimum coordinate along each axis, or rescaling the z axis
> to match the voxel spacing used during meshing.

---

## Step 5 — Save as `.npy`

```python
np.save("data/example/2025-10-22-12-30-spots.npy", coords_wall)
# Shape: (N, 3)  —  columns: Z, Y, X  (µm)
```

This file is the input to notebook 04 (`04_pHH3_density_mapping.py`).

---

## Output file format

| File | Shape | Columns |
|---|---|---|
| `{name}-spots.npy` | (N, 3) | Z, Y, X (µm) |

Note: columns are in **Z, Y, X** order (matching Imaris export convention).
Notebook 04 uses all three coordinates for 3D distance calculations and does
not require a specific axis ordering.

# SPATCHCOCKING

Spatchcocking: Chicken Neural Tube Data Processing Utility Module

Analyzing surface properties of curved 3d tubes

This module provides a geometric pipeline for transforming 3D neural tube lumen 
data into flattened 2D projections. It handles the transition from volumetric 
masks to manifold meshes, extracts topological centerline (medial axes), 
and performs the "spatchcock" unfolding transformation.

NOTE: For jupyter notebook you have to set backend
    ```
    from vedo import settings
    settings.default_backend = "vtk"
    ```

The pipeline generally follows this order:
    1. Mask to Mesh (STL)
    2. Medial Axis/Spline Extraction
    3. Curvature & Signal Mapping (e.g., PHH3)
    4. 3D to 2D Flattening
    
Main Features:
    - Mesh Generation: Converts binary TIFF lumen masks to STL meshes.
    - Differential Geometry: Calculates surface curvature and topological metrics.
    - Signal Extraction: Maps PHH3 (mitotic) signals onto the lumen surface.
    - Flattening: Unrolls the 3D geometry along a neural tube axis for 2D analysis.

Example:
    >>> from spatchcocking import *
    >>> import spatchcocking as sp
    
Dependencies:
    - vedo: For visualization and maniputating mesh data.
    - tifffile: Loading volumetric biological data.
    - scipy: Spline interpolation and medial axis extraction.
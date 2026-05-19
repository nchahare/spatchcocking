"""
sectioning_utils.py

Image segmentation and TIFF preprocessing utilities.

The full segmentation workflow (Fiji preprocessing, 3D Slicer volumetric
segmentation, napari visualization) is documented in segmentation/README.md
and requires software outside this package (napari, 3D Slicer).

The standalone TIFF post-processing script (hole-filling, smoothing,
marching cubes mesh export) that runs without napari is in:
    segmentation/preprocess_tiff.py
"""

# Notes

## File Organization Overview

This project follows the **Python `src` layout**, which is the industry standard for creating maintainable, testable, and distributable Python packages. This structure separates your source code from your configuration and documentation files to ensure clean builds and avoid import conflicts.

Below is the directory structure for `spatchcocking`:

* **`docs/`**: Contains documentation, such as `notes.md`, to help others understand your design decisions.
* **`src/spatchcocking/`**: The core package directory containing your actual source code.
* **`__init__.py`**: Marks the directory as a Python package and exposes your utility functions for easy access.
* **`sectioning_utils.py` & `spatchcocking_utils.py`**: The individual modules containing your specialized utility functions.
* **`.gitignore`**: Specifies files and folders that Git should intentionally ignore, such as `__pycache__` and `*.egg-info`.
* **`README.md`**: The primary entry point for users to understand how to install and use your library.
* **`requirements.txt`**: A clean, portable list of external dependencies required for the project.
* **`setup.py`**: The installation script that handles the configuration and packaging of your project via `pip`.

*Key Principles of this Layout*

* **Isolation**: By placing code in `src/`, you ensure that tests and imports are only possible if the package is correctly installed, which prevents "import-from-source" bugs.
* **Modularity**: Your utility functions are organized into logical modules (`sectioning_utils`, `spatchcocking_utils`), making the codebase easier to navigate and extend.
* **Cleaner Version Control**: With your `.gitignore` configured to exclude build-time files (like `__pycache__` and `*.egg-info`), your repository stays focused strictly on the source code and documentation.


## Installation in a Specific Environment

If you have a dedicated environment (e.g., ni-vedo-env) and want to work on your code locally while having your library accessible, follow these steps:

1. Activate your environment
First, ensure your environment is loaded in your terminal:

```DOS
# If using Anaconda/Conda
conda activate ni-vedo-env

# If using standard venv (Windows)
.\ni-vedo-env\Scripts\activate
```

2. Navigate and Install
Once your terminal prompt shows (ni-vedo-env), navigate to the root directory of your repository where the setup.py file is located:

```DOS
cd path\to\your\spatchcocking_repo
pip install -e .
```

*Why this works:*
The `-e` flag (Editable Mode): This creates a symbolic link between your Python environment and your folder. If you edit image_processing.py, those changes are instantly available in your scripts—no need to reinstall!

Environment Isolation: Because the environment is active, all dependencies (numpy, vedo, etc.) listed in your setup.py are installed into the ni-vedo-env site-packages folder, keeping your global Python installation clean.

*Verifying the Installation*
After installing, you can verify that your specific environment can "see" the package by running this:

```DOS
# Check if Python finds your package
python -c "import spatchcocking; print(spatchcocking.__file__)"
```

If it returns the path to your src/spatchcocking folder, you are successfully set up!

## How to segment manually

**Goal:** Manually segment large 3D volumes by downsampling the Y-axis for efficient annotation, then interpolating and reconstructing to full resolution.

1. Data Preparation & Orientation
* **Load Data:** Import the multi-channel TIFF (DAPI/PHH3).
* **Scaling:** Adjust the Z-axis voxel size (e.g., multiply by `1.5` correction factor).
* **Reslice:** Transpose the volume from `(Z, Y, X)` to `(Y, Z, X)` to make the Y-axis the primary dimension for "slicing" during annotation.

2. Sparse Downsampling (Annotation Prep)
* **Define Density:** Select the number of slices to annotate manually (e.g., `55` slices).
* **Index Mapping:** The script calculates equidistant Y-indices to ensure even coverage across the volume.
* **Initialization:** Generate empty label arrays matching the downsampled shape.

3. Manual Annotation
* **Execution:** Annotate the `Inner` and `Outer` masks using the **Paint** or **Polygon** tools.
* **Best Practices:**
* **Save Frequently:** Export your progress as TIFF files regularly to prevent data loss.
* **Label ID 0:** Remember that Label `0` is your eraser.

4. Upsampling & Reconstruction
* **Map Back:** Use your index mapping to place the annotated slices back into their original "home" positions within the full-size `(1971, 420, 1485)` volume.
* **Fix Gaps:** Manually inspect the start and end of the volume; fill any missing labels and save as `[filename]-fixed.tif`.

5. Morphological Interpolation
* **Automate:** Use the `napari-label-interpolator` plugin to fill the gaps between your manual slices.
* **Memory Management:** * Interpolate one mask (Inner or Outer) at a time.
* Close unnecessary layers to avoid memory overload.
* Save results as `[filename]-fixed-interpolated.tif`.

6. Final Review
* **Validation:** Toggle the visibility of your interpolated masks over the raw data to ensure the alignment and shape remain biologically accurate.

**Pro-Tips for Success:**
* **Keep track of `y_indices`:** Save the index array to a `.npy` file; you need it to reconstruct your volume correctly.
* **Kernel Stability:** If Napari becomes unresponsive or crashes, restart your Jupyter Kernel immediately.
* **Maintain Order:** Ensure the scaling factors remain synced with your axes after every transposition.


## How to convert to mesh


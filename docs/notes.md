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


"""
MkDocs build hook — copies .ipynb files into docs/notebooks/ before each build.
Uses absolute paths derived from mkdocs.yml location so it works in CI too.
docs/notebooks/ is gitignored (build artefact).
"""
import shutil
import os
import glob


def on_pre_build(config):
    # Base directory = folder containing mkdocs.yml (works locally and in CI)
    base_dir = os.path.dirname(os.path.abspath(config["config_file_path"]))

    dst = os.path.join(config["docs_dir"], "notebooks")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    # Main pipeline notebooks
    for nb in glob.glob(os.path.join(base_dir, "notebooks", "*.ipynb")):
        shutil.copy(nb, dst)

    # FEA notebook
    fem_nb = os.path.join(base_dir, "finite_element", "notebook-fem.ipynb")
    if os.path.exists(fem_nb):
        shutil.copy(fem_nb, dst)

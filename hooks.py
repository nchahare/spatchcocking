"""
MkDocs build hook — copies .ipynb files from notebooks/ into docs/notebooks/
before each build. docs/notebooks/ is gitignored (build artefact).
"""
import shutil
import os
import glob


def on_pre_build(config):
    dst = os.path.join(config["docs_dir"], "notebooks")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for nb in glob.glob("notebooks/*.ipynb"):
        shutil.copy(nb, dst)
    # Also copy the FEA notebook from finite_element/
    shutil.copy("finite_element/notebook-fem.ipynb", dst)

from setuptools import setup, find_packages
import os

# Read the contents of your requirements.txt
# This ensures GitHub and Colab install the same libraries you use locally
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="spatchcocking",
    version="0.1.0",
    author="nchahare",
    description="A utility package for analyzing 3D curved tubes",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/nchahare/spatchcocking", # Replace with your actual URL
    
    # Tells setuptools code is in the 'src' directory
    package_dir={"": "src"},
    # Automatically finds the 'spatchcocking' folder inside 'src'
    packages=find_packages(where="src"),
    
    # Dependencies pulled from requirements.txt
    install_requires=required,
    
    python_requires=">=3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
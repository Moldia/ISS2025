from setuptools import setup, find_packages
import os
from pathlib import Path

here = Path(__file__).resolve().parent
readme_path = here / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

VERSION = "0.1.0"
DESCRIPTION = "Preprocess and process ISS data"
LONG_DESCRIPTION = (
    "Process ISS data: align cycles, stitch/retile, export SpaceTx format, and decode."
)

setup(
    name="ISS_preprocessing",
    version=VERSION,
    author="Christoffer Mattsson Langseth",
    author_email="christoffer.langseth@scilifelab.se",
    description=DESCRIPTION,
    long_description=long_description or LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("Notebooks", "tests", "tests.*")),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "tifffile",
        "PyYAML",
        # add other dependencies here
    ],
    python_requires=">=3.9",
    keywords=[
        "python",
        "spatial transcriptomics",
        "spatially resolved transcriptomics",
        "in situ sequencing",
        "ISS",
        "decoding",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

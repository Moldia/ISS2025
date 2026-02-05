from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

# Read long description from README if available
try:
    with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="iss-care",  # PyPI-style name (lowercase, dash-separated)
    version="0.0.23",
    author="Marco Grillo, Saga Helgadottir",
    author_email="marco.grillo@scilifelab.se",
    description="Preprocessing utilities for fast ML-based denoising of ISS images using CARE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],  # intentionally empty; managed via conda env
    python_requires=">=3.9,<3.12",
    keywords=[
        "spatial transcriptomics",
        "spatially resolved transcriptomics",
        "in situ sequencing",
        "ISS",
        "CARE",
        "denoising",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

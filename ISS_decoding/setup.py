from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="ISS_decoding",
    version="0.0.23",
    author="Marco Grillo",
    author_email="marco.grillo@scilifelab.se",
    description="Decode preprocessed ISS images, including SpaceTx formatting and plotting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "docs", "*.ipynb_checkpoints")),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.4",
        "xarray>=0.20",
        "scikit-image>=0.19",
        "slicedimage>=3

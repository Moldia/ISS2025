from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="ISS_decoding",
    version="0.0.24",
    author="Marco Grillo",
    author_email="marco.grillo@scilifelab.se",
    description="Decode preprocessed ISS images, including SpaceTx formatting and plotting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "docs", "*.ipynb_checkpoints")),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22,<2",
        "pandas>=1.4",
        "xarray>=0.20",
        "scikit-image>=0.19",
        "slicedimage>=3",
        "starfish>=0.3",  # add other dependencies you use
    ],
    extras_require={
        "postcode": [
            "pyarrow>=10",
            "torch>=2.1,<3",
            "pyro-ppl>=1.9.1,<2",
            "postcode @ git+https://github.com/mgcizzu/postcode.git@4db68cc5cc398128bcfd97a764bef3c98ee3c583",
        ],
    },
    keywords=[
        "python", "spatial transcriptomics",
        "spatially resolved transcriptomics",
        "in situ sequencing", "ISS", "decoding",
    ],
)

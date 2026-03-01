# ISS_decoding

`ISS_decoding` uses the **starfish** library to extract transcript information from our processed images.  
Starfish is a Python library for image processing in image-based spatial transcriptomics.  
Documentation: https://spacetx-starfish.readthedocs.io/en/latest/

The module includes utilities to:

- **Convert** preprocessed images into a starfish-compatible **SpaceTx** format.
- **Decode** the SpaceTx images to extract spots / barcodes (transcripts).
- **Visualize and filter** decoding results for downstream analysis and QC.

---

# Installation & Updating (Users)

The package is installed in **non-editable mode** (recommended for standard users).

```bash
# Clone the repository (skip if already cloned)
git clone https://github.com/Moldia/ISS2025.git ISS2025
cd ISS2025

# Ensure you are on the main branch and up to date
git fetch origin
git checkout main
git pull --ff-only

# Create environment (first-time setup)
conda env create --name ISS_decoding --file ISS_decoding/ISS_decoding.yml

# If the environment already exists, update it instead:
# conda env update --name ISS_decoding --file ISS_decoding/ISS_decoding.yml --prune

# Activate the environment
conda activate ISS_decoding

# Install / reinstall the package (required after pulling updates)
python -m pip install ./ISS_decoding --upgrade

# (Optional) Register a Jupyter kernel
python -m ipykernel install --user --name ISS_decoding

# (Optional) Verify installation
python -c "import ISS_decoding; print(ISS_decoding.__file__)"

# (Optional) Deactivate when finished
# conda deactivate

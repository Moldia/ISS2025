The `ISS_decoding`  we make use of the starfish library to extract the information contained in our images. starfish is a Python library for processing images of image-based spatial transcriptomics. You can read more about it here: https://spacetx-starfish.readthedocs.io/en/latest/

The module contains function to first format our preprocessed images to a starfish-compatible format (SpaceTx).
In the next steps we then proceed to the actual decoding of the data from the SpaceTx images, and a further set of functions allows the user to plot the decoding results and filter the data appropriately.

## Installation instructions

```bash
# 1) Clone the repo and enter it
git clone https://github.com/Moldia/ISS2025.git ISS2025
cd ISS2025

# 2) Make sure you're on the main branch and up to date
git fetch origin
git checkout main # git checkout saga-updates
git pull --ff-only       

# 3) Create the conda environment from the decoding YAML
# (auto-installs the ISS_decoding package in non-editable mode)
conda env create --name ISS_decoding --file ISS_decoding/ISS_decoding.yml
# If the env already exists, update instead:
# conda env update --name ISS_decoding --file ISS_decoding/ISS_decoding.yml --prune

# 4) Activate the environment
conda activate ISS_decoding

# 5) Register a Jupyter kernel for this environment
python -m ipykernel install --user --name ISS_decoding 

# 6) (Optional) Verify the install
python -c "import ISS_decoding; print('OK:', ISS_decoding.__file__)"

# 7) (Optional) When you're done, go back to base
conda deactivate

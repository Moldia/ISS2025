The `ISS_preprocessing` module aims to transform files extracted from the microscope, into image files that can be used for decoding. Although the images from the microscopes represent a 3D space, our analysis works on 2D images. For this reason, the first step we'll do is a maximum Z-projection of the 3D images obtained from the microscope. The resulting 2D projected images ("tiles") are stitched and the stitched images are then aligned between cycles. Finally, for computational reasons, we slice the aligned big tiffs obtained into smaller (aligned between cycles) tiles, which will be the perfect input to start decoding our samples. 

## Installation instructions

```bash
# 1) Clone the repo and enter it (skip if you have previously cloned it)
git clone https://github.com/Moldia/ISS2025.git ISS2025
cd ISS2025

# 2) Make sure you're on the main branch and up to date
git fetch origin
git checkout main
git pull --ff-only

# 3) Create the conda environment from the YAML
# (auto-installs the ISS_preprocessing package in non-editable mode)
conda env create --name ISS_preprocessing --file ISS_preprocessing/ISS_preprocessing.yml
# If the env already exists, update instead:
# conda env update -n ISS_preprocessing -f ISS_preprocessing/ISS_preprocessing.yml --prune

# 4) Activate the environment
conda activate ISS_preprocessing

# 5) Register a Jupyter kernel for this environment
python -m ipykernel install --user --name ISS_preprocessing

# 6) (Optional) Verify the install
python -c "import ISS_preprocessing; print('OK:', ISS_preprocessing.__file__)"

# 7) (Optional) When you're done, go back to base
conda deactivate

ISS_postprocessing is a Python package used to postprocess In situ sequencinig data. 

## Installation instructions

```bash
# 1) Clone the repo and enter it 
git clone https://github.com/Moldia/ISS2025.git ISS2025
cd ISS2025

# 2) Make sure you're on the main branch and up to date
git fetch origin
git checkout main        
git pull --ff-only       

# 3) Create the conda environment from the postprocessing YML
# (auto-installs the ISS_postprocessing package in non-editable mode)
conda env create -n ISS_postprocessing -f ISS_postprocessing/ISS_postprocessing.yml
# If the env already exists, update instead:
# conda env update -n ISS_postprocessing -f ISS_postprocessing/ISS_postprocessing.yml --prune

# 4) Activate the environment
conda activate ISS_postprocessing

# 5) Register a Jupyter kernel for this environment
python -m ipykernel install --user --name ISS_postprocessing 

# 6) (Optional) Verify the install
python -c "import ISS_postprocessing; print('OK:', ISS_postprocessing.__file__)"

# 7) (Optional) When you're done, go back to base
conda deactivate

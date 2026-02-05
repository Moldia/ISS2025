# 1) Clone the repo and enter it
git clone https://github.com/Moldia/ISS2025.git ISS2025
cd ISS2025

# 2) Make sure you're on the main branch and up to date
git fetch origin
git checkout main 
git pull --ff-only       

# 3) Create the conda environment from the CARE YAML
# (auto-installs the ISS_CARE package in non-editable mode)
conda env create --name ISS_CARE --file ISS_CARE/ISS_CARE.yml
# If the env already exists, update instead:
# conda env update --name ISS_CARE --file ISS_CARE/ISS_CARE.yml --prune

# 4) Activate the environment
conda activate ISS_CARE

# 5) Register a Jupyter kernel for this environment
python -m ipykernel install --user --name ISS_CARE 

# 6) (Optional) Verify the install
python -c "import ISS_CARE; print('OK:', ISS_CARE.__file__)"

# 7) (Optional) When you're done, go back to base
conda deactivate
`ISS_Care` module uses CARE (The Content Aware Image Restoration - a ML-based method for image denoising). When appropriately trained, it can be used as a very fast alternative to image deconvolution. The main advantage is that it can work directly on projected images, significantly reducing the computing requirements. This speeds up the denoising process considerably, although we acknowledge it is likely less accurate than "true" deconvolution. From our benchmarking, CARE seems to be the way to go for most experiments. We advise the user to do a test run on a small sample with both Redlionfish and CARE, compare the results and proceed using CARE only if the results are satisfyingly similar.

You can find the examples on how to use it in the folder "Notebooks".

You can find the pretrained models in the folder "models"

## Installation instructions

```bash
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
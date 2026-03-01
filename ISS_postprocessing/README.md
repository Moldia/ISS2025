# ISS_postprocessing

`ISS_postprocessing` is a Python package for postprocessing In Situ Sequencing (ISS) data.

It provides utilities for:

- Quality control and filtering of decoded transcripts  
- Data cleaning and normalization  
- Aggregation and formatting for downstream analysis  
- Visualization of processed spatial transcriptomics results  

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
conda env create --name ISS_postprocessing --file ISS_postprocessing/ISS_postprocessing.yml

# If the environment already exists, update it instead:
# conda env update --name ISS_postprocessing --file ISS_postprocessing/ISS_postprocessing.yml --prune

# Activate the environment
conda activate ISS_postprocessing

# Install / reinstall the package (required after pulling updates)
python -m pip install ./ISS_postprocessing --upgrade

# (Optional) Register a Jupyter kernel
python -m ipykernel install --user --name ISS_postprocessing

# (Optional) Verify installation
python -c "import ISS_postprocessing; print(ISS_postprocessing.__file__)"

# (Optional) Deactivate when finished
# conda deactivate

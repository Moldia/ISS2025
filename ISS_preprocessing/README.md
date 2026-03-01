# ISS_preprocessing

`ISS_preprocessing` converts raw microscope output into aligned 2D image tiles ready for decoding.

Microscope acquisitions are 3D Z-stacks, while the decoding pipeline operates on 2D images.  
The preprocessing workflow performs:

1. **Maximum Z-projection** – Convert 3D stacks to 2D images.  
2. **Tile stitching** – Reconstruct larger fields of view.  
3. **Cycle alignment** – Align stitched images across sequencing cycles.  
4. **Final tiling** – Subdivide aligned TIFFs into smaller, cycle-aligned tiles suitable for decoding.

The result is a set of spatially aligned, computationally manageable 2D tiles ready for downstream analysis.

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
conda env create --name ISS_preprocessing --file ISS_preprocessing/ISS_preprocessing.yml

# If the environment already exists, update it instead:
# conda env update --name ISS_preprocessing --file ISS_preprocessing/ISS_preprocessing.yml --prune

# Activate the environment
conda activate ISS_preprocessing

# Install / reinstall the package (required after pulling updates)
python -m pip install ./ISS_preprocessing --upgrade

# (Optional) Register Jupyter kernel
python -m ipykernel install --user --name ISS_preprocessing

# (Optional) Verify installation
python -c "import ISS_preprocessing; print(ISS_preprocessing.__file__)"

# (Optional) Deactivate when finished
# conda deactivate

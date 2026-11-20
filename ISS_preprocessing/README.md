# ISS_preprocessing

Pipeline stages can be run together with the standard region layout or
independently with explicit MIP, metadata, OME-TIFF, stitched-image, and retiled
output paths. See [`../PIPELINE_FLEXIBILITY.md`](../PIPELINE_FLEXIBILITY.md).

`ISS_preprocessing` converts raw microscope output into aligned 2D image tiles ready for decoding.

Microscope acquisitions are 3D Z-stacks, while the decoding pipeline operates on 2D images.  
The preprocessing workflow performs:

1. **Maximum Z-projection** – Convert 3D stacks to 2D images.  
2. **Tile stitching** – Reconstruct larger fields of view.  
3. **Cycle alignment** – Align stitched images across sequencing cycles.  
4. **Final tiling** – Subdivide aligned TIFFs into smaller, cycle-aligned tiles suitable for decoding.

The result is a set of spatially aligned, computationally manageable 2D tiles ready for downstream analysis.

## DAPI or RCA-composite registration

ASHLAR uses one reference signal for both first-cycle tile stitching and
cross-cycle registration. The default remains the existing DAPI workflow:

```python
align_and_stitch(
    region_directories,
    cycles=[1, 2, 3, 4],
    n_total_cycles=4,
    align_channel=4,
    registration_reference="dapi",
)
```

RCA mode generates a synthetic channel for every tile, appends it to temporary
OME-TIFF inputs, and runs the same ASHLAR processing path with that appended
channel. ASHLAR therefore calculates stitching and registration from the RCA
signal and applies each resulting cycle geometry to all original channels:

```python
align_and_stitch(
    region_directories,
    cycles=[1, 2, 3, 4],
    n_total_cycles=4,
    align_channel=4,  # excluded from the RCA composite by default
    registration_reference="rca_composite",
    rca_channels=[0, 1, 2, 3],
    rca_composite_method="normalized_max",
    rca_normalization_percentiles=(1, 99.9),
    rca_background_subtraction=True,
    reference_cycle=1,
    save_registration_composites=True,
    registration_qc=True,
)
```

Supported composite methods are `"max"`, `"sum"`, and
`"normalized_max"` (default). Each channel is robustly normalized before the
maximum in `normalized_max`; empty and low-information channels contribute
zeros. If `rca_channels=None`, every original channel except `align_channel` is
used. DAPI is included only when its index is explicitly listed.

Temporary augmented OME-TIFFs retain the original tile positions and pixel
size and are removed after ASHLAR finishes. Original stitched-channel filenames
and downstream inputs are unchanged. `stitched_output_dirs` still controls the
main outputs. `registration_diagnostic_dirs={"R1": path}` can place diagnostic
outputs elsewhere; otherwise they are written below
`R*/preprocessing/registration_diagnostics/`.

QC can save the registered RCA composites, magenta/green overlays, residual
subpixel phase-correlation shifts, normalized cross-correlation values, and a
`registration_metadata.json` summary. These measurements are diagnostic only;
they do not apply local or elastic corrections.

Important limitations:

- Fluorophores changing channels between cycles can expose channel-dependent
  chromatic offsets. This implementation measures but does not correct them.
- RCA registration currently rejects ASHLAR `ffp`/`dfp` profiles because those
  profiles do not contain the appended composite channel. Apply illumination
  correction upstream when using RCA mode.
- The augmented inputs require temporary disk space approximately equal to the
  original OME-TIFF inputs plus one channel.
- ASHLAR's public processing wrapper does not export its internal per-tile
  transforms. The metadata records settings and post-registration residual
  shifts rather than an ASHLAR transform matrix.
- QC is currently global and intensity-based. Local-window displacement maps
  and held-out RCA spot-correspondence metrics are possible future additions.

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

# ISS_postprocessing

`ISS_postprocessing` is a Python package for postprocessing In Situ Sequencing (ISS) data.

It provides utilities for:

- Quality control and filtering of decoded transcripts  
- Data cleaning and normalization  
- Aggregation and formatting for downstream analysis  
- Visualization of processed spatial transcriptomics results  

Cellpose, StarDist, AnnData creation, and PCIseq now accept exact input/output
paths in addition to the conventional region tree. See
[`../PIPELINE_FLEXIBILITY.md`](../PIPELINE_FLEXIBILITY.md) for partial override,
independent-stage, and recomputation examples.

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
```

## GPU-accelerated PCIseq

The GPU-accelerated `mgcizzu/pciSeq` fork is the default PCIseq implementation.
The main `ISS_postprocessing.yml` installs it through the `pciseq-gpu` extra,
pinning the tested revision
[`ccbcf8c`](https://github.com/mgcizzu/pciSeq/commit/ccbcf8c36efb478ea2030ea5820d12360487a9fa).
Because the fork and the upstream distribution both install the same `pciSeq`
Python package, they are alternatives within an environment rather than
packages that can be imported side by side.

To upgrade an existing environment to the default GPU implementation:

```bash
conda activate ISS_postprocessing
python -m pip install --upgrade "./ISS_postprocessing[pciseq-gpu]"
```

To use upstream pciSeq instead, replace the fork with the standard package and
pass `backend="cpu"` to `run_pciseq`:

```bash
python -m pip install --upgrade --force-reinstall "pciSeq>=0.0.61"
```

Verify the GPU installation and CUDA runtime before starting a long fit:

```python
import torch
from ISS_postprocessing.pciseq import pciseq_gpu_backend_installed

assert pciseq_gpu_backend_installed()
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
```

GPU execution is also the runtime default:

```python
cellData, geneData, most_probable, most_probable_filtered = run_pciseq(
    region="R1",
    spots=spots,
    coo_mask=coo,
    sc_expression_matrix=scrnaseq_clean,
    output_dir_prefix=output_dir_prefix,
    gpu_device=0,
    gpu_spots_to_cell=True,
    gpu_block_mb=256,
)
```

`gpu_spots_to_cell=True` enables both accelerated kernels; set it to `False`
to accelerate only cell-type likelihoods. The fork automatically uses bounded
streaming when its full working set would not safely fit in free VRAM. The
wrapper checks both the fork API and CUDA availability before fitting and does
not silently fall back. On a machine without CUDA, explicitly set
`backend="cpu"`; the fork can also execute its optimized CPU path.

## Segmentation

Cellpose and StarDist remain available through
`ISS_postprocessing.segmentation`. Transcript-aware adapters for **Segger**,
**Proseg**, and **BIDCell** now accept the decoded ISS transcript table and
produce the same sparse `.npz` label-mask format used by the existing pipeline.

These methods have different native runtimes and unavoidable method-specific
inputs, so their heavy dependencies are intentionally not installed into the
base ISS environment. Cellpose and StarDist remain in the base environment;
Proseg, Segger, and BIDCell each have a dedicated YAML under `environments/`.
See [SEGMENTATION_WRAPPERS.md](SEGMENTATION_WRAPPERS.md) and the method
notebooks for setup, input requirements, and examples.

## Segmentation-free, spatial-domain, and topology analysis

**SAINSC** support provides kernel-density maps and local-maxima cell proxies,
with optional reference-based cell typing. **SFUMATO** support provides
multi-resolution spatial-domain clustering with posterior uncertainty. These
outputs are not cell-instance masks and are therefore kept separate from the
segmentation dispatcher. **CellGPS/COSTE** support builds directed spatial
topology StructureMaps from either annotated cell centroids in a spatial
AnnData object or decoded transcript points labelled by gene.

See [TRANSCRIPT_ANALYSIS.md](TRANSCRIPT_ANALYSIS.md) and the method-specific
demo notebooks for environment setup, coordinate handling, outputs, and ISS
parameter guidance. SFUMATO's clustering stage requires a CUDA GPU; SAINSC and
both CellGPS workflows run on CPU.

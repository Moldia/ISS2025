# Transcript-aware segmentation wrappers

For SAINSC segmentation-free analysis and SFUMATO spatial-domain clustering,
see [`TRANSCRIPT_ANALYSIS.md`](TRANSCRIPT_ANALYSIS.md). Those methods do not
produce individual-cell label masks and are intentionally not dispatched by
`run_transcript_segmentation()`.

`ISS_postprocessing.segmentation` exposes adapters for **Segger**, **Proseg**,
and **BIDCell**. The external programs keep their native intermediate outputs,
but every adapter finishes with the same pipeline artifact:

```text
<output root>/<region>/postprocessing/segmentation/
    <region>_<method>_stitched_expanded.npz
```

The `.npz` is a SciPy sparse two-dimensional label image. Background is `0`
and cells are positive integers. It can be loaded with the existing
`SEG.load_sparse_mask(...)` and inspected with
`SEG.inspect_and_work_with_segmentation(...)` functions.

## Common input contract

All three wrappers take a transcript table as their first argument and an
optional image as their second argument:

```python
import ISS_postprocessing.segmentation as SEG

labels, labels_coo = SEG.run_transcript_segmentation(
    "proseg",
    transcripts="/data/R1/decoding/2_decoded/R1_decoded.csv",
    image="/data/R1/preprocessing/Cycle1/3_stitched/Cycle1_ch4.tif",
    region="R1",
    input_dir="/data",
    gene_col="target",
    x_col="xc",
    y_col="yc",
    pixel_size_um=0.325,
)
```

`transcripts` may be a pandas `DataFrame` or a CSV, TSV, compressed CSV/TSV,
or Parquet path. The decoded ISS defaults are `target`, `xc`, and `yc`.
Coordinates are assumed to be image pixels. `pixel_size_um` is used when a
method works in physical coordinates.

An image, initial mask, or explicit `mask_shape=(height, width)` is required to
define the output canvas. When Segger or Proseg receives an image but no
`initial_mask`, the wrapper generates a Cellpose prior. Passing an existing
mask is faster and more reproducible.

## Isolated method environments

Cellpose and StarDist run in the normal `ISS_postprocessing` environment. Only
the three transcript-aware segmentation programs use isolated runtimes because
their installation requirements conflict with, or are unrelated to, that
environment. Create them from the repository root:

```bash
conda env create -f ISS_postprocessing/environments/proseg.yml
conda env create -f ISS_postprocessing/environments/segger.yml
conda env create -f ISS_postprocessing/environments/bidcell.yml
```

- `proseg.yml` installs the self-contained `rust-proseg=3.2.0` executable from
  Bioconda. It does not install a Rust toolchain or a second analysis stack.
- `segger.yml` follows Segger v0.3.0's upstream Linux environment: Python 3.11,
  CUDA 12.1, PyTorch 2.5, and RAPIDS 24.10. It requires Linux and an NVIDIA GPU.
- `bidcell.yml` installs BIDCell 1.0.3 with Python 3.10, PyTorch/CUDA 11.8,
  NumPy 1.26, Cellpose below 4, and OpenCV below 4.11. These pins prevent pip
  from selecting the incompatible Cellpose 4 and NumPy 2 stacks.

Keep the notebooks on the `ISS_postprocessing` kernel. The wrappers invoke the
isolated executable or Python environment through `conda run`; those method
dependencies are not imported into the notebook process.

Commands are passed as argument lists and never through a shell. For example,
Segger can be launched from an ISS notebook with:

```python
labels, labels_coo = SEG.segger_segmentation(
    transcripts=transcripts_file,
    image=dapi_file,
    region="R1",
    input_dir=input_dir,
    pixel_size_um=0.325,
    segger_command=["conda", "run", "-n", "segger", "segger"],
    segger_python=["conda", "run", "-n", "segger", "python"],
)
```

The wrapper prepares a minimal standardized input bundle, runs `segger
segment`, exports cell boundaries, rasterizes them at the original image
resolution, and writes the standard sparse mask.

## Proseg

The Rust executable is distributed through Bioconda, so users do not need a
Rust toolchain just to run the notebook:

```bash
conda env create -f ISS_postprocessing/environments/proseg.yml
```

Pass the environment's executable directly or use a command prefix such as
`proseg_command=["conda", "run", "-n", "proseg", "proseg"]`.

Proseg needs a prior estimate of the cells or nuclei. Use one of:

1. `cell_id_col="..."` if the transcript table already has preliminary cell
   assignments;
2. `initial_mask=...` for a NumPy, TIFF, or sparse `.npz` label mask; or
3. `image=...` to generate a Cellpose prior automatically.

```python
labels, labels_coo = SEG.proseg_segmentation(
    transcripts_file,
    dapi_file,
    region="R1",
    input_dir=input_dir,
    pixel_size_um=0.325,
    proseg_command=["conda", "run", "-n", "proseg", "proseg"],
    nthreads=16,
    # Proseg-native options can be appended when needed:
    extra_args=["--voxel-size", "1.0", "--samples", "200"],
)
```

The wrapper requests Proseg's consensus 2D GeoJSON polygons and rasterizes
them. Proseg is stochastic, so repeated runs can differ slightly.

See `Notebooks/Proseg segmentation.ipynb` for the notebook workflow.

## Segger

Segger requires initial boundaries for training. Supply `initial_mask`, or a
DAPI image from which the wrapper can generate a Cellpose seed. Its work
directory contains the standardized input, native segmentation Parquet, and
exported boundary GeoParquet for reproducibility.

`extra_args` is forwarded to `segger segment`, for example:

```python
extra_args=["--n-epochs", "20", "--prediction-mode", "cell"]
```

## BIDCell

BIDCell always requires a DAPI image and a biological expression reference.
The recommended interface accepts an annotated `.h5ad` directly and creates
BIDCell's three internal CSV files in the work directory:

```python
labels, labels_coo = SEG.bidcell_segmentation(
    transcripts_file,
    dapi_file,
    region="R1",
    input_dir=input_dir,
    pixel_size_um=0.325,
    reference_adata="/refs/chicken_reference.h5ad",
    cell_type_col="cell_type",
    # Select one when expression is not stored in adata.X:
    # reference_layer="counts",
    # reference_use_raw=True,
    bidcell_python=["conda", "run", "-n", "bidcell", "python"],
    cpus=8,
    total_steps=4000,
)
```

The selected AnnData expression matrix must be non-negative counts or
normalized expression, not centered/scaled values. By default, the generator
uses the highest and lowest 10% of the spatial panel per cell type as positive
and negative markers and removes positive markers shared by at least one-third
of the cell types. Gene names must match the transcript panel exactly. Missing
reference genes raise an error unless `allow_missing_reference_genes=True` is
explicitly selected; allowed missing genes are included as unmarked zeros.

Reference preparation is also available separately. The result can be passed
directly into the legacy-compatible wrapper interface:

```python
reference_files = SEG.prepare_bidcell_reference(
    "/refs/chicken_reference.h5ad",
    cell_type_col="cell_type",
    spatial_genes=transcript_table["target"].dropna().unique(),
    output_dir="/refs/generated_bidcell",
)

labels, labels_coo = SEG.bidcell_segmentation(
    transcripts_file,
    dapi_file,
    region="R1",
    input_dir=input_dir,
    **reference_files,
)
```

Existing BIDCell reference files remain supported through `reference_path`,
`positive_markers_path`, and `negative_markers_path`. The wrapper validates
their gene coverage, cell-type mapping, binary marker values, and positive /
negative marker exclusivity before starting BIDCell.

The generated configuration uses the source image/transcript pixel size and a
configurable `target_pixel_size_um`. BIDCell's final `*_connected.tif` is
resized to the source DAPI dimensions with nearest-neighbour interpolation and
then written as the common sparse mask.

## Intermediate files and reruns

Each wrapper keeps method-native intermediate files next to the final mask in
`<region>_<method>_work/`. This is intentional: the files are useful for
diagnostics and preserve information that cannot be represented in the raster
mask. Final masks are not replaced unless `overwrite=True` is passed.

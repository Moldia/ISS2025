# Segmentation-free, spatial-domain, and topology analysis

The ISS postprocessing package integrates transcript-first workflows and
spatial-topology analysis that do not create individual-cell segmentation
masks:

- **SAINSC 0.4.0** estimates transcript kernel-density maps, optionally assigns
  reference cell types, and exports local maxima as cell proxies in AnnData.
- **SFUMATO** performs multi-resolution Bayesian clustering of spatial bins and
  reports both hard domains and posterior uncertainty.
- **CellGPS/COSTE 0.0.6** compares labelled spatial point sets and returns
  normalized Searcher and Findee StructureMaps. It accepts either spatial cells
  from AnnData or decoded transcript points labelled by gene.

These methods intentionally live outside `run_transcript_segmentation()`.
SAINSC cell-type maps and SFUMATO semantic domains must not be passed to code
that expects every positive integer to identify one cell. CellGPS matrices are
topology summaries, not label images.

## Coordinate contract

The transcript-table adapters accept the decoded ISS defaults `target`, `xc`,
and `yc`. Coordinates are assumed to be image pixels unless
`coordinate_unit="um"` is selected. SAINSC and SFUMATO additionally require the
physical image resolution as `pixel_size_um`.

- SAINSC retains source coordinates and constructs a grid at `grid_size_um`.
  Its exported AnnData stores image-pixel coordinates in `obsm["spatial"]`,
  micrometres in `obsm["spatial_um"]`, and the native grid in
  `obsm["sainsc_grid"]`.
- SFUMATO input is converted to micrometres before execution. Its standardized
  bin AnnData stores pixels in `obsm["spatial"]` and micrometres in
  `obsm["spatial_um"]`.
- CellGPS uses the coordinate values as supplied. Uniformly converting all
  coordinates from pixels to micrometres does not change its normalized
  cophenetic StructureMaps, but the coordinate unit is recorded in metadata.

## SAINSC

SAINSC 0.4.0 requires Python 3.12+, while the base ISS environment uses an
older Python. Create the supplied environment once:

```bash
conda env create -f ISS_postprocessing/environments/sainsc.yml
```

Run the adapter from the normal ISS notebook kernel and point it at the SAINSC
environment:

```python
import ISS_postprocessing.segmentation_free as SEGFREE

outputs = SEGFREE.run_sainsc_analysis(
    transcripts="/data/R1/decoding/2_decoded/R1_decoded.csv",
    input_dir="/data",
    region="R1",
    pixel_size_um=0.325,
    grid_size_um=0.5,
    kernel_bandwidth_um=4.0,
    min_peak_distance_um=4.0,
    sainsc_python=["conda", "run", "-n", "sainsc", "python"],
)
```

The default output directory is:

```text
<input_dir>/<region>/postprocessing/segmentation_free/sainsc/
```

The primary output is `<region>_sainsc_local_maxima.h5ad`. The prepared input,
complete JSON configuration, metadata, and—by default—the serialized SAINSC
analysis are retained for reproducibility. Set `save_maps=True` to save dense
KDE, background, cell-type, and uncertainty maps; these can be large.

Reference cell typing is enabled by passing `signatures=` as either a pandas
DataFrame or CSV path. Rows must be genes, columns must be cell types, and
values must be finite and non-negative. Set `signatures_log_transformed=True`
when the signatures were derived from log-transformed expression.

For interactive parameter selection without starting SAINSC, use
`prepare_sainsc_analysis(...)`. The demo notebook also shows how to inspect the
prepared transcript density before committing to a whole-slide run.

## SFUMATO

SFUMATO is currently distributed as a source repository rather than an
installable package. The integration was tested against commit:

```text
93806f5c374f88bf4de0bf6e13df4bb7c85df4a4
```

Clone and pin that revision, then create the CUDA environment:

```bash
git clone https://github.com/ale2sia-g/SFUMATO.git
git -C SFUMATO checkout 93806f5c374f88bf4de0bf6e13df4bb7c85df4a4
conda env create -f ISS_postprocessing/environments/sfumato.yml
```

SFUMATO preprocessing runs on CPU. Its Bayesian Gaussian mixture stage exits
unless CUDA is available.

```python
import ISS_postprocessing.spatial_domains as DOMAINS

outputs = DOMAINS.run_sfumato_analysis(
    transcripts="/data/R1/decoding/2_decoded/R1_decoded.csv",
    input_dir="/data",
    region="R1",
    pixel_size_um=0.325,
    sfumato_root="/software/SFUMATO",
    sfumato_python=["conda", "run", "-n", "sfumato", "python"],
    k_list=[8, 12, 16],
    bin_width_um=10,
    min_transcripts_per_bin=4,
)
```

Outputs are placed under:

```text
<input_dir>/<region>/postprocessing/spatial_domains/sfumato/
```

Native caches, posterior tables, dendrograms, configurations, and `.h5ad`
files are preserved. For every requested `K`, the adapter additionally creates
`<region>_sfumato_K<K>_bins.h5ad` with standardized spatial coordinates and
uncertainty columns.

SFUMATO's native `min_genes_per_bin` parameter measures smoothed event density,
so the adapter exposes the less misleading name `min_transcripts_per_bin`.
Its current parser truncates `bin_width` to an integer; the adapter therefore
requires `bin_width_um` to be a whole number.

Use `prepare_sfumato_analysis(...)` to generate the normalized CSV and config
without invoking CUDA. This is useful for reviewing parameter sweeps or
submitting the generated config to a GPU cluster.

## CellGPS/COSTE

CellGPS 0.0.6 and its scientific Python dependencies are available from
conda-forge. Create the isolated CPU environment once:

```bash
conda env create -f ISS_postprocessing/environments/cellgps.yml
```

The adapter uses CellGPS's memory-optimized dataframe workflow and launches it
through `conda run`, so the notebooks remain on the normal
`ISS_postprocessing` kernel. No GPU is required.

### Spatial AnnData input

The AnnData route requires:

- one row per segmented cell;
- a cell-type or cluster label in `adata.obs[cluster_col]`; and
- cell centroids in the first two columns of `adata.obsm[coordinate_key]`,
  normally `adata.obsm["spatial"]`.

A non-spatial scRNA-seq reference object alone is not a valid CellGPS input.
The adapter omits unlabelled cells, filters groups below
`min_cells_per_group`, and leaves the source AnnData object unchanged.

```python
import ISS_postprocessing.spatial_topology as TOPOLOGY

outputs = TOPOLOGY.run_cellgps_from_h5ad(
    "/data/All regions/postprocessing/segmentation/anndata_cellpose.h5ad",
    region="all_regions",
    cluster_col="norm_leiden_2",
    coordinate_key="spatial",
    output_dir="/data/All regions/postprocessing/spatial_topology/cellgps/cells",
    cellgps_python=["conda", "run", "-n", "cellgps", "python"],
)
```

### Decoded transcript-table input

The transcript route accepts the same decoded ISS defaults as the other
transcript-first methods: `target`, `xc`, and `yc`, with optional quality
filtering. Here each gene is a labelled point set, so the resulting matrices
describe **gene-distribution topology**, not cell-type topology and not cell
segmentation. Low-count genes are removed with `min_transcripts_per_gene`.

```python
outputs = TOPOLOGY.run_cellgps_from_transcripts(
    "/data/R1/decoding/2_decoded/R1_decoded.csv",
    input_dir="/data",
    region="R1",
    min_transcripts_per_gene=5,
    cellgps_python=["conda", "run", "-n", "cellgps", "python"],
)
```

Both routes save the normalized point table, complete runner configuration,
entity counts, Searcher and Findee matrices as CSV, both StructureMap plots as
PDF, and a metadata JSON. The default locations are:

```text
<input_dir>/<region>/postprocessing/spatial_topology/cellgps/cells/
<input_dir>/<region>/postprocessing/spatial_topology/cellgps/transcripts/
```

Use `prepare_cellgps_from_h5ad(...)` or
`prepare_cellgps_from_transcripts(...)` to validate and inspect inputs without
running CellGPS. The two CellGPS demo notebooks cover these paths separately.

## ISS parameter guidance

ISS panels are often smaller and sparser than Xenium panels. Start with modest
SVD and cluster counts and inspect multiple physical scales:

- SAINSC: sweep `kernel_bandwidth_um` and `min_peak_distance_um`; avoid treating
  every local maximum as a cell without density and morphology QC.
- SFUMATO cell-scale exploration: try bins near a cell diameter and a low
  `min_transcripts_per_bin`.
- SFUMATO niche-scale exploration: increase `bin_width_um` and smoothing, then
  compare adjacent `K` values and inspect `p1`, `p2`, and `compl_p1`.
- CellGPS: inspect entity counts before interpreting topology. Very small cell
  groups or low-count genes can produce unstable nearest-neighbour summaries;
  raise the relevant minimum-count filter and compare sensitivity.

The supplied notebooks are executable templates rather than universal default
parameter claims. Physical scale and transcript density must be checked for
each experiment.

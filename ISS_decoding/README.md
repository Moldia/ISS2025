# ISS_decoding

`ISS_decoding` uses the **starfish** library to extract transcript information from our processed images.  
Starfish is a Python library for image processing in image-based spatial transcriptomics.  
Documentation: https://spacetx-starfish.readthedocs.io/en/latest/

The module includes utilities to:

- **Convert** preprocessed images into a starfish-compatible **SpaceTx** format.
- **Decode** the SpaceTx images to extract spots / barcodes (transcripts).
- **Visualize and filter** decoding results for downstream analysis and QC.

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
conda env create --name ISS_decoding --file ISS_decoding/ISS_decoding.yml

# If the environment already exists, update it instead:
# conda env update --name ISS_decoding --file ISS_decoding/ISS_decoding.yml --prune

# Activate the environment
conda activate ISS_decoding

# Install / reinstall the package (required after pulling updates)
python -m pip install "./ISS_decoding[postcode,spotiflow,istdeco,bardensr]" --upgrade

# (Optional) Register a Jupyter kernel
python -m ipykernel install --user --name ISS_decoding

# (Optional) Verify installation
python -c "import ISS_decoding; print(ISS_decoding.__file__)"

# (Optional) Deactivate when finished
# conda deactivate
```

## Decoding outputs

Both the standard Starfish workflow and the PoSTcode workflow use Parquet as
their canonical tabular output. Per-tile tables are kept in `tiles/` to support
restarting interrupted runs, and a region-level CSV is also written for
compatibility with existing analysis tools. Tiles with no decoded spots are
stored as empty Parquet checkpoints so they are not needlessly processed again.

```text
2_decoded/
├── R1_decoded.parquet
├── R1_decoded.csv
├── tiles/fov_*.parquet
└── decoding_run_<timestamp>.xml
```

Dense Starfish decoding uses the same layout under `2_decoded_dense/`. Existing
completed CSV-only Starfish runs are still recognized and skipped. Parquet is
recommended for analysis because it preserves column types and per-round QC
arrays; the CSV is intended as an interchange copy.

## Spotiflow detection

Spotiflow can replace Starfish's blob detector while keeping registration,
filtering, intensity measurement, decoding, QC, and output formatting unchanged.
The ISS-specific `hybiss` model is the default; `general` or a local trained model
path can be selected explicitly. The integration pins Spotiflow 0.6.5, which uses
the maintained GitHub release registry for pretrained models.

A complete one-region example, including optional SpaceTx creation, detector QC,
and a side-by-side BlobDetector comparison, is available in
[`Notebooks/ISS_Spotiflow_decoding.ipynb`](Notebooks/ISS_Spotiflow_decoding.ipynb).

```python
from ISS_decoding.decoding import process_experiment

process_experiment(
    input_dir="/path/to/experiment",
    spot_detection_mode="spotiflow",
    spotiflow_kwargs={
        "model": "hybiss",
        "probability_threshold": None,  # model-optimized threshold
        "min_distance": 2,
        "n_tiles": None,               # e.g. (2, 2) to reduce GPU memory use
        "measurement_type": "mean",
    },
)
```

The Spotiflow model is loaded lazily and reused across every selected tile and
region. Detection uses the same 2D reference projections as the Starfish path.
The `spotiflow_probability` output column preserves the model confidence before
Starfish replaces the reference intensity with measured barcode intensities.

Detector alternatives are stored separately so they can be compared safely:

```text
2_decoded/                         # Starfish detector + Starfish decoder
2_decoded_spotiflow/              # Spotiflow detector + Starfish decoder
2_decoded_postcode/               # Starfish detector + PoSTcode decoder
2_decoded_postcode_spotiflow/     # Spotiflow detector + PoSTcode decoder
2_decoded_dense/                   # dense Starfish detection
2_decoded_dense_spotiflow/        # dense Spotiflow detection
2_decoded_istdeco/                # joint ISTDECO detection + decoding
2_decoded_bardensr/               # joint Bardensr detection + decoding
```

`prob_threshold` remains the PoSTcode assignment threshold. Spotiflow's
detection threshold is independently configured as
`spotiflow_kwargs["probability_threshold"]`. Spotiflow model/version/settings
are recorded in the decoding metadata.

## PoSTcode decoding

PoSTcode is available as an alternative decoder while registration, filtering,
and the selected Starfish or Spotiflow detector remain unchanged. The dependency
is pinned to the tested compatibility fork commit
[`4db68cc`](https://github.com/mgcizzu/postcode/commit/4db68cc5cc398128bcfd97a764bef3c98ee3c583).

A complete one-region example, including both creating SpaceTx from retiled
TIFFs and starting from an existing SpaceTx experiment, is available in
[`Notebooks/ISS_PoSTcode_decoding.ipynb`](Notebooks/ISS_PoSTcode_decoding.ipynb).

The adapter converts Starfish spot traces from `spots x rounds x channels` to
PoSTcode's `spots x channels x rounds` layout. It applies the same conversion to
the SpaceTx codebook and validates that the codebook is one-hot.

```python
from ISS_decoding.decoding import process_experiment

process_experiment(
    input_dir="/path/to/experiment",
    decode_mode="POSTCODE",
    prob_threshold=0.7,
    postcode_kwargs={
        "num_iter": 60,
        "batch_size": 15000,
        "device": "auto",  # CUDA on Ubuntu when available; otherwise CPU
        "set_seed": 1,
    },
    # Optional: save the full posterior matrix and fitted model for every tile.
    save_postcode_artifacts=False,
)
```

PoSTcode results are written under `decoding/2_decoded_postcode`, or
`decoding/2_decoded_postcode_spotiflow` when Spotiflow performs detection, so
alternatives can be compared. Parquet is the canonical format and preserves the
per-round QC arrays; a region-level CSV is also written for compatibility:

```text
2_decoded_postcode/
├── R1_decoded_postcode.parquet
├── R1_decoded_postcode.csv
├── tiles/fov_*.parquet
├── posteriors/fov_*.npz       # when save_postcode_artifacts=True
├── models/fov_*.npz           # when save_postcode_artifacts=True
├── decoding_run_<timestamp>.json
└── decoding_run_<timestamp>.xml
```

Every detected spot is retained and has a stable `spot_uid` built from its
region, tile, and detector spot ID. Assignment columns have deliberately
different meanings:

- `target`: accepted gene only; missing for rejected, background, infeasible,
  or NaN assignments.
- `candidate_target`: highest-probability gene regardless of the winning class.
- `assignment_class`: the raw winning class (`gene`, `background`,
  `infeasible`, or `nan`).
- `passes_thresholds`: true only when a gene wins and meets `prob_threshold`.
- `assignment_probability`: posterior probability of the raw winning class.
- `best_gene_probability`, `second_gene_probability`, and
  `gene_probability_margin`: gene-specific confidence and ambiguity.
- `background_probability` and `infeasible_probability`: posterior mass for
  those special classes. Infeasible is an aggregate over excluded barcodes, not
  a probability for one gene.

The complete posterior matrix, fitted weights/covariances, normalization
constants, training loss history, codebook snapshot, seed/device settings, and
pinned PoSTcode commit can be saved per tile with
`save_postcode_artifacts=True`. This is off by default because posterior files
can be large. The legacy `postcode_probability` and `postcode_class` columns
remain as aliases for compatibility.

For an Ubuntu CUDA machine, install the PyTorch build appropriate for the
machine's NVIDIA driver before installing
`ISS_decoding[postcode,spotiflow,istdeco]` if
the pip default is not suitable.

## ISTDECO joint detection and decoding

ISTDECO is an image-level alternative to the separate detector/decoder paths
above. It jointly deconvolves and assigns barcodes, so it does not use the
Starfish blob detector or Spotiflow. SpaceTx remains the common input: the
adapter projects the preprocessed image stack to `(rounds, channels, y, x)`
and passes the one-hot SpaceTx codebook directly as
`(targets, rounds, channels)`.

The optional dependency is pinned to the tested modernized fork commit
[`2200b4e`](https://github.com/mgcizzu/istdeco/commit/2200b4e969528e46588fbe75b6b039f72cd962eb).
The fork adds standard packaging, Python 3.10+ support, current PyTorch support,
input validation, device-safe outputs, and numerical safeguards while retaining
the original ISTDECO update equations.

```python
from ISS_decoding.decoding import process_experiment

process_experiment(
    input_dir="/path/to/experiment",
    regions_to_process=[1],
    decode_mode="ISTDECO",
    istdeco_kwargs={
        "sigma": 1.2,
        "background": 1e-8,
        "niter": 75,
        "acceleration": 1.0,
        "suppress_radius": 1,
        "tile_size": (512, 512),
        "overlap": None,             # automatic PSF-safe halo
        "intensity_percentile": 99.0,
        "intensity_threshold": None, # overrides percentile when set
        "quality_threshold": 0.5,
        "device": "auto",           # CUDA when available, otherwise CPU
        "z_projection": "max",
    },
)
```

The full SpaceTx FOV is decoded in overlapping reads. Only the non-overlapping
core of each read is retained, which avoids duplicate detections at internal
seams while bounding GPU memory. Dense barcode intensity and quality images are
not saved by default because they can be extremely large. The spot table stores
`istdeco_intensity`, `istdeco_quality`, the actual intensity threshold, and the
internal `istdeco_tile`; quality is the algorithm's filtering score, not a
calibrated probability.

Outputs use restartable per-FOV Parquet checkpoints plus region-level Parquet
and CSV copies under `decoding/2_decoded_istdeco/`. Exact settings, package
version, and the pinned fork commit are stored in XML and JSON run manifests.
See [`Notebooks/ISS_ISTDECO_decoding.ipynb`](Notebooks/ISS_ISTDECO_decoding.ipynb)
for a complete example, including the optional SpaceTx formatting step.

## Bardensr joint detection and decoding

Bardensr is a second image-level alternative. Like ISTDECO, it jointly estimates
barcode density and calls peaks, so it bypasses both Starfish/Spotiflow detection
and the downstream Starfish/PoSTcode decoders. SpaceTx remains the input format:
the adapter flattens `(rounds, channels, z, y, x)` images to Bardensr's
`(frames, z, y, x)` layout and transposes the codebook to `(frames, targets)`.

The optional dependency is pinned to the tested compatibility fork commit
[`79cf8f9`](https://github.com/mgcizzu/bardensr/commit/79cf8f9f1f28c8dbd00ab2dd948a214574948307).
The fork retains Bardensr's algorithms while adding Python 3.10+ packaging,
current TensorFlow support for float32 inputs, and repaired regression tests.

```python
from ISS_decoding.decoding import process_experiment

process_experiment(
    input_dir="/path/to/experiment",
    regions_to_process=[1],
    decode_mode="BARDENSR",
    bardensr_kwargs={
        "method": "singleshot",       # fast correlation-based decoder
        "noisefloor": 0.05,
        "peak_threshold": 0.72,
        "peak_threshold_fraction": None,
        "poolsize": (1, 1, 1),
        "tile_size": (512, 512),
        "overlap": None,              # automatic peak/PSF-safe halo
        "normalize_frames": True,
        "device": "auto",            # TensorFlow GPU when available
        "z_projection": "max",
    },
)
```

For the slower optimization-based method, set `method="iterative"`. Its default
threshold is `0.1` times each internal tile's maximum evidence, matching the
upstream example; set an absolute `peak_threshold` after inspecting a pilot
region if cross-tile threshold consistency is important. Iterative-only controls
include `l1_penalty`, `psf_radius`, `iterations`, `estimate_codebook_gain`, and
`estimate_colormixing`. Upstream phasing estimation is unfinished and is rejected
rather than silently doing nothing.

The full FOV is processed in overlapping reads and only each read's owned core is
retained, preventing duplicate calls at seams. Results are written under
`decoding/2_decoded_bardensr/` as restartable per-FOV Parquet files plus
region-level Parquet and CSV copies. Each row includes `bardensr_evidence`, the
actual `bardensr_peak_threshold`, `bardensr_tile_max`, `bardensr_method`, and the
internal `bardensr_tile`. Evidence is an algorithm score, not a calibrated
probability. Dense target-density maps are not retained because they can be very
large.

On Linux, the `bardensr` extra installs TensorFlow's CUDA runtime dependencies via
`tensorflow[and-cuda]`; these are independent of the system CUDA toolkit used by
the PyTorch decoders. Verify both frameworks after installation, and avoid running
the PyTorch and TensorFlow decoders simultaneously when GPU memory is limited.
See [`Notebooks/ISS_Bardensr_decoding.ipynb`](Notebooks/ISS_Bardensr_decoding.ipynb)
for a complete pilot workflow and the optional SpaceTx formatting step.

# Flexible pipeline paths and independent stages

The standard `R*/preprocessing`, `decoding`, and `postprocessing` tree remains
the default. New arguments are optional: an explicit path wins over an inferred
path, and paths that are not overridden continue to use the standard layout.
Both strings and `pathlib.Path` objects are accepted.

Explicit inputs are validated. A missing explicit file or directory raises
`FileNotFoundError`; the code does not fall back to a similarly named standard
path. External inputs must still satisfy the existing data contracts:

- preprocessing images are TIFFs with the established tile/channel filename
  pattern and TileScanInfo XML schema;
- SpaceTX tile directories contain `Cycle*_s*_ch*.tif` plus one coordinate CSV;
- decoded transcript CSVs contain the columns required by the selected stage;
- segmentation labels are SciPy sparse `.npz` label images;
- AnnData inputs use the documented `.h5ad` conventions.

## Conventional workflow

Existing calls keep their behavior:

```python
preprocessing_main(raw_dirs, cycles, output_root, "czi", len(cycles))
make_spacetx_format(output_root, codebook_csv)
process_experiment(output_root)
cell_pose_segmentation_to_coo(output_root, "R1")
create_anndata_obj(output_root, "R1", "cellpose_stitched")
```

## Independent preprocessing stages

`preprocessing_main` now has `run_extraction`, `run_ome_conversion`,
`run_alignment`, and `run_retiling` switches. When extraction is disabled,
pass `region_directories`. Lower-level stages also accept per-region/cycle maps
whose keys are `(region_name, cycle_number)`:

```python
from ISS_preprocessing.preprocessing import (
    align_and_stitch,
    mipped_to_OME_tiffs,
    retile_stitched_images,
)

key = ("R1", 1)
mipped_to_OME_tiffs(
    [workspace / "R1"],
    [1],
    mipped_dirs={key: scratch / "mips"},
    metadata_files={key: imports / "tilescan.xml"},
    ome_output_files={key: exports / "registered-input.ome.tiff"},
    overwrite=True,
)

align_and_stitch(
    [workspace / "R1"],
    [1],
    n_total_cycles=1,
    ome_input_files={key: exports / "registered-input.ome.tiff"},
    stitched_output_dirs={key: exports / "stitched"},
    work_dirs={"R1": scratch / "ashlar"},
)

retile_stitched_images(
    [workspace / "R1"],
    [1],
    stitched_dirs={key: exports / "stitched"},
    retiled_dirs={key: exports / "tiles"},
)
```

The historical cache behavior remains the default. `overwrite=True` forces
recomputation. `align_and_stitch` now honors the supplied `maximum_shift` and
`filter_sigma` instead of replacing them internally.

## CARE with arbitrary cycle folders

Mappings use `(region_name, cycle_name)` keys. They reach the per-cycle file
consumer directly:

```python
ISS_CARE_predict(
    input_dir=None,
    model_dir=model_dir,
    model_name="CARE_model",
    dapi_ch=4,
    cycle_directories={("R3", "Cycle8"): imports / "cycle-eight"},
    input_tile_dirs={("R3", "Cycle8"): imports / "raw-tiles"},
    output_tile_dirs={("R3", "Cycle8"): exports / "care-tiles"},
    overwrite=True,
)
```

## SpaceTX and decoding outside the pipeline tree

SpaceTX accepts arbitrary region and tile directories while keeping the
existing TIFF/CSV schema:

```python
make_spacetx_format(
    input_dir=None,
    codebook_csv=codebook_csv,
    region_directories={"R5": imports / "sample-five"},
    tile_directories={"R5": {"Cycle9": imports / "cycle-nine-tiles"}},
    cycle_names_by_region={"R5": ["Cycle9"]},
    output_dirs={"R5": exports / "spacetx"},
    overwrite=True,
)
```

Decoding input, per-FOV cache, and final CSV can be placed independently. The
final filename is used exactly as supplied:

```python
process_experiment(
    input_dir=None,
    region_names=["sample-five"],
    spacetx_dirs={"sample-five": exports / "spacetx"},
    decoded_dirs={"sample-five": scratch / "fov-cache"},
    final_csv_files={"sample-five": exports / "transcripts.custom.csv"},
    overwrite=True,
)
```

Without `overwrite`, an existing final CSV is reused. With it, every FOV is
recomputed. Detector settings including `spot_detection_mode` and
`prob_threshold` now reach `ISS_pipeline`.

## Segmentation and AnnData from external files

Cellpose and StarDist accept an exact stitched image or an ordered list of
retiled images plus a coordinate CSV. Their historical behavior was to replace
outputs, so `overwrite` defaults to `True`; set it to `False` to reuse an
existing final mask.

```python
cell_pose_segmentation_to_coo(
    input_dir=None,
    region="sample-five",
    input_image_type="stitched",
    stitched_image_path=imports / "nuclei.tif",
    output_file=exports / "labels.custom.npz",
    overwrite=False,
)

adata = create_anndata_obj(
    input_dir=None,
    region=None,
    spots_path=exports / "transcripts.custom.csv",
    segmentation_path=exports / "labels.custom.npz",
    output_file=exports / "cells.custom.h5ad",
    overwrite=True,
)
```

AnnData output reuse remains the default (`overwrite=False`). Any one of its
three path arguments can be overridden while the others retain their defaults.
`concat_anndata` similarly accepts a partial `anndata_files` mapping.

PCIseq accepts `output_dir` for an exact destination. Its legacy behavior is to
replace serialized outputs, so `overwrite` defaults to `True`; use
`overwrite=False` to fail before a costly run when output already exists.

# Pipeline flexibility audit

This audit distinguishes user-facing filesystem conventions from constants that
belong to an external format or scientific algorithm.  Existing conventions
remain the default throughout the refactor.

| Area | Affected entry points | Rigid assumption | Targeted change | Risk |
| --- | --- | --- | --- | --- |
| Preprocessing orchestration | `preprocessing_main` | All four stages always run in one fixed order. | Add stage switches and allow caller-supplied region directories when extraction is skipped. | Medium: stage dependencies need explicit validation. |
| Preprocessing conversion | `mipped_to_OME_tiffs` | MIPs, metadata XML, and OME outputs must use the pipeline tree. | Add per-region/cycle input and output path maps plus `overwrite`. | Medium: OME metadata must remain unchanged. |
| Preprocessing stitching/retiling | `align_and_stitch`, `retile_stitched_images` | Inputs and outputs are inferred below each region; Ashlar tuning arguments are overwritten internally. | Add path maps/`overwrite`; honor caller tuning arguments. | High: these stages move or replace large image files. |
| CARE inference | `ISS_CARE_predict`, `process_one_cycle` | Regions/cycles and `4_retiled` are discovered only from the standard tree. | Accept explicit cycle input/output directories while retaining discovery defaults. | Medium: output provenance must refer to actual paths. |
| SpaceTX generation | `make_spacetx_format`, tile fetchers | Regions, cycles, tile directories, and output directories are inferred from the preprocessing tree. | Accept explicit region/tile/output mappings and configurable tile filenames; retain the current format contract. | High: Starfish JSON embeds tile paths. |
| Decoding | `process_experiment` | SpaceTX input and decoded output must share the same mirrored root; an existing final CSV always wins. `spot_detection_mode` and `prob_threshold` are not forwarded. | Accept explicit SpaceTX/output/final CSV maps, add `overwrite`, validate explicit inputs, and propagate detector options. | High: decoding is expensive and partial per-FOV caches must remain usable. |
| Image segmentation | `stardist_segmentation`, `cell_pose_segmentation_to_coo`, QC loaders | DAPI input is fixed to Cycle1 stitched/retiled names and output names are fixed. | Accept explicit stitched image, retiled image list, coordinate CSV, output directory/final file, and `overwrite`. | Medium: retiled input ordering must match coordinate rows. |
| AnnData creation | `create_anndata_obj`, `concat_anndata`, `create_ann_tiles` | Decoded spots, masks, and `.h5ad` files are inferred from one tree. | Accept explicit inputs/outputs, independent partial overrides, and recomputation control. | Low: computation is already separate once tables/masks are loaded. |
| PCIseq | `run_pciseq` | Output is always `<prefix>/<region>/PCIseq`; existing files are overwritten. | Accept an exact output directory and explicit overwrite behavior without changing PCIseq inputs. | Medium: PCIseq itself also writes through global config. |
| Transcript-aware/segmentation-free/domain/topology wrappers | `segmentation_wrappers.py`, `segmentation_free.py`, `spatial_domains.py`, `spatial_topology.py` | No blocking coupling found: these already accept tables/arrays or explicit paths, exact output locations, and overwrite controls. | Keep as-is; document them as examples of independent stages. | Low. |
| Training/data generation helpers | `ISS_CARE_datagen.py`, `ISS_CARE_training.py` | Inputs and model/patch outputs are already explicit at their public boundaries. | Keep internal filenames as format/provenance conventions. | Low. |
| Notebooks | notebooks under each package | Examples primarily demonstrate the conventional tree. | Preserve calls and add README examples for arbitrary locations rather than mechanically rewriting notebooks. | Low; notebook JSON edits are noisy. |

Legitimate constants deliberately retained include SpaceTX/Starfish JSON schema
names, external-tool command-line flags, required table column schemas, model
axes, and scientific numerical defaults.  The refactor does not claim that
arbitrary external formats are accepted: explicit files must still match the
documented TIFF, sparse-label, decoded-transcript, or AnnData schemas.

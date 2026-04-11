from __future__ import annotations

# -----------------------------------------------------------------------------
# GPU configuration for CARE inference
#
# CARE runs inference on a single GPU. On multi-GPU systems (e.g. A100 nodes),
# it is good practice to explicitly select a GPU to avoid:
#   - accidental initialization / memory reservation on all GPUs
#   - interference with other jobs on shared nodes
#   - non-reproducible GPU placement across runs
#
# IMPORTANT:
#   - Environment variables like CUDA_VISIBLE_DEVICES must be set BEFORE importing
#     TensorFlow or CSBDeep (csbdeep imports TensorFlow internally).
#
# Recommended practice in pipelines:
#   - Prefer setting CUDA_VISIBLE_DEVICES in the launcher (shell/Slurm), e.g.
#       CUDA_VISIBLE_DEVICES=0 python run_care.py
#   - As a fallback (or for interactive use), we set a default here ONLY if the
#     user has not already set CUDA_VISIBLE_DEVICES.
# -----------------------------------------------------------------------------

import os

# Only set a default GPU selection if the user hasn't already specified one.
# This prevents surprising behavior when running under Slurm or other schedulers.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# -----------------------------------------------------------------------------
# Optional: reduce TensorFlow logging verbosity
#
# 0 = all logs (default)
# 1 = INFO logs filtered
# 2 = INFO + WARNING logs filtered
# 3 = INFO + WARNING + ERROR logs filtered
#
# NOTE: keep this disabled during initial debugging.
# -----------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# -----------------------------------------------------------------------------
# TensorFlow import and memory configuration
# -----------------------------------------------------------------------------

import tensorflow as tf

# Enable memory growth so TensorFlow does NOT pre-allocate all GPU memory.
# This helps CARE "play nicely" with other processes and reduces startup cost.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import re
import shutil
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import numpy as np
import tifffile
from csbdeep.models import CARE


def file_exists_and_valid(path: Path, min_size: int = 1024) -> bool:
    """Fast corruption guard for outputs from previous runs."""
    try:
        return path.exists() and path.stat().st_size > int(min_size)
    except Exception:
        return False


def to_uint16_safe(
    arr: np.ndarray,
    *,
    context: str = "",
) -> np.ndarray:
    """
    Safely cast an image array to uint16.

    This function:
      - Detects NaN / ±Inf values
      - Replaces them with safe numeric values
      - Clips intensities to the uint16 range [0, 65535]
      - Emits a single [INFO] message only if correction was needed
    """
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()

    if has_nan or has_inf:
        print(
            f"[INFO] to_uint16_safe"
            f"{f' ({context})' if context else ''}: "
            f"nan={has_nan}, inf={has_inf}"
        )

    safe = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=65535.0,
        neginf=0.0,
    )

    safe = np.clip(safe, 0, 65535)

    return safe.astype(np.uint16)


def ISS_CARE_predict(
    input_dir,
    model_dir,
    model_name,
    dapi_ch,
    regions_to_process=None,
    output_dir_prefix=None,
):
    """
    Apply a pretrained CARE model to ISS images stored as channel-coded TIFF files.

    Args:
        input_dir (str or Path): Top-level experiment directory containing region folders (e.g., R1, R2).
        model_dir (str or Path): Path to the folder containing pretrained CSBDeep/CARE models.
        model_name (str): Name of the model to load.
        dapi_ch (int): 0-based channel index for DAPI. DAPI TIFFs end with `_ch{dapi_ch}.tif`
            or `_ch0*{dapi_ch}.tif` and are copied unchanged.
        regions_to_process (list[int] or None): 1-based region numbers to process (e.g., [1, 2]). None => all.
        output_dir_prefix (str or Path or None): If set, write outputs under this prefix while mirroring region names.

    Behavior:
        - Discovers region directories matching R\\d+.
        - For each region, finds Cycle* folders under <region>/preprocessing/.
        - For each cycle, reads TIFF files from: <region>/preprocessing/CycleX/4_retiled/
        - Writes CARE outputs to: <region>/preprocessing/CycleX/4_retiled/CARE/
          or under output_dir_prefix mirroring region names:
            <output_dir_prefix>/R#/preprocessing/CycleX/4_retiled/CARE/
        - CARE is applied to all TIFFs except DAPI (identified by `_ch{dapi_ch}.tif` or `_ch0*{dapi_ch}.tif`).
        - DAPI TIFFs are copied unchanged.
        - Any CSV metadata found in the input tile folder is copied alongside outputs.
        - The function never overwrites valid existing outputs; if outputs are partial, only missing/invalid files
          are generated/copied.
        - XML provenance:
            * We ONLY write an XML if this run actually wrote/copied any outputs (TIFF and/or CSV).
            * We NEVER overwrite existing XMLs: each productive run writes a uniquely-named XML.
            * Runs that find everything already exists produce NO XML.
    """

    def choose_n_tiles_yx(shape_yx):
        """
        Heuristic tiling selection for 2D CARE inference (axes="YX").

        n_tiles = (nY, nX) splits an image into nY tiles along Y and nX tiles along X.
        More tiles => lower peak GPU memory, but slightly higher overhead.
        """
        y, x = shape_yx
        m = max(y, x)

        if m <= 2048:
            return (1, 1)
        if m <= 4096:
            return (1, 2)
        if m <= 6144:
            return (2, 2)
        return (2, 4)

    # --- Fixed configuration (kept out of the function signature by request) ---
    preprocessing_relpath = "preprocessing"
    tiles_subpath = "4_retiled"
    axes = "YX"

    # NOTE: output dtype is intentionally fixed to uint16 for pipeline consistency.
    #       If you want float32 outputs later, we can add a separate mode with explicit naming.
    out_dtype = "uint16"

    input_dir = Path(input_dir)
    print(f"[INFO] Processing directory: {input_dir.resolve()}")

    model_dir = Path(model_dir)
    print(f"[INFO] Model base directory: {model_dir.resolve()}")
    print(f"[INFO] Model name: {model_name}")

    if output_dir_prefix is not None:
        output_dir_prefix = Path(output_dir_prefix)
        output_dir_prefix.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Using output_dir_prefix: {output_dir_prefix.resolve()}")
    else:
        print("[INFO] Using default output location under each region directory")

    # -------------------------------------------------------------------------
    # XML provenance run id
    #
    # We generate ONE run_id per ISS_CARE() invocation. If this invocation ends up
    # writing outputs for multiple cycles, each cycle will get its own XML file
    # stamped with the same run_id so they can be associated back to the same run.
    #
    # IMPORTANT: We do NOT write any XML for a cycle if it wrote nothing (i.e.
    # everything already existed and was valid).
    # -------------------------------------------------------------------------
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

    # --- Step 1: Find/select region directories matching R\d+ ---
    region_pattern = re.compile(r"^R(\d+)$")

    regions_found = []
    for r in input_dir.iterdir():
        if not r.is_dir():
            continue
        m = region_pattern.match(r.name)
        if m:
            regions_found.append((int(m.group(1)), r))

    regions_found.sort(key=lambda t: t[0])

    if not regions_found:
        raise RuntimeError(f"No regions found in {input_dir} (expected folders like R1, R2, ...)")

    available_numbers = [n for n, _ in regions_found]
    available_map = {n: p for n, p in regions_found}

    all_regions = [f"R{n}" for n in available_numbers]
    print(f"[INFO] Regions found on disk ({len(all_regions)}): {all_regions}")

    # --- Select regions to process ---
    if regions_to_process is None:
        region_numbers = available_numbers
    else:
        if not isinstance(regions_to_process, (list, tuple)):
            raise TypeError("regions_to_process must be a list of 1-based ints, e.g. [1, 2].")

        region_numbers = [int(x) for x in regions_to_process]
        if any(x < 1 for x in region_numbers):
            raise ValueError(f"regions_to_process contains invalid region numbers: {regions_to_process}")

        missing = [n for n in region_numbers if n not in available_map]
        if missing:
            raise FileNotFoundError(
                f"Requested region(s) not found: {[f'R{n}' for n in missing]}. "
                f"Available regions: {all_regions}"
            )

    region_directories = [available_map[n] for n in region_numbers]

    selected_regions = [f"R{n}" for n in region_numbers]
    skipped_regions = [r for r in all_regions if r not in selected_regions]

    print(f"[INFO] Regions selected ({len(selected_regions)}): {selected_regions}")
    if skipped_regions:
        print(f"[INFO] Regions skipped ({len(skipped_regions)}): {skipped_regions}")

    # --- Step 2: Load CARE model once ---
    model = CARE(config=None, name=model_name, basedir=str(model_dir))
    print(f"[INFO] Loaded CARE model: {model.name}")

    # Accepts both `_ch4.tif` and `_ch04.tif` (any number of leading zeros).
    dapi_suffix_re = re.compile(
        rf"_ch0*{int(dapi_ch)}\.(tif|tiff)$",
        re.IGNORECASE,
    )

    def _write_care_xml(
        xml_path,
        *,
        region_name,
        cycle_name,
        in_tile_dir,
        out_tile_dir,
        n_pred,
        n_copy,
        n_skip,
        n_tiles,
        probe_shape,
    ):
        """
        Write a minimal provenance XML capturing CARE settings and context.

        Policy:
          - Caller must ensure this is only invoked when this run wrote/copied outputs.
          - File name should be unique per productive run (no overwrites).
        """
        root = Element("care_run")

        SubElement(root, "timestamp_utc").text = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        SubElement(root, "run_id").text = str(run_id)  # ties cycle XMLs back to the same ISS_CARE() run
        SubElement(root, "region").text = str(region_name)
        SubElement(root, "cycle").text = str(cycle_name)

        paths = SubElement(root, "paths")
        SubElement(paths, "input_tile_dir").text = str(in_tile_dir)
        SubElement(paths, "output_tile_dir").text = str(out_tile_dir)
        SubElement(paths, "model_dir").text = str(model_dir)
        SubElement(paths, "model_name").text = str(model_name)

        params = SubElement(root, "parameters")
        SubElement(params, "axes").text = str(axes)
        SubElement(params, "n_tiles").text = f"{n_tiles[0]},{n_tiles[1]}"
        SubElement(params, "out_dtype").text = str(out_dtype)
        SubElement(params, "dapi_ch").text = str(int(dapi_ch))
        SubElement(params, "dapi_filename_suffix_regex").text = dapi_suffix_re.pattern

        img = SubElement(root, "image")
        SubElement(img, "probe_shape_yx").text = f"{probe_shape[0]},{probe_shape[1]}"

        counts = SubElement(root, "counts")
        SubElement(counts, "predicted").text = str(int(n_pred))
        SubElement(counts, "copied_dapi").text = str(int(n_copy))
        SubElement(counts, "skipped_existing").text = str(int(n_skip))

        meta = SubElement(root, "model_metadata")
        try:
            SubElement(meta, "csbdeep_model_name").text = str(model.name)
        except Exception:
            SubElement(meta, "csbdeep_model_name").text = ""

        try:
            cfg = getattr(model, "config", None)
            if cfg is not None:
                SubElement(meta, "model_axes").text = str(getattr(cfg, "axes", ""))
                SubElement(meta, "model_n_channel_in").text = str(getattr(cfg, "n_channel_in", ""))
                SubElement(meta, "model_n_channel_out").text = str(getattr(cfg, "n_channel_out", ""))
        except Exception:
            pass

        xml_path.parent.mkdir(parents=True, exist_ok=True)
        ElementTree(root).write(str(xml_path), encoding="utf-8", xml_declaration=True)
        print(f" CARE XML written to: {xml_path}")

    # --- Step 3: Process each region ---
    for region_directory in region_directories:
        region_name = region_directory.name

        width = 80
        print("=" * width + "\033[0m")
        print(f"\033[1;90mProcessing region: {region_name}\033[0m")

        preprocessing_root = region_directory / preprocessing_relpath
        if not preprocessing_root.exists():
            print(f"[{region_name}] [WARN] Missing preprocessing folder, skipping: {preprocessing_root}")
            continue

        # Find cycles (Cycle1, Cycle2, ...)
        cycle_pattern = re.compile(r"^Cycle(\d+)$")
        cycles_found = []
        for c in preprocessing_root.iterdir():
            if not c.is_dir():
                continue
            m = cycle_pattern.match(c.name)
            if m:
                cycles_found.append((int(m.group(1)), c))
        cycles_found.sort(key=lambda t: t[0])

        if not cycles_found:
            print(f"[{region_name}] [WARN] No Cycle* folders found under: {preprocessing_root}")
            continue

        print(f"[{region_name}] Cycles found: {[c.name for _, c in cycles_found]}")

        # --- Step 3a: Iterate cycles and process their retiled images ---
        for _, cycle_dir in cycles_found:
            cycle_name = cycle_dir.name

            # Input cycle directory (always under the original region directory)
            in_cycle_dir = cycle_dir

            # Output cycle directory:
            # - in-place if output_dir_prefix is None
            # - mirrored under output_dir_prefix/R#/... otherwise
            if output_dir_prefix is None:
                out_cycle_dir = in_cycle_dir
            else:
                out_cycle_dir = (
                    Path(output_dir_prefix)
                    / region_name
                    / preprocessing_relpath
                    / cycle_name
                )

            # Input folder containing channel-coded TIFFs
            in_tile_dir = in_cycle_dir / tiles_subpath
            if not in_tile_dir.exists():
                print(f"[{region_name} | {cycle_name}] [WARN] Missing tile folder, skipping: {in_tile_dir}")
                continue

            # Output folder inside the cycle's 4_retiled directory:
            # <...>/preprocessing/CycleX/4_retiled/CARE/
            out_tile_dir = out_cycle_dir / tiles_subpath / "CARE"
            out_tile_dir.mkdir(parents=True, exist_ok=True)

            # Collect all TIFFs in the input folder (defines expected outputs)
            in_tifs = sorted(
                [p for p in in_tile_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")],
                key=lambda p: p.name,
            )

            if not in_tifs:
                print(f"[{region_name} | {cycle_name}] [WARN] No TIFFs found, skipping.")
                continue

            # Compute expected outputs and evaluate "complete" status using a validity guard
            expected_out_paths = [out_tile_dir / p.name for p in in_tifs]
            all_outputs_valid = all(file_exists_and_valid(p) for p in expected_out_paths)

            # Choose tiling ONCE per cycle based on a representative non-DAPI image
            probe_path = next((p for p in in_tifs if not dapi_suffix_re.search(p.name)), in_tifs[0])
            probe_img = tifffile.imread(str(probe_path))
            n_tiles = choose_n_tiles_yx(probe_img.shape)

            # Informative per-cycle log for debugging and performance tuning
            y, x = probe_img.shape
            ty, tx = n_tiles
            print(
                f"[{region_name} | {cycle_name}] CARE tiling: image {y}×{x} (Y×X) → "
                f"{ty}×{tx} tiles (≈{y//ty}×{x//tx} px/tile)"
            )

            # EARLY EXIT: never overwrite if all expected outputs already exist AND look valid
            #
            # Provenance policy requested:
            #   - If everything already exists/valid, we do NOTHING and write NO XML.
            #     (Because we didn't generate anything in this run.)
            if all_outputs_valid:
                print(f"[{region_name} | {cycle_name}] Skipping: all expected outputs already exist and look valid.")
                print(f"  ✔ {out_tile_dir}")
                continue

            print(f"[{region_name} | {cycle_name}] Input  -> {in_tile_dir}")
            print(f"[{region_name} | {cycle_name}] Output -> {out_tile_dir}")

            # Track whether this run wrote/copied ANY outputs for this cycle.
            # If False at the end, we do NOT write an XML for this cycle.
            wrote_anything = False

            # Process each TIFF:
            # - If a valid output exists, skip it
            # - DAPI: copy unchanged
            # - Non-DAPI: CARE predict and write
            n_pred, n_copy, n_skip = 0, 0, 0
            for tif_path in in_tifs:
                out_path = out_tile_dir / tif_path.name

                # Skip if output exists and passes the "fast validity" check
                if file_exists_and_valid(out_path):
                    n_skip += 1
                    continue

                # DAPI is copied unchanged to avoid degrading nuclei signal
                if dapi_suffix_re.search(tif_path.name):
                    shutil.copyfile(tif_path, out_path)
                    n_copy += 1
                    wrote_anything = True  # we created/updated an output file in this run
                    continue

                # Read, predict, and write CARE output
                x_in = tifffile.imread(str(tif_path))
                restored = model.predict(x_in, axes=axes, n_tiles=n_tiles)

                # Always write a clean uint16 TIFF for pipeline consistency
                restored_u16 = to_uint16_safe(restored, context=f"{region_name}/{cycle_name}/{tif_path.name}")
                tifffile.imwrite(str(out_path), restored_u16)
                n_pred += 1
                wrote_anything = True  # we created/updated an output file in this run

            print(
                f"[{region_name} | {cycle_name}] Done: "
                f"predicted={n_pred}, copied_dapi={n_copy}, skipped_existing={n_skip}"
            )

            # Copy CSV metadata files alongside outputs if they don't already exist (or are invalid)
            for csv_path in in_tile_dir.glob("*.csv"):
                dst_csv = out_tile_dir / csv_path.name
                if file_exists_and_valid(dst_csv, min_size=64):
                    continue
                try:
                    shutil.copyfile(csv_path, dst_csv)
                    wrote_anything = True  # copying metadata counts as output generation for provenance
                except Exception as e:
                    print(f"[{region_name} | {cycle_name}] [WARN] Failed to copy {csv_path.name}: {e}")

            # Write provenance XML for this cycle ONLY IF this run produced outputs.
            # - We DO NOT overwrite existing XMLs.
            # - Each productive run writes a uniquely named XML including run_id.
            if wrote_anything:
                # Unique per-run XML filename (no overwrites). One per cycle per productive run.
                xml_path = out_tile_dir / f"CARE_run_{run_id}.xml"
                _write_care_xml(
                    xml_path,
                    region_name=region_name,
                    cycle_name=cycle_name,
                    in_tile_dir=in_tile_dir,
                    out_tile_dir=out_tile_dir,
                    n_pred=n_pred,
                    n_copy=n_copy,
                    n_skip=n_skip,
                    n_tiles=n_tiles,
                    probe_shape=probe_img.shape,
                )

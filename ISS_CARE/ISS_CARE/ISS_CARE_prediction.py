from __future__ import annotations

# -----------------------------------------------------------------------------
# GPU configuration for CARE inference
#
# Simple, automatic behavior:
# - choose one GPU before importing TensorFlow / CSBDeep
# - print a short informative line
# - enable TensorFlow memory growth
#
# IMPORTANT:
#   GPU selection must happen BEFORE importing tensorflow or csbdeep.
# -----------------------------------------------------------------------------

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def choose_gpu_for_rl(
    preferred_max_mem_mb: int = 2000,
    preferred_max_util: int = 20,
) -> int | None:
    """
    Select one GPU using a simple "prefer free GPU" strategy.

    Priority
    --------
    1. Prefer GPUs with:
         - memory.used <= preferred_max_mem_mb
         - utilization.gpu <= preferred_max_util
    2. Otherwise choose the GPU with the lowest memory use.
    3. Break ties by lower utilization, then lower GPU index.

    Side effects
    ------------
    Sets:
      - CUDA_VISIBLE_DEVICES
      - PYOPENCL_CTX

    Returns
    -------
    int | None
        Selected physical GPU index, or None if selection failed.
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,nounits,noheader",
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace").strip()

        if not result:
            print("[WARN] nvidia-smi returned no GPU information.")
            return None

        rows: list[tuple[int, int, int]] = []
        for line in result.splitlines():
            idx, mem, util = [x.strip() for x in line.split(",")]
            rows.append((int(idx), int(mem), int(util)))

        if not rows:
            print("[WARN] No GPUs parsed from nvidia-smi output.")
            return None

        preferred = [
            row for row in rows
            if row[1] <= preferred_max_mem_mb and row[2] <= preferred_max_util
        ]

        candidates = preferred if preferred else rows
        candidates.sort(key=lambda x: (x[1], x[2], x[0]))
        gpu_id, mem_mb, util_pct = candidates[0]

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["PYOPENCL_CTX"] = f"0:{gpu_id}"

        print(f"[INFO] Selected GPU {gpu_id} (mem={mem_mb} MiB, util={util_pct}%)")
        return gpu_id

    except Exception as e:
        print(f"[WARN] Automatic GPU selection failed: {e}")
        print("[WARN] Proceeding without forcing CUDA_VISIBLE_DEVICES.")
        return None


# Must run before TensorFlow import.
choose_gpu_for_rl()

# -----------------------------------------------------------------------------
# TensorFlow import and memory configuration
# -----------------------------------------------------------------------------

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] TensorFlow sees {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[WARN] Could not set TensorFlow memory growth: {e}")
else:
    print("[INFO] TensorFlow sees no GPU. Inference will run on CPU.")

# -----------------------------------------------------------------------------
# Remaining imports
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from csbdeep.models import CARE


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def file_exists_and_valid(path: Path, min_size: int = 1024) -> bool:
    """Fast corruption guard for outputs from previous runs."""
    try:
        return path.exists() and path.stat().st_size > int(min_size)
    except Exception:
        return False


def load_json_file(path: Path | str) -> dict | None:
    """Load a JSON file safely. Returns None if missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read JSON file {path}: {e}")
        return None


def _print_array_stats(arr: np.ndarray, label: str) -> None:
    """Print concise debug statistics for an array."""
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        print(
            f"[DEBUG] {label}: dtype={arr.dtype}, shape={arr.shape}, "
            "all values are non-finite"
        )
        return

    p1, p50, p99 = np.percentile(finite, [1, 50, 99])

    print(
        f"[DEBUG] {label}: "
        f"dtype={arr.dtype}, shape={arr.shape}, "
        f"min={float(np.min(finite)):.6g}, "
        f"max={float(np.max(finite)):.6g}, "
        f"mean={float(np.mean(finite)):.6g}, "
        f"std={float(np.std(finite)):.6g}, "
        f"p1={float(p1):.6g}, "
        f"p50={float(p50):.6g}, "
        f"p99={float(p99):.6g}"
    )


def normalize_percentile(
    arr: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.8,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Percentile-normalize a single image to [0, 1].

    Use this only if the CARE model was trained with the same normalization.
    """
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.percentile(arr, pmin)
    hi = np.percentile(arr, pmax)
    return np.clip((arr - lo) / (hi - lo + eps), 0, 1)


def to_uint16_safe(
    arr: np.ndarray,
    *,
    context: str = "",
) -> np.ndarray:
    """
    Safely cast an image array to uint16.

    This function:
      - detects NaN / Inf values
      - replaces them safely
      - clips to uint16 range [0, 65535]
      - logs only if correction was needed
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


def _safe_percentile_pair(arr: np.ndarray, pmin: float = 1.0, pmax: float = 99.8) -> tuple[float, float]:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, pmin))
    hi = float(np.percentile(finite, pmax))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _display_rescale(arr: np.ndarray, pmin: float = 1.0, pmax: float = 99.8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = _safe_percentile_pair(arr, pmin=pmin, pmax=pmax)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _fraction_positive(arr: np.ndarray, threshold: float = 0.0) -> float:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite > threshold))


def _print_scale_checks(
    *,
    raw_input: np.ndarray,
    model_input: np.ndarray,
    restored_float: np.ndarray,
    restored_u16: np.ndarray,
    context: str,
    normalize_input: bool,
) -> None:
    """
    Print extra debug checks aimed at catching scaling / normalization mistakes.
    """
    raw_max = float(np.nanmax(raw_input)) if np.size(raw_input) else 0.0
    model_in_max = float(np.nanmax(model_input)) if np.size(model_input) else 0.0
    pred_max = float(np.nanmax(restored_float)) if np.size(restored_float) else 0.0
    pred_mean = float(np.nanmean(restored_float)) if np.size(restored_float) else 0.0
    out_u16_max = int(np.max(restored_u16)) if np.size(restored_u16) else 0

    print(f"[DEBUG] scale_check ({context})")
    print(f"[DEBUG]   normalize_input={normalize_input}")
    print(f"[DEBUG]   raw_input_max={raw_max:.6g}")
    print(f"[DEBUG]   model_input_max={model_in_max:.6g}")
    print(f"[DEBUG]   prediction_float_max={pred_max:.6g}")
    print(f"[DEBUG]   prediction_float_mean={pred_mean:.6g}")
    print(f"[DEBUG]   saved_uint16_max={out_u16_max}")
    print(
        f"[DEBUG]   raw_fraction_gt0={_fraction_positive(raw_input):.4f}, "
        f"model_input_fraction_gt0={_fraction_positive(model_input):.4f}, "
        f"prediction_fraction_gt0={_fraction_positive(restored_float):.4f}"
    )

    if normalize_input and model_in_max > 1.05:
        print(
            f"[WARN] {context}: normalized model input has max > 1.05 "
            f"({model_in_max:.4g}). Check normalization."
        )

    if normalize_input and pred_max < 100:
        print(
            f"[WARN] {context}: prediction max is very low after rescaling "
            f"({pred_max:.4g}). This can indicate a normalization / scale mismatch."
        )

    if not normalize_input and pred_max <= 1.5:
        print(
            f"[WARN] {context}: prediction max is near 0-1 range without normalization "
            f"({pred_max:.4g}). The model may have been trained with normalized input."
        )

    if out_u16_max < 50:
        print(
            f"[WARN] {context}: saved uint16 output max is very low "
            f"({out_u16_max}). Output may be severely compressed."
        )


def save_debug_visualization(
    *,
    raw_input: np.ndarray,
    model_input: np.ndarray,
    restored_float: np.ndarray,
    restored_u16: np.ndarray,
    vis_path: Path,
    context: str,
    normalize_input: bool,
) -> None:
    """
    Save a quick-look debug figure with raw input, model input, prediction, and histogram.

    This is intended to quickly reveal scaling mistakes and overly dim predictions.
    """
    vis_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(_display_rescale(raw_input), cmap="gray")
    axes[0, 0].set_title("Raw input (display-scaled)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_display_rescale(model_input), cmap="gray")
    axes[0, 1].set_title("Model input (display-scaled)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(_display_rescale(restored_float), cmap="gray")
    axes[0, 2].set_title("Prediction float (display-scaled)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(_display_rescale(restored_u16), cmap="gray")
    axes[1, 0].set_title("Saved uint16 output")
    axes[1, 0].axis("off")

    raw_vals = np.asarray(raw_input)[np.isfinite(raw_input)].ravel()
    pred_vals = np.asarray(restored_float)[np.isfinite(restored_float)].ravel()

    if raw_vals.size > 0:
        axes[1, 1].hist(raw_vals, bins=100, alpha=0.7, label="raw")
    if pred_vals.size > 0:
        axes[1, 1].hist(pred_vals, bins=100, alpha=0.7, label="pred")
    axes[1, 1].set_title("Raw vs prediction histogram")
    axes[1, 1].legend()

    text = [
        context,
        f"normalize_input={normalize_input}",
        f"raw max={float(np.nanmax(raw_input)):.6g}",
        f"model input max={float(np.nanmax(model_input)):.6g}",
        f"pred float max={float(np.nanmax(restored_float)):.6g}",
        f"pred u16 max={int(np.max(restored_u16))}",
    ]
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 1.0, "\n".join(text), va="top", family="monospace")

    fig.tight_layout()
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Debug visualization written: {vis_path}")


# -----------------------------------------------------------------------------
# Metadata / normalization helpers
# -----------------------------------------------------------------------------

def find_training_metadata_file(model_dir: Path | str, model_name: str) -> Path | None:
    """
    Look for training metadata in the model output directory.

    Expected location:
        <model_dir>/<model_name>/training_metadata.json
    """
    model_dir = Path(model_dir)
    candidate = model_dir / model_name / "training_metadata.json"
    if candidate.exists():
        return candidate
    return None


def resolve_inference_normalization_settings(
    *,
    model_dir: Path | str,
    model_name: str,
    normalize_input: bool | None,
    normalization_pmin: float | None,
    normalization_pmax: float | None,
    normalization_eps: float | None,
) -> tuple[bool, float, float, float, Path | None, dict | None]:
    """
    Resolve inference normalization settings.

    Priority
    --------
    1. If normalize_input is explicitly True/False, keep that choice.
    2. If normalize_input is None, try to infer from training_metadata.json.
    3. If metadata is unavailable, default to False.

    If normalization params are None, try to load them from training metadata.
    Otherwise use standard defaults.
    """
    metadata_file = find_training_metadata_file(model_dir, model_name)
    metadata = load_json_file(metadata_file) if metadata_file is not None else None

    metadata_norm = {}
    if isinstance(metadata, dict):
        metadata_norm = metadata.get("normalization", {}) or {}

    if normalize_input is None:
        normalize_input_resolved = bool(metadata_norm.get("ENABLED", False))
    else:
        normalize_input_resolved = bool(normalize_input)

    pmin = normalization_pmin
    pmax = normalization_pmax
    eps = normalization_eps

    if pmin is None:
        pmin = metadata_norm.get("PMIN", 1.0)
    if pmax is None:
        pmax = metadata_norm.get("PMAX", 99.8)
    if eps is None:
        eps = metadata_norm.get("EPS", 1e-8)

    if pmin is None:
        pmin = 1.0
    if pmax is None:
        pmax = 99.8
    if eps is None:
        eps = 1e-8

    return (
        normalize_input_resolved,
        float(pmin),
        float(pmax),
        float(eps),
        metadata_file,
        metadata,
    )


# -----------------------------------------------------------------------------
# Discovery helpers
# -----------------------------------------------------------------------------

def discover_regions(input_dir: Path) -> tuple[list[int], dict[int, Path]]:
    """Find region folders named R1, R2, ..."""
    region_pattern = re.compile(r"^R(\d+)$")
    regions_found: list[tuple[int, Path]] = []

    for path in input_dir.iterdir():
        if not path.is_dir():
            continue
        m = region_pattern.match(path.name)
        if m:
            regions_found.append((int(m.group(1)), path))

    regions_found.sort(key=lambda t: t[0])

    if not regions_found:
        raise RuntimeError(f"No regions found in {input_dir} (expected folders like R1, R2, ...)")

    available_numbers = [n for n, _ in regions_found]
    available_map = {n: p for n, p in regions_found}
    return available_numbers, available_map


def select_region_directories(
    available_numbers: list[int],
    available_map: dict[int, Path],
    regions_to_process,
) -> tuple[list[int], list[Path]]:
    """Resolve requested region numbers into directories."""
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
            all_regions = [f"R{n}" for n in available_numbers]
            raise FileNotFoundError(
                f"Requested region(s) not found: {[f'R{n}' for n in missing]}. "
                f"Available regions: {all_regions}"
            )

    region_directories = [available_map[n] for n in region_numbers]
    return region_numbers, region_directories


def discover_cycles(preprocessing_root: Path) -> list[tuple[int, Path]]:
    """Find cycle folders named Cycle1, Cycle2, ..."""
    cycle_pattern = re.compile(r"^Cycle(\d+)$")
    cycles_found: list[tuple[int, Path]] = []

    for path in preprocessing_root.iterdir():
        if not path.is_dir():
            continue
        m = cycle_pattern.match(path.name)
        if m:
            cycles_found.append((int(m.group(1)), path))

    cycles_found.sort(key=lambda t: t[0])
    return cycles_found


def choose_n_tiles_yx(shape_yx: tuple[int, int]) -> tuple[int, int]:
    """
    Heuristic tiling selection for 2D CARE inference (axes='YX').

    More tiles reduce peak GPU memory at the cost of some overhead.
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


# -----------------------------------------------------------------------------
# Output / XML helpers
# -----------------------------------------------------------------------------

def write_care_xml(
    xml_path: Path,
    *,
    run_id: str,
    region_name: str,
    cycle_name: str,
    in_tile_dir: Path,
    out_tile_dir: Path,
    model_dir: Path,
    model_name: str,
    training_metadata_file: Path | None,
    axes: str,
    n_tiles: tuple[int, int],
    out_dtype: str,
    dapi_ch: int,
    dapi_suffix_pattern: str,
    normalize_input: bool,
    normalization_pmin: float,
    normalization_pmax: float,
    normalization_eps: float,
    debug_prints: bool,
    debug_print_limit: int,
    visualize_debug_predictions: bool,
    overwrite: bool,
    n_pred: int,
    n_copy: int,
    n_skip: int,
    probe_shape: tuple[int, int],
    model: CARE,
) -> None:
    """Write a compact provenance XML for one productive cycle run."""
    root = Element("care_run")

    SubElement(root, "timestamp_utc").text = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    SubElement(root, "run_id").text = str(run_id)
    SubElement(root, "region").text = str(region_name)
    SubElement(root, "cycle").text = str(cycle_name)

    paths = SubElement(root, "paths")
    SubElement(paths, "input_tile_dir").text = str(in_tile_dir)
    SubElement(paths, "output_tile_dir").text = str(out_tile_dir)
    SubElement(paths, "model_dir").text = str(model_dir)
    SubElement(paths, "model_name").text = str(model_name)
    SubElement(paths, "training_metadata_file").text = "" if training_metadata_file is None else str(training_metadata_file)

    params = SubElement(root, "parameters")
    SubElement(params, "axes").text = str(axes)
    SubElement(params, "n_tiles").text = f"{n_tiles[0]},{n_tiles[1]}"
    SubElement(params, "out_dtype").text = str(out_dtype)
    SubElement(params, "dapi_ch").text = str(int(dapi_ch))
    SubElement(params, "dapi_filename_suffix_regex").text = str(dapi_suffix_pattern)
    SubElement(params, "normalize_input").text = str(bool(normalize_input))
    SubElement(params, "normalization_pmin").text = str(normalization_pmin)
    SubElement(params, "normalization_pmax").text = str(normalization_pmax)
    SubElement(params, "normalization_eps").text = str(normalization_eps)
    SubElement(params, "rescale_prediction_by_65535").text = str(bool(normalize_input))
    SubElement(params, "debug_prints").text = str(bool(debug_prints))
    SubElement(params, "debug_print_limit").text = str(int(debug_print_limit))
    SubElement(params, "visualize_debug_predictions").text = str(bool(visualize_debug_predictions))
    SubElement(params, "overwrite").text = str(bool(overwrite))
    SubElement(params, "normalization_inferred_from_metadata").text = str(training_metadata_file is not None)

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
    print(f"[INFO] XML written: {xml_path}")


# -----------------------------------------------------------------------------
# Per-file / per-cycle processing
# -----------------------------------------------------------------------------

def predict_one_image(
    *,
    tif_path: Path,
    out_path: Path,
    model: CARE,
    axes: str,
    n_tiles: tuple[int, int],
    normalize_input: bool,
    normalization_pmin: float,
    normalization_pmax: float,
    normalization_eps: float,
    debug_this_file: bool,
    debug_context: str,
    debug_vis_path: Path | None = None,
) -> None:
    """Load one image, optionally normalize, run CARE, and save uint16 output."""
    x_in = tifffile.imread(str(tif_path)).astype(np.float32)

    if normalize_input:
        x_in_model = normalize_percentile(
            x_in,
            pmin=normalization_pmin,
            pmax=normalization_pmax,
            eps=normalization_eps,
        )
    else:
        x_in_model = x_in

    restored = model.predict(x_in_model, axes=axes, n_tiles=n_tiles)

    # If normalized inference is used, restore back to uint16-like range.
    if normalize_input:
        restored = restored * 65535.0

    if debug_this_file:
        print(f"[DEBUG] {debug_context}")
        _print_array_stats(x_in, "raw_input")
        if normalize_input:
            _print_array_stats(x_in_model, "normalized_input")
        else:
            _print_array_stats(x_in_model, "model_input_raw")
        _print_array_stats(restored, "prediction_before_uint16")

    restored_u16 = to_uint16_safe(restored, context=debug_context)

    if debug_this_file:
        _print_array_stats(restored_u16, "saved_uint16_output")
        _print_scale_checks(
            raw_input=x_in,
            model_input=x_in_model,
            restored_float=restored,
            restored_u16=restored_u16,
            context=debug_context,
            normalize_input=normalize_input,
        )

        if debug_vis_path is not None:
            save_debug_visualization(
                raw_input=x_in,
                model_input=x_in_model,
                restored_float=restored,
                restored_u16=restored_u16,
                vis_path=debug_vis_path,
                context=debug_context,
                normalize_input=normalize_input,
            )

    tifffile.imwrite(str(out_path), restored_u16)


def copy_cycle_csvs(in_tile_dir: Path, out_tile_dir: Path, *, overwrite: bool) -> int:
    """
    Copy CSV sidecar files.

    overwrite=False:
        copy only when destination is missing or invalid
    overwrite=True:
        always replace destination CSVs
    """
    n_csv_copied = 0
    for csv_path in in_tile_dir.glob("*.csv"):
        dst_csv = out_tile_dir / csv_path.name

        if not overwrite and file_exists_and_valid(dst_csv, min_size=64):
            continue

        try:
            shutil.copyfile(csv_path, dst_csv)
            n_csv_copied += 1
        except Exception as e:
            print(f"[WARN] Failed to copy {csv_path.name}: {e}")
    return n_csv_copied


def process_one_cycle(
    *,
    region_name: str,
    cycle_name: str,
    cycle_dir: Path,
    out_cycle_dir: Path,
    model: CARE,
    model_dir: Path,
    model_name: str,
    training_metadata_file: Path | None,
    dapi_ch: int,
    normalize_input: bool,
    normalization_pmin: float,
    normalization_pmax: float,
    normalization_eps: float,
    debug_prints: bool,
    debug_print_limit: int,
    visualize_debug_predictions: bool,
    overwrite: bool,
    run_id: str,
    preprocessing_relpath: str,
    tiles_subpath: str,
    axes: str,
    out_dtype: str,
) -> None:
    """Process one Cycle folder end-to-end."""
    del preprocessing_relpath  # kept for signature clarity if later extended

    in_tile_dir = cycle_dir / tiles_subpath
    if not in_tile_dir.exists():
        print(f"[WARN] {region_name}/{cycle_name}: missing tile folder, skipping: {in_tile_dir}")
        return

    out_tile_dir = out_cycle_dir / tiles_subpath / "CARE"
    out_tile_dir.mkdir(parents=True, exist_ok=True)

    debug_vis_dir = out_tile_dir / "_debug_vis"
    if visualize_debug_predictions:
        debug_vis_dir.mkdir(parents=True, exist_ok=True)

    in_tifs = sorted(
        [p for p in in_tile_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: p.name,
    )

    if not in_tifs:
        print(f"[WARN] {region_name}/{cycle_name}: no TIFFs found, skipping.")
        return

    print(f"[INFO] {region_name}/{cycle_name}: {len(in_tifs)} TIFF(s) found")

    dapi_suffix_re = re.compile(
        rf"_ch0*{int(dapi_ch)}\.(tif|tiff)$",
        re.IGNORECASE,
    )

    expected_out_paths = [out_tile_dir / p.name for p in in_tifs]

    # Skip whole cycle only when overwrite is False and every output already looks valid.
    all_outputs_valid = all(file_exists_and_valid(p) for p in expected_out_paths)
    if not overwrite and all_outputs_valid:
        print(f"[INFO] {region_name}/{cycle_name}: all outputs already exist and look valid, skipping.")
        return

    probe_path = next((p for p in in_tifs if not dapi_suffix_re.search(p.name)), in_tifs[0])
    probe_img = tifffile.imread(str(probe_path))
    n_tiles = choose_n_tiles_yx(probe_img.shape)

    y, x = probe_img.shape
    ty, tx = n_tiles
    print(
        f"[INFO] {region_name}/{cycle_name}: tiling {ty}x{tx} "
        f"for image {y}x{x} (YxX)"
    )

    print(f"[INFO] {region_name}/{cycle_name}: input  -> {in_tile_dir}")
    print(f"[INFO] {region_name}/{cycle_name}: output -> {out_tile_dir}")
    print(f"[INFO] {region_name}/{cycle_name}: overwrite={overwrite}")
    print(f"[INFO] {region_name}/{cycle_name}: visualize_debug_predictions={visualize_debug_predictions}")

    wrote_anything = False
    n_pred = 0
    n_copy = 0
    n_skip = 0
    n_debug_printed = 0

    for tif_path in in_tifs:
        out_path = out_tile_dir / tif_path.name

        if not overwrite and file_exists_and_valid(out_path):
            n_skip += 1
            continue

        if dapi_suffix_re.search(tif_path.name):
            shutil.copyfile(tif_path, out_path)
            n_copy += 1
            wrote_anything = True
            continue

        debug_this_file = debug_prints and (n_debug_printed < debug_print_limit)

        debug_vis_path = None
        if debug_this_file and visualize_debug_predictions:
            debug_vis_path = debug_vis_dir / f"{tif_path.stem}__debug.png"

        predict_one_image(
            tif_path=tif_path,
            out_path=out_path,
            model=model,
            axes=axes,
            n_tiles=n_tiles,
            normalize_input=normalize_input,
            normalization_pmin=normalization_pmin,
            normalization_pmax=normalization_pmax,
            normalization_eps=normalization_eps,
            debug_this_file=debug_this_file,
            debug_context=f"{region_name}/{cycle_name}/{tif_path.name}",
            debug_vis_path=debug_vis_path,
        )

        if debug_this_file:
            n_debug_printed += 1

        n_pred += 1
        wrote_anything = True

    n_csv_copied = copy_cycle_csvs(
        in_tile_dir=in_tile_dir,
        out_tile_dir=out_tile_dir,
        overwrite=overwrite,
    )
    if n_csv_copied > 0:
        wrote_anything = True

    print(
        f"[INFO] {region_name}/{cycle_name}: "
        f"predicted={n_pred}, copied_dapi={n_copy}, skipped_existing={n_skip}, copied_csv={n_csv_copied}"
    )

    if wrote_anything:
        xml_path = out_tile_dir / f"CARE_run_{run_id}.xml"
        write_care_xml(
            xml_path=xml_path,
            run_id=run_id,
            region_name=region_name,
            cycle_name=cycle_name,
            in_tile_dir=in_tile_dir,
            out_tile_dir=out_tile_dir,
            model_dir=model_dir,
            model_name=model_name,
            training_metadata_file=training_metadata_file,
            axes=axes,
            n_tiles=n_tiles,
            out_dtype=out_dtype,
            dapi_ch=dapi_ch,
            dapi_suffix_pattern=dapi_suffix_re.pattern,
            normalize_input=normalize_input,
            normalization_pmin=normalization_pmin,
            normalization_pmax=normalization_pmax,
            normalization_eps=normalization_eps,
            debug_prints=debug_prints,
            debug_print_limit=debug_print_limit,
            visualize_debug_predictions=visualize_debug_predictions,
            overwrite=overwrite,
            n_pred=n_pred,
            n_copy=n_copy,
            n_skip=n_skip,
            probe_shape=probe_img.shape,
            model=model,
        )


# -----------------------------------------------------------------------------
# Main public function
# -----------------------------------------------------------------------------

def ISS_CARE_predict(
    input_dir,
    model_dir,
    model_name,
    dapi_ch,
    regions_to_process=None,
    output_dir_prefix=None,
    normalize_input: bool | None = None,
    normalization_pmin: float | None = None,
    normalization_pmax: float | None = None,
    normalization_eps: float | None = None,
    debug_prints: bool = True,
    debug_print_limit: int = 3,
    visualize_debug_predictions: bool = True,
    overwrite: bool = False,
):
    """
    Apply a pretrained CARE model to ISS images stored as channel-coded TIFF files.

    Notebook-friendly usage
    -----------------------
    ISS_CARE_predict(
        input_dir=input_dir,
        model_dir=model_dir,
        model_name=model_name,
        dapi_ch=4,
        regions_to_process=None,
        output_dir_prefix=None,
        normalize_input=None,   # auto-detect from training metadata
        normalization_pmin=None,
        normalization_pmax=None,
        normalization_eps=None,
        debug_prints=True,
        debug_print_limit=3,
        visualize_debug_predictions=True,
        overwrite=False,
    )

    Args
    ----
    input_dir : str or Path
        Top-level experiment directory containing region folders (e.g. R1, R2).
    model_dir : str or Path
        Path to the folder containing pretrained CSBDeep/CARE models.
    model_name : str
        Name of the model to load.
    dapi_ch : int
        0-based channel index for DAPI. DAPI TIFFs matching `_ch{dapi_ch}.tif`
        or `_ch0*{dapi_ch}.tif` are copied unchanged.
    regions_to_process : list[int] or None
        1-based region numbers to process, e.g. [1, 2]. None means all.
    output_dir_prefix : str or Path or None
        If set, write outputs under this prefix while mirroring region names.
    normalize_input : bool or None
        If True, percentile-normalize non-DAPI images before CARE prediction.
        If False, use raw intensities.
        If None, infer from training_metadata.json when possible.
    normalization_pmin, normalization_pmax, normalization_eps : float or None
        Optional normalization settings. If None, try metadata first, then defaults.
    debug_prints : bool
        If True, print debug stats for the first few predicted images per cycle.
    debug_print_limit : int
        Number of non-DAPI predicted images per cycle to print debug stats for.
    visualize_debug_predictions : bool
        If True, save debug PNG summaries for the first few predicted images per cycle.
    overwrite : bool
        If False, keep valid existing outputs and skip them.
        If True, recompute predictions and replace existing TIFF/CSV outputs.

    Behavior
    --------
    - Discovers region directories matching R\\d+.
    - For each region, finds Cycle* folders under <region>/preprocessing/.
    - Reads TIFFs from <region>/preprocessing/CycleX/4_retiled/
    - Writes outputs to <region>/preprocessing/CycleX/4_retiled/CARE/
      or to output_dir_prefix mirroring region names.
    - Applies CARE to all TIFFs except DAPI.
    - Copies DAPI TIFFs unchanged.
    - Copies any CSV metadata alongside outputs.
    - overwrite=False: never overwrites valid existing outputs.
    - overwrite=True: replaces existing outputs.
    - Writes one provenance XML only if this run actually produced output.
    """
    preprocessing_relpath = "preprocessing"
    tiles_subpath = "4_retiled"
    axes = "YX"
    out_dtype = "uint16"

    input_dir = Path(input_dir)
    model_dir = Path(model_dir)

    print(f"[INFO] Processing directory: {input_dir.resolve()}")
    print(f"[INFO] Model base directory: {model_dir.resolve()}")
    print(f"[INFO] Model name: {model_name}")

    (
        normalize_input,
        normalization_pmin,
        normalization_pmax,
        normalization_eps,
        training_metadata_file,
        training_metadata,
    ) = resolve_inference_normalization_settings(
        model_dir=model_dir,
        model_name=model_name,
        normalize_input=normalize_input,
        normalization_pmin=normalization_pmin,
        normalization_pmax=normalization_pmax,
        normalization_eps=normalization_eps,
    )

    del training_metadata  # loaded for possible future use; not needed below

    print(f"[INFO] Training metadata file: {training_metadata_file}")
    print(f"[INFO] normalize_input (resolved): {normalize_input}")
    if normalize_input:
        print(
            f"[INFO] Normalization: pmin={normalization_pmin}, "
            f"pmax={normalization_pmax}, eps={normalization_eps}"
        )
        print("[INFO] Prediction output will be rescaled by 65535 before uint16 saving.")
    else:
        print("[INFO] Using raw input intensities for prediction.")

    print(f"[INFO] debug_prints: {debug_prints}")
    print(f"[INFO] debug_print_limit: {debug_print_limit}")
    print(f"[INFO] visualize_debug_predictions: {visualize_debug_predictions}")
    print(f"[INFO] overwrite: {overwrite}")

    if output_dir_prefix is not None:
        output_dir_prefix = Path(output_dir_prefix)
        output_dir_prefix.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Using output_dir_prefix: {output_dir_prefix.resolve()}")
    else:
        print("[INFO] Using default output location under each region directory")

    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

    available_numbers, available_map = discover_regions(input_dir)
    all_regions = [f"R{n}" for n in available_numbers]
    print(f"[INFO] Regions found on disk ({len(all_regions)}): {all_regions}")

    region_numbers, region_directories = select_region_directories(
        available_numbers=available_numbers,
        available_map=available_map,
        regions_to_process=regions_to_process,
    )

    selected_regions = [f"R{n}" for n in region_numbers]
    skipped_regions = [r for r in all_regions if r not in selected_regions]

    print(f"[INFO] Regions selected ({len(selected_regions)}): {selected_regions}")
    if skipped_regions:
        print(f"[INFO] Regions skipped ({len(skipped_regions)}): {skipped_regions}")

    model = CARE(config=None, name=model_name, basedir=str(model_dir))
    print(f"[INFO] Loaded CARE model: {model.name}")

    for region_directory in region_directories:
        region_name = region_directory.name
        print("=" * 80)
        print(f"[INFO] Processing region: {region_name}")

        preprocessing_root = region_directory / preprocessing_relpath
        if not preprocessing_root.exists():
            print(f"[WARN] {region_name}: missing preprocessing folder, skipping: {preprocessing_root}")
            continue

        cycles_found = discover_cycles(preprocessing_root)
        if not cycles_found:
            print(f"[WARN] {region_name}: no Cycle* folders found under: {preprocessing_root}")
            continue

        print(f"[INFO] {region_name}: cycles found -> {[c.name for _, c in cycles_found]}")

        for _, cycle_dir in cycles_found:
            cycle_name = cycle_dir.name

            if output_dir_prefix is None:
                out_cycle_dir = cycle_dir
            else:
                out_cycle_dir = (
                    output_dir_prefix
                    / region_name
                    / preprocessing_relpath
                    / cycle_name
                )

            process_one_cycle(
                region_name=region_name,
                cycle_name=cycle_name,
                cycle_dir=cycle_dir,
                out_cycle_dir=out_cycle_dir,
                model=model,
                model_dir=model_dir,
                model_name=model_name,
                training_metadata_file=training_metadata_file,
                dapi_ch=int(dapi_ch),
                normalize_input=normalize_input,
                normalization_pmin=normalization_pmin,
                normalization_pmax=normalization_pmax,
                normalization_eps=normalization_eps,
                debug_prints=debug_prints,
                debug_print_limit=debug_print_limit,
                visualize_debug_predictions=visualize_debug_predictions,
                overwrite=overwrite,
                run_id=run_id,
                preprocessing_relpath=preprocessing_relpath,
                tiles_subpath=tiles_subpath,
                axes=axes,
                out_dtype=out_dtype,
            )
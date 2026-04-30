from __future__ import annotations

# -----------------------------------------------------------------------------
# GPU configuration for CARE inference
# -----------------------------------------------------------------------------

import json
import os
import random
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# -----------------------------------------------------------------------------
# Pretty terminal printing
# -----------------------------------------------------------------------------

USE_COLOR_PRINTS = True


class T:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


def color_text(text: str, color: str = "", bold: bool = False) -> str:
    if not USE_COLOR_PRINTS:
        return text

    prefix = ""
    if bold:
        prefix += T.BOLD
    prefix += color
    return f"{prefix}{text}{T.RESET}"


def print_section(title: str, color: str = T.CYAN) -> None:
    line = "=" * 90
    print("\n" + color_text(line, color, bold=True))
    print(color_text(title, color, bold=True))
    print(color_text(line, color, bold=True))


def print_subsection(title: str, color: str = T.BLUE) -> None:
    line = "-" * 80
    print("\n" + color_text(line, color, bold=True))
    print(color_text(title, color, bold=True))
    print(color_text(line, color, bold=True))


def print_info(msg: str) -> None:
    print(color_text("[INFO] ", T.GREEN, bold=True) + msg)


def print_warn(msg: str) -> None:
    print(color_text("[WARN] ", T.YELLOW, bold=True) + msg)


def print_debug(msg: str) -> None:
    print(color_text("[DEBUG] ", T.MAGENTA, bold=True) + msg)


# -----------------------------------------------------------------------------
# GPU selection
# -----------------------------------------------------------------------------

def choose_gpu_for_rl(
    preferred_max_mem_mb: int = 2000,
    preferred_max_util: int = 20,
) -> int | None:
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
            print_warn("nvidia-smi returned no GPU information.")
            return None

        rows: list[tuple[int, int, int]] = []
        for line in result.splitlines():
            idx, mem, util = [x.strip() for x in line.split(",")]
            rows.append((int(idx), int(mem), int(util)))

        if not rows:
            print_warn("No GPUs parsed from nvidia-smi output.")
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

        print_info(f"Selected GPU {gpu_id} (mem={mem_mb} MiB, util={util_pct}%)")
        return gpu_id

    except Exception as e:
        print_warn(f"Automatic GPU selection failed: {e}")
        print_warn("Proceeding without forcing CUDA_VISIBLE_DEVICES.")
        return None


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
        print_info(f"TensorFlow sees {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print_warn(f"Could not set TensorFlow memory growth: {e}")
else:
    print_info("TensorFlow sees no GPU. Inference will run on CPU.")


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
    try:
        return path.exists() and path.stat().st_size > int(min_size)
    except Exception:
        return False


def load_json_file(path: Path | str) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print_warn(f"Could not read JSON file {path}: {e}")
        return None


def _print_array_stats(arr: np.ndarray, label: str) -> None:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        print_debug(f"{label}: dtype={arr.dtype}, shape={arr.shape}, all values are non-finite")
        return

    p1, p50, p99 = np.percentile(finite, [1, 50, 99])

    print_debug(
        f"{label}: "
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
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.percentile(arr, pmin)
    hi = np.percentile(arr, pmax)
    return np.clip((arr - lo) / (hi - lo + eps), 0, 1)


def rescale_prediction_for_saving(
    pred: np.ndarray,
    *,
    raw_input: np.ndarray | None = None,
    output_rescale_mode: str = "none",
    output_rescale_pmin: float = 1.0,
    output_rescale_pmax: float = 99.8,
    output_rescale_eps: float = 1e-8,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)

    if output_rescale_mode == "none":
        return pred

    if output_rescale_mode == "x65535":
        return pred * 65535.0

    if output_rescale_mode == "percentile":
        lo = np.percentile(pred, output_rescale_pmin)
        hi = np.percentile(pred, output_rescale_pmax)
        pred01 = np.clip((pred - lo) / (hi - lo + output_rescale_eps), 0, 1)
        return pred01 * 65535.0

    if output_rescale_mode == "match_raw_max":
        if raw_input is None:
            raise ValueError("raw_input is required for output_rescale_mode='match_raw_max'")
        pred_max = float(np.nanmax(pred)) if np.size(pred) else 0.0
        raw_max = float(np.nanmax(raw_input)) if np.size(raw_input) else 0.0
        if pred_max <= 0:
            return np.zeros_like(pred, dtype=np.float32)
        return pred * (raw_max / pred_max)

    raise ValueError(
        "output_rescale_mode must be one of: "
        "'none', 'x65535', 'percentile', 'match_raw_max'"
    )


def to_uint16_safe(arr: np.ndarray, *, context: str = "") -> np.ndarray:
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()

    if has_nan or has_inf:
        print_info(
            f"to_uint16_safe"
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


def _safe_percentile_pair(
    arr: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.8,
) -> tuple[float, float]:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, pmin))
    hi = float(np.percentile(finite, pmax))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _display_rescale(
    arr: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.8,
) -> np.ndarray:
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
    restored_float_raw: np.ndarray,
    restored_float_saved_scale: np.ndarray,
    restored_u16: np.ndarray,
    context: str,
    normalize_input: bool,
    output_rescale_mode: str,
) -> None:
    raw_max = float(np.nanmax(raw_input)) if np.size(raw_input) else 0.0
    model_in_max = float(np.nanmax(model_input)) if np.size(model_input) else 0.0
    pred_raw_max = float(np.nanmax(restored_float_raw)) if np.size(restored_float_raw) else 0.0
    pred_saved_max = float(np.nanmax(restored_float_saved_scale)) if np.size(restored_float_saved_scale) else 0.0
    pred_saved_mean = float(np.nanmean(restored_float_saved_scale)) if np.size(restored_float_saved_scale) else 0.0
    out_u16_max = int(np.max(restored_u16)) if np.size(restored_u16) else 0

    print_debug(f"scale_check ({context})")
    print_debug(f"  normalize_input={normalize_input}")
    print_debug(f"  output_rescale_mode={output_rescale_mode}")
    print_debug(f"  raw_input_max={raw_max:.6g}")
    print_debug(f"  model_input_max={model_in_max:.6g}")
    print_debug(f"  prediction_float_raw_max={pred_raw_max:.6g}")
    print_debug(f"  prediction_float_saved_scale_max={pred_saved_max:.6g}")
    print_debug(f"  prediction_float_saved_scale_mean={pred_saved_mean:.6g}")
    print_debug(f"  saved_uint16_max={out_u16_max}")
    print_debug(
        f"  raw_fraction_gt0={_fraction_positive(raw_input):.4f}, "
        f"model_input_fraction_gt0={_fraction_positive(model_input):.4f}, "
        f"prediction_fraction_gt0={_fraction_positive(restored_float_saved_scale):.4f}"
    )

    if normalize_input and model_in_max > 1.05:
        print_warn(
            f"{context}: normalized model input has max > 1.05 "
            f"({model_in_max:.4g}). Check normalization."
        )

    if out_u16_max < 50:
        print_warn(
            f"{context}: saved uint16 output max is very low "
            f"({out_u16_max}). Output may be severely compressed."
        )

    if out_u16_max >= 65535:
        print_warn(
            f"{context}: saved uint16 output is saturated "
            f"(max={out_u16_max}). Check output scaling."
        )


# -----------------------------------------------------------------------------
# Metadata / normalization helpers
# -----------------------------------------------------------------------------

def find_training_metadata_file(model_dir: Path | str, model_name: str) -> Path | None:
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
# Manual visualization after prediction
# -----------------------------------------------------------------------------

def visualize_random_care_predictions(
    *,
    input_dir: Path | str,
    output_dir_prefix: Path | str | None = None,
    regions_to_process: list[int] | None = None,
    n_pairs: int = 5,
    dapi_ch: int = 4,
    random_seed: int | None = 42,
    pmin: float = 1.0,
    pmax: float = 99.8,
    bins: int = 100,
) -> list[tuple[Path, Path]]:
    input_dir = Path(input_dir)
    output_dir_prefix = Path(output_dir_prefix) if output_dir_prefix is not None else None

    available_numbers, available_map = discover_regions(input_dir)
    region_numbers, region_dirs = select_region_directories(
        available_numbers=available_numbers,
        available_map=available_map,
        regions_to_process=regions_to_process,
    )

    dapi_suffix_re = re.compile(
        rf"_ch0*{int(dapi_ch)}\.(tif|tiff)$",
        re.IGNORECASE,
    )

    pairs: list[tuple[Path, Path]] = []

    for region_number, region_dir in zip(region_numbers, region_dirs):
        region_name = f"R{region_number}"
        preprocessing_root = region_dir / "preprocessing"

        if not preprocessing_root.exists():
            continue

        for _, cycle_dir in discover_cycles(preprocessing_root):
            raw_dir = cycle_dir / "4_retiled"

            if output_dir_prefix is None:
                pred_dir = cycle_dir / "4_retiled" / "CARE"
            else:
                pred_dir = (
                    output_dir_prefix
                    / region_name
                    / "preprocessing"
                    / cycle_dir.name
                    / "4_retiled"
                    / "CARE"
                )

            if not raw_dir.exists() or not pred_dir.exists():
                continue

            for raw_path in sorted(raw_dir.glob("*.tif")):
                if dapi_suffix_re.search(raw_path.name):
                    continue

                pred_path = pred_dir / raw_path.name
                if pred_path.exists():
                    pairs.append((raw_path, pred_path))

    if not pairs:
        print_warn("No raw/prediction TIFF pairs found.")
        return []

    rng = random.Random(random_seed)
    selected = rng.sample(pairs, k=min(n_pairs, len(pairs)))

    for i, (raw_path, pred_path) in enumerate(selected, start=1):
        raw = tifffile.imread(str(raw_path)).astype(np.float32)
        pred = tifffile.imread(str(pred_path)).astype(np.float32)

        print_subsection(f"Visualization {i}/{len(selected)}", color=T.MAGENTA)
        print_info(f"Raw : {raw_path}")
        print_info(f"Pred: {pred_path}")

        _print_array_stats(raw, "raw")
        _print_array_stats(pred, "prediction")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(_display_rescale(raw, pmin=pmin, pmax=pmax), cmap="gray")
        axes[0].set_title("Raw input")
        axes[0].axis("off")

        axes[1].imshow(_display_rescale(pred, pmin=pmin, pmax=pmax), cmap="gray")
        axes[1].set_title("CARE prediction")
        axes[1].axis("off")

        raw_vals = raw[np.isfinite(raw)].ravel()
        pred_vals = pred[np.isfinite(pred)].ravel()

        if raw_vals.size:
            axes[2].hist(raw_vals, bins=bins, alpha=0.6, label="raw")
        if pred_vals.size:
            axes[2].hist(pred_vals, bins=bins, alpha=0.6, label="prediction")

        axes[2].set_title("Intensity histogram")
        axes[2].legend()

        fig.suptitle(f"{raw_path.parent.parent.name} / {raw_path.name}")
        fig.tight_layout()
        plt.show()

    return selected


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
    output_rescale_mode: str,
    output_rescale_pmin: float,
    output_rescale_pmax: float,
    output_rescale_eps: float,
    debug_prints: bool,
    debug_print_limit: int,
    overwrite: bool,
    n_pred: int,
    n_copy: int,
    n_skip: int,
    probe_shape: tuple[int, int],
    model: CARE,
) -> None:
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
    SubElement(params, "output_rescale_mode").text = str(output_rescale_mode)
    SubElement(params, "output_rescale_pmin").text = str(output_rescale_pmin)
    SubElement(params, "output_rescale_pmax").text = str(output_rescale_pmax)
    SubElement(params, "output_rescale_eps").text = str(output_rescale_eps)
    SubElement(params, "debug_prints").text = str(bool(debug_prints))
    SubElement(params, "debug_print_limit").text = str(int(debug_print_limit))
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
    print_info(f"XML written: {xml_path}")


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
    output_rescale_mode: str,
    output_rescale_pmin: float,
    output_rescale_pmax: float,
    output_rescale_eps: float,
    debug_this_file: bool,
    debug_context: str,
) -> None:
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

    restored_raw = model.predict(x_in_model, axes=axes, n_tiles=n_tiles)

    restored_saved_scale = rescale_prediction_for_saving(
        restored_raw,
        raw_input=x_in,
        output_rescale_mode=output_rescale_mode,
        output_rescale_pmin=output_rescale_pmin,
        output_rescale_pmax=output_rescale_pmax,
        output_rescale_eps=output_rescale_eps,
    )

    if debug_this_file:
        print_debug(debug_context)
        _print_array_stats(x_in, "raw_input")
        if normalize_input:
            _print_array_stats(x_in_model, "normalized_input")
        else:
            _print_array_stats(x_in_model, "model_input_raw")
        _print_array_stats(restored_raw, "prediction_before_rescale")
        _print_array_stats(restored_saved_scale, "prediction_before_uint16")

    restored_u16 = to_uint16_safe(restored_saved_scale, context=debug_context)

    if debug_this_file:
        _print_array_stats(restored_u16, "saved_uint16_output")
        _print_scale_checks(
            raw_input=x_in,
            model_input=x_in_model,
            restored_float_raw=restored_raw,
            restored_float_saved_scale=restored_saved_scale,
            restored_u16=restored_u16,
            context=debug_context,
            normalize_input=normalize_input,
            output_rescale_mode=output_rescale_mode,
        )

    tifffile.imwrite(str(out_path), restored_u16)


def copy_cycle_csvs(in_tile_dir: Path, out_tile_dir: Path, *, overwrite: bool) -> int:
    n_csv_copied = 0
    for csv_path in in_tile_dir.glob("*.csv"):
        dst_csv = out_tile_dir / csv_path.name

        if not overwrite and file_exists_and_valid(dst_csv, min_size=64):
            continue

        try:
            shutil.copyfile(csv_path, dst_csv)
            n_csv_copied += 1
        except Exception as e:
            print_warn(f"Failed to copy {csv_path.name}: {e}")
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
    output_rescale_mode: str,
    output_rescale_pmin: float,
    output_rescale_pmax: float,
    output_rescale_eps: float,
    debug_prints: bool,
    debug_print_limit: int,
    overwrite: bool,
    run_id: str,
    preprocessing_relpath: str,
    tiles_subpath: str,
    axes: str,
    out_dtype: str,
) -> None:
    del preprocessing_relpath

    in_tile_dir = cycle_dir / tiles_subpath
    if not in_tile_dir.exists():
        print_warn(f"{region_name}/{cycle_name}: missing tile folder, skipping: {in_tile_dir}")
        return

    out_tile_dir = out_cycle_dir / tiles_subpath / "CARE"
    out_tile_dir.mkdir(parents=True, exist_ok=True)

    in_tifs = sorted(
        [p for p in in_tile_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: p.name,
    )

    if not in_tifs:
        print_warn(f"{region_name}/{cycle_name}: no TIFFs found, skipping.")
        return

    print_info(f"{region_name}/{cycle_name}: {len(in_tifs)} TIFF(s) found")

    dapi_suffix_re = re.compile(
        rf"_ch0*{int(dapi_ch)}\.(tif|tiff)$",
        re.IGNORECASE,
    )

    expected_out_paths = [out_tile_dir / p.name for p in in_tifs]
    all_outputs_valid = all(file_exists_and_valid(p) for p in expected_out_paths)

    if not overwrite and all_outputs_valid:
        print_info(f"{region_name}/{cycle_name}: all outputs already exist and look valid, skipping.")
        return

    probe_path = next((p for p in in_tifs if not dapi_suffix_re.search(p.name)), in_tifs[0])
    probe_img = tifffile.imread(str(probe_path))
    n_tiles = choose_n_tiles_yx(probe_img.shape)

    y, x = probe_img.shape
    ty, tx = n_tiles

    print_info(f"Tiling: {ty}x{tx} for image {y}x{x} (YxX)")
    print_info(f"Input : {in_tile_dir}")
    print_info(f"Output: {out_tile_dir}")
    print_info(f"Overwrite: {overwrite}")

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
            output_rescale_mode=output_rescale_mode,
            output_rescale_pmin=output_rescale_pmin,
            output_rescale_pmax=output_rescale_pmax,
            output_rescale_eps=output_rescale_eps,
            debug_this_file=debug_this_file,
            debug_context=f"{region_name}/{cycle_name}/{tif_path.name}",
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

    print_info(
        f"Cycle summary: predicted={n_pred}, copied_dapi={n_copy}, "
        f"skipped_existing={n_skip}, copied_csv={n_csv_copied}"
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
            output_rescale_mode=output_rescale_mode,
            output_rescale_pmin=output_rescale_pmin,
            output_rescale_pmax=output_rescale_pmax,
            output_rescale_eps=output_rescale_eps,
            debug_prints=debug_prints,
            debug_print_limit=debug_print_limit,
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
    output_rescale_mode: str = "none",
    output_rescale_pmin: float = 1.0,
    output_rescale_pmax: float = 99.8,
    output_rescale_eps: float = 1e-8,
    debug_prints: bool = True,
    debug_print_limit: int = 3,
    overwrite: bool = False,
):
    
    preprocessing_relpath = "preprocessing"
    tiles_subpath = "4_retiled"
    axes = "YX"
    out_dtype = "uint16"
    
    input_dir = Path(input_dir)
    model_dir = Path(model_dir)
    
    print_section("CARE prediction setup")
    
    print_info(f"Processing directory: {input_dir.resolve()}")
    print_info(f"Model base directory: {model_dir.resolve()}")
    print_info(f"Model name: {model_name}")
    
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
    
    del training_metadata
    
    print_info(f"Training metadata file: {training_metadata_file}")
    
    print_subsection("Input preprocessing")
    print_info(f"normalize_input resolved: {normalize_input}")
    
    if normalize_input:
        print_info(
            f"Input normalization ON: percentile scaling before prediction "
            f"(pmin={normalization_pmin}, pmax={normalization_pmax}, eps={normalization_eps})"
        )
    else:
        print_info("Input normalization OFF: raw input intensities are passed to CARE.")
    
    print_subsection("Output saving")
    print_info(f"Saved output dtype: {out_dtype}")
    print_info(f"Output rescale mode: {output_rescale_mode}")
    
    if output_rescale_mode == "none":
        print_info("Output rescale OFF: CARE prediction values are saved directly, then clipped to uint16.")
    elif output_rescale_mode == "x65535":
        print_info("Output rescale ON: prediction is multiplied by 65535 before saving.")
    elif output_rescale_mode == "percentile":
        print_info(
            f"Output rescale ON: prediction is percentile-scaled before saving "
            f"(pmin={output_rescale_pmin}, pmax={output_rescale_pmax}, eps={output_rescale_eps})"
        )
    elif output_rescale_mode == "match_raw_max":
        print_info("Output rescale ON: prediction max is matched to raw input max before saving.")
    else:
        print_warn(f"Unknown output_rescale_mode: {output_rescale_mode}")
    
    print_subsection("Run options")
    print_info(f"debug_prints: {debug_prints}")
    print_info(f"debug_print_limit: {debug_print_limit}")
    print_info(f"overwrite: {overwrite}")
    
    if output_dir_prefix is not None:
        output_dir_prefix = Path(output_dir_prefix)
        output_dir_prefix.mkdir(parents=True, exist_ok=True)
        print_info(f"Output location: {output_dir_prefix.resolve()}")
    else:
        print_info("Output location: default CARE folder inside each cycle directory")
    
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    print_info(f"Run ID: {run_id}")
    
    print_subsection("Region discovery")
    
    available_numbers, available_map = discover_regions(input_dir)
    all_regions = [f"R{n}" for n in available_numbers]
    print_info(f"Regions found on disk ({len(all_regions)}): {all_regions}")
    
    region_numbers, region_directories = select_region_directories(
        available_numbers=available_numbers,
        available_map=available_map,
        regions_to_process=regions_to_process,
    )
    
    selected_regions = [f"R{n}" for n in region_numbers]
    skipped_regions = [r for r in all_regions if r not in selected_regions]
    
    print_info(f"Regions selected ({len(selected_regions)}): {selected_regions}")
    if skipped_regions:
        print_info(f"Regions skipped ({len(skipped_regions)}): {skipped_regions}")
    
    print_subsection("Model loading")
    model = CARE(config=None, name=model_name, basedir=str(model_dir))
    print_info(f"Loaded CARE model: {model.name}")

    for region_directory in region_directories:
        region_name = region_directory.name
        print_section(f"Region: {region_name}", color=T.CYAN)

        preprocessing_root = region_directory / preprocessing_relpath
        if not preprocessing_root.exists():
            print_warn(f"{region_name}: missing preprocessing folder, skipping: {preprocessing_root}")
            continue

        cycles_found = discover_cycles(preprocessing_root)
        if not cycles_found:
            print_warn(f"{region_name}: no Cycle* folders found under: {preprocessing_root}")
            continue

        print_info(f"{region_name}: cycles found -> {[c.name for _, c in cycles_found]}")

        for _, cycle_dir in cycles_found:
            cycle_name = cycle_dir.name
            print_subsection(f"{region_name} / {cycle_name}", color=T.BLUE)

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
                output_rescale_mode=output_rescale_mode,
                output_rescale_pmin=output_rescale_pmin,
                output_rescale_pmax=output_rescale_pmax,
                output_rescale_eps=output_rescale_eps,
                debug_prints=debug_prints,
                debug_print_limit=debug_print_limit,
                overwrite=overwrite,
                run_id=run_id,
                preprocessing_relpath=preprocessing_relpath,
                tiles_subpath=tiles_subpath,
                axes=axes,
                out_dtype=out_dtype,
            )

    print_section("CARE prediction complete", color=T.GREEN)
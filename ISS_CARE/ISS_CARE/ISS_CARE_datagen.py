from __future__ import annotations

import json
import getpass
import os
import random
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from csbdeep.data import RawData, create_patches, no_background_patches
from csbdeep.utils import plot_some
from tifffile import imread

CH_RE = re.compile(r"_ch(\d+)\.tif+$", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def append_timestamp_to_filename(filename: str, timestamp: str) -> str:
    p = Path(filename)
    return f"{p.stem}__{timestamp}{p.suffix}"


def patch_size_to_jsonable(patch_size):
    return list(patch_size) if isinstance(patch_size, (tuple, list)) else patch_size


# -----------------------------------------------------------------------------
# Basic image / pair quality checks
# -----------------------------------------------------------------------------

def image_is_usable(
    arr: np.ndarray,
    *,
    min_max: float = 0.0,
    min_std: float = 1e-6,
    extreme_value_cutoff: float = 1e6,
) -> tuple[bool, str]:
    arr = np.asarray(arr)

    if arr.size == 0:
        return False, "empty array"

    if not np.all(np.isfinite(arr)):
        return False, "contains NaN/Inf"

    absmax = float(np.max(np.abs(arr)))
    if absmax > extreme_value_cutoff:
        return False, f"absmax={absmax:.6g} exceeds cutoff={extreme_value_cutoff:.6g}"

    vmax = float(np.max(arr))
    if vmax <= min_max:
        return False, f"max={vmax:.6g} <= required minimum {min_max:.6g}"

    vstd = float(np.std(arr))
    if vstd <= min_std:
        return False, f"std={vstd:.6g} <= required minimum {min_std:.6g}"

    return True, "ok"


def robust_stats(
    arr: np.ndarray,
    *,
    pmin: float = 1.0,
    pmax: float = 99.0,
) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    p_lo, p_hi = np.percentile(arr, [pmin, pmax])
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": arr_min,
        "max": arr_max,
        "range": float(arr_max - arr_min),
        "robust_range": float(p_hi - p_lo),
    }


def has_suspicious_half_plane_artifact(
    arr: np.ndarray,
    *,
    dark_fraction_threshold: float = 0.90,
    dynamic_ratio_threshold: float = 0.12,
    near_zero_std: float = 1e-8,
) -> tuple[bool, str]:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim < 2:
        return False, "not 2D+"

    h, w = arr.shape[-2], arr.shape[-1]
    if h < 8 or w < 8:
        return False, "too small"

    global_stats = robust_stats(arr)
    global_std = global_stats["std"]
    global_rng = global_stats["range"]

    if global_std <= near_zero_std or global_rng <= near_zero_std:
        return False, "nearly constant"

    left = arr[..., :, : w // 2]
    right = arr[..., :, w // 2 :]
    top = arr[..., : h // 2, :]
    bottom = arr[..., h // 2 :, :]

    halves = {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }
    stats = {name: robust_stats(half) for name, half in halves.items()}

    dark_cutoff = global_stats["min"] + 0.05 * max(global_rng, near_zero_std)

    for axis_name, a_name, b_name in [
        ("left/right", "left", "right"),
        ("top/bottom", "top", "bottom"),
    ]:
        a = halves[a_name]
        b = halves[b_name]

        frac_a_dark = float(np.mean(a <= dark_cutoff))
        frac_b_dark = float(np.mean(b <= dark_cutoff))

        if frac_a_dark >= dark_fraction_threshold and frac_b_dark < 0.5:
            return True, f"{axis_name}: {a_name} half mostly dark"
        if frac_b_dark >= dark_fraction_threshold and frac_a_dark < 0.5:
            return True, f"{axis_name}: {b_name} half mostly dark"

        a_dyn = stats[a_name]["robust_range"]
        b_dyn = stats[b_name]["robust_range"]
        smaller = min(a_dyn, b_dyn)
        larger = max(a_dyn, b_dyn)

        if larger > near_zero_std and smaller / larger < dynamic_ratio_threshold:
            return True, f"{axis_name}: one half much flatter than the other"

    return False, "ok"


def pair_has_signal_mismatch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    signal_std_threshold: float = 3e-3,
    empty_std_threshold: float = 1e-6,
    signal_max_threshold: float = 0.0,
    empty_max_threshold: float = 0.0,
) -> tuple[bool, str]:
    x = np.asarray(x)
    y = np.asarray(y)

    x_std = float(np.std(x))
    y_std = float(np.std(y))
    x_max = float(np.max(x))
    y_max = float(np.max(y))

    x_has_signal = (x_std > signal_std_threshold) or (x_max > signal_max_threshold)
    y_has_signal = (y_std > signal_std_threshold) or (y_max > signal_max_threshold)

    x_is_empty = (x_std <= empty_std_threshold) and (x_max <= empty_max_threshold)
    y_is_empty = (y_std <= empty_std_threshold) and (y_max <= empty_max_threshold)

    if x_has_signal and y_is_empty:
        return True, "source has signal but target is near-empty"
    if y_has_signal and x_is_empty:
        return True, "target has signal but source is near-empty"

    return False, "ok"


def pair_has_low_information_target(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_robust_range_floor: float = 1e-3,
    min_target_to_source_robust_range_ratio: float = 0.12,
    min_target_to_source_std_ratio: float = 0.15,
    eps: float = 1e-8,
) -> tuple[bool, str]:
    x_stats = robust_stats(x)
    y_stats = robust_stats(y)

    x_std = x_stats["std"]
    y_std = y_stats["std"]
    x_rr = x_stats["robust_range"]
    y_rr = y_stats["robust_range"]

    rr_ratio = y_rr / max(x_rr, eps)
    std_ratio = y_std / max(x_std, eps)

    if y_rr < target_robust_range_floor and x_rr > 5 * target_robust_range_floor:
        return True, "target nearly flat by robust range"

    if rr_ratio < min_target_to_source_robust_range_ratio and std_ratio < min_target_to_source_std_ratio:
        return True, "target suspiciously low-information relative to source"

    return False, "ok"


def pair_is_usable(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_source_max: float,
    min_source_std: float,
    min_target_max: float,
    min_target_std: float,
    extreme_value_cutoff: float,
    check_half_plane_artifacts: bool,
    check_signal_consistency: bool,
    check_low_information_target: bool,
    signal_std_threshold: float,
    empty_std_threshold: float,
    signal_max_threshold: float,
    empty_max_threshold: float,
    target_robust_range_floor: float,
    min_target_to_source_robust_range_ratio: float,
    min_target_to_source_std_ratio: float,
) -> tuple[bool, dict]:
    info = {
        "source_reason": None,
        "target_reason": None,
        "pair_reason": None,
    }

    x_ok, x_reason = image_is_usable(
        x,
        min_max=min_source_max,
        min_std=min_source_std,
        extreme_value_cutoff=extreme_value_cutoff,
    )
    y_ok, y_reason = image_is_usable(
        y,
        min_max=min_target_max,
        min_std=min_target_std,
        extreme_value_cutoff=extreme_value_cutoff,
    )

    if not x_ok:
        info["source_reason"] = x_reason
    if not y_ok:
        info["target_reason"] = y_reason

    if not (x_ok and y_ok):
        return False, info

    if x.shape != y.shape:
        info["pair_reason"] = f"shape mismatch: {x.shape} vs {y.shape}"
        return False, info

    if check_half_plane_artifacts:
        bad, reason = has_suspicious_half_plane_artifact(x)
        if bad:
            info["pair_reason"] = f"source artifact: {reason}"
            return False, info

        bad, reason = has_suspicious_half_plane_artifact(y)
        if bad:
            info["pair_reason"] = f"target artifact: {reason}"
            return False, info

    if check_signal_consistency:
        bad, reason = pair_has_signal_mismatch(
            x,
            y,
            signal_std_threshold=signal_std_threshold,
            empty_std_threshold=empty_std_threshold,
            signal_max_threshold=signal_max_threshold,
            empty_max_threshold=empty_max_threshold,
        )
        if bad:
            info["pair_reason"] = reason
            return False, info

    if check_low_information_target:
        bad, reason = pair_has_low_information_target(
            x,
            y,
            target_robust_range_floor=target_robust_range_floor,
            min_target_to_source_robust_range_ratio=min_target_to_source_robust_range_ratio,
            min_target_to_source_std_ratio=min_target_to_source_std_ratio,
        )
        if bad:
            info["pair_reason"] = reason
            return False, info

    return True, info


# -----------------------------------------------------------------------------
# File discovery / pairing
# -----------------------------------------------------------------------------

def find_all_samples(
    root: Path,
    subdirs: Sequence[str],
    source_dirname: str,
    target_dirname: str,
) -> list[Path]:
    search_roots = [root / d for d in subdirs] if subdirs else [root]
    sample_dirs: list[Path] = []

    def _onerror(err):
        print(f"WARNING: Could not access {err.filename}: {err}")

    for base in search_roots:
        if not base.exists():
            print(f"WARNING: {base} does not exist, skipping.")
            continue

        for dirpath, dirnames, _ in os.walk(base, topdown=True, onerror=_onerror):
            dirpath = Path(dirpath)
            if source_dirname in dirnames and target_dirname in dirnames:
                sample_dirs.append(dirpath)
                dirnames[:] = []

    return sorted(set(sample_dirs))


def get_channel_from_name(p: Path) -> Optional[int]:
    m = CH_RE.search(p.name)
    return int(m.group(1)) if m else None


def collect_pairs(
    sample_dir: Path,
    source_dirname: str,
    target_dirname: str,
    pattern: str,
) -> list[tuple[Path, Path]]:
    src_root = sample_dir / source_dirname
    tgt_root = sample_dir / target_dirname

    src_files = sorted(src_root.glob(pattern))
    tgt_files = sorted(tgt_root.glob(pattern))

    if not src_files:
        raise RuntimeError(f"No SOURCE files found: {src_root} pattern={pattern}")
    if not tgt_files:
        raise RuntimeError(f"No TARGET files found: {tgt_root} pattern={pattern}")

    tgt_rel_set = {f.relative_to(tgt_root) for f in tgt_files}

    pairs: list[tuple[Path, Path]] = []
    for src in src_files:
        rel = src.relative_to(src_root)
        if rel in tgt_rel_set:
            pairs.append((src, tgt_root / rel))

    if not pairs:
        raise RuntimeError("No source/target pairs matched by relative path.")

    return pairs


def subsample_pairs(
    file_pairs: list[tuple[Path, Path]],
    max_images: Optional[int] = None,
    fraction: Optional[float] = None,
    seed: int = 42,
) -> list[tuple[Path, Path]]:
    n = len(file_pairs)

    if max_images is not None:
        k = min(max_images, n)
    elif fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in (0, 1].")
        k = max(1, int(round(n * fraction)))
    else:
        return file_pairs

    rng = random.Random(seed)
    return rng.sample(file_pairs, k)


def split_pairs_excluding_dapi(
    file_pairs: list[tuple[Path, Path]],
    dapi_channel_index: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], int]:
    non_dapi: list[tuple[Path, Path]] = []
    dapi: list[tuple[Path, Path]] = []
    excluded = 0

    for src, tgt in file_pairs:
        ch = get_channel_from_name(src)
        if ch is None:
            excluded += 1
            continue
        if ch == dapi_channel_index:
            dapi.append((src, tgt))
        else:
            non_dapi.append((src, tgt))

    return non_dapi, dapi, excluded


def filter_pairs_with_usable_images(
    file_pairs: list[tuple[Path, Path]],
    *,
    filter_settings: dict,
) -> tuple[list[tuple[Path, Path]], list[dict]]:
    kept_pairs: list[tuple[Path, Path]] = []
    removed_info: list[dict] = []

    for src, tgt in file_pairs:
        x = imread(src)
        y = imread(tgt)

        ok, info = pair_is_usable(x, y, **filter_settings)

        if ok:
            kept_pairs.append((src, tgt))
        else:
            removed_info.append(
                {
                    "source": str(src),
                    "target": str(tgt),
                    "source_reason": info["source_reason"],
                    "target_reason": info["target_reason"],
                    "pair_reason": info["pair_reason"],
                }
            )

    return kept_pairs, removed_info


def rawdata_from_pairs(source_target_pairs: list[tuple[Path, Path]], axes: str) -> RawData:
    def gen():
        for source_path, target_path in source_target_pairs:
            yield imread(source_path), imread(target_path), axes, None

    return RawData(generator=gen, size=len(source_target_pairs), description="paired_generator")


# -----------------------------------------------------------------------------
# Signal-biased pair weighting
# -----------------------------------------------------------------------------

def foreground_fraction(
    arr: np.ndarray,
    *,
    pmin: float = 1.0,
    pmax: float = 99.0,
    signal_fraction: float = 0.10,
    eps: float = 1e-8,
) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [pmin, pmax])
    thr = lo + signal_fraction * max(hi - lo, eps)
    return float(np.mean(arr > thr))


def compute_pair_signal_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    use_target: bool = True,
    pmin: float = 1.0,
    pmax: float = 99.0,
    signal_foreground_fraction: float = 0.10,
) -> dict:
    arr = np.asarray(y if use_target else x, dtype=np.float32)
    st = robust_stats(arr, pmin=pmin, pmax=pmax)

    return {
        "std": st["std"],
        "max": st["max"],
        "robust_range": st["robust_range"],
        "foreground_fraction": foreground_fraction(
            arr,
            pmin=pmin,
            pmax=pmax,
            signal_fraction=signal_foreground_fraction,
        ),
    }


def compute_pair_signal_score(
    metrics: dict,
    *,
    weight_std: float = 1.5,
    weight_robust_range: float = 1.5,
    weight_foreground: float = 3.0,
    weight_max: float = 0.25,
) -> float:
    return float(
        weight_std * np.log1p(metrics["std"])
        + weight_robust_range * np.log1p(metrics["robust_range"])
        + weight_foreground * metrics["foreground_fraction"]
        + weight_max * np.log1p(metrics["max"])
    )


def apply_signal_bias_to_pairs(
    pairs: list[tuple[Path, Path]],
    *,
    pair_selection_mode: str = "signal_biased",
    signal_score_use_target: bool = True,
    max_pair_repeats: int = 3,
    drop_low_score_fraction: float = 0.20,
    min_pairs_to_keep_after_signal_bias: int = 8,
    signal_score_pmin: float = 1.0,
    signal_score_pmax: float = 99.0,
    signal_foreground_fraction: float = 0.10,
    signal_weight_std: float = 1.5,
    signal_weight_robust_range: float = 1.5,
    signal_weight_foreground: float = 3.0,
    signal_weight_max: float = 0.25,
) -> tuple[list[tuple[Path, Path]], dict]:
    if not pairs:
        return [], {
            "mode": pair_selection_mode,
            "n_input_pairs": 0,
            "n_kept_pairs": 0,
            "n_expanded_pairs": 0,
            "score_min": None,
            "score_median": None,
            "score_max": None,
        }

    if pair_selection_mode not in {"uniform", "signal_biased"}:
        raise ValueError("pair_selection_mode must be 'uniform' or 'signal_biased'")

    if pair_selection_mode == "uniform":
        return pairs, {
            "mode": pair_selection_mode,
            "n_input_pairs": len(pairs),
            "n_kept_pairs": len(pairs),
            "n_expanded_pairs": len(pairs),
            "score_min": None,
            "score_median": None,
            "score_max": None,
        }

    annotated = []
    for src, tgt in pairs:
        x = imread(src)
        y = imread(tgt)
        metrics = compute_pair_signal_metrics(
            x,
            y,
            use_target=signal_score_use_target,
            pmin=signal_score_pmin,
            pmax=signal_score_pmax,
            signal_foreground_fraction=signal_foreground_fraction,
        )
        score = compute_pair_signal_score(
            metrics,
            weight_std=signal_weight_std,
            weight_robust_range=signal_weight_robust_range,
            weight_foreground=signal_weight_foreground,
            weight_max=signal_weight_max,
        )
        annotated.append({"source": src, "target": tgt, "score": float(score)})

    annotated = sorted(annotated, key=lambda d: d["score"])
    n_input = len(annotated)

    n_drop = int(np.floor(n_input * drop_low_score_fraction)) if drop_low_score_fraction > 0 else 0
    max_drop_allowed = max(0, n_input - min_pairs_to_keep_after_signal_bias)
    n_drop = min(n_drop, max_drop_allowed)

    kept = annotated[n_drop:] if n_drop > 0 else annotated

    scores = np.array([d["score"] for d in kept], dtype=np.float32)
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    s_med = float(np.median(scores))

    expanded_pairs: list[tuple[Path, Path]] = []
    for item in kept:
        if max_pair_repeats == 1 or s_max <= s_min:
            repeats = 1
        else:
            norm = (item["score"] - s_min) / (s_max - s_min + 1e-8)
            repeats = 1 + int(round(norm * (max_pair_repeats - 1)))
        expanded_pairs.extend([(item["source"], item["target"])] * repeats)

    summary = {
        "mode": pair_selection_mode,
        "n_input_pairs": n_input,
        "n_dropped_pairs": int(n_drop),
        "n_kept_pairs": len(kept),
        "n_expanded_pairs": len(expanded_pairs),
        "score_min": s_min,
        "score_median": s_med,
        "score_max": s_max,
    }

    return expanded_pairs, summary


# -----------------------------------------------------------------------------
# Patch creation / validation
# -----------------------------------------------------------------------------

def validate_patch_array(
    arr: np.ndarray,
    name: str,
    *,
    extreme_value_cutoff: float = 1e6,
) -> None:
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError(f"{name} is empty after patch creation.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf after patch creation.")

    absmax = float(np.max(np.abs(arr)))
    if absmax > extreme_value_cutoff:
        raise ValueError(f"{name} contains extreme values (absmax={absmax:.6g}).")


def patch_filter_from_threshold(patch_filter_threshold: float):
    if patch_filter_threshold and patch_filter_threshold > 0:
        return no_background_patches(patch_filter_threshold)
    return None


def save_patch_metadata(
    *,
    patch_file: Path,
    variant_name: str,
    axes: str,
    X_shape,
    Y_shape,
    n_pairs_used: int,
    n_patches_total: int,
    patch_size,
    n_patches_per_image,
    patch_filter_threshold: float,
    extra_summary: dict,
) -> Path:
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "patch_file": str(patch_file),
        "variant_name": variant_name,
        "axes": axes,
        "X_shape": list(X_shape),
        "Y_shape": list(Y_shape),
        "n_pairs_used": int(n_pairs_used),
        "n_patches_total": int(n_patches_total),
        "patch_parameters": {
            "PATCH_SIZE": patch_size_to_jsonable(patch_size),
            "N_PATCHES_PER_IMAGE": n_patches_per_image,
            "PATCH_FILTER_THRESHOLD": patch_filter_threshold,
        },
        "summary": extra_summary,
    }

    metadata_file = patch_file.with_name(f"{patch_file.stem}__metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)

    print("Saved patch metadata:", metadata_file)
    return metadata_file


def make_patches_for_variant(
    *,
    raw_data: RawData,
    out_file: Path,
    patch_size,
    n_patches_per_image,
    patch_axes,
    patch_filter_threshold: float,
    extreme_value_cutoff: float = 1e6,
    variant_name: str = "UNKNOWN",
    extra_summary: Optional[dict] = None,
):
    out_file.parent.mkdir(parents=True, exist_ok=True)

    patch_filter = patch_filter_from_threshold(patch_filter_threshold)

    X, Y, XY_axes = create_patches(
        raw_data=raw_data,
        patch_size=patch_size,
        n_patches_per_image=n_patches_per_image,
        patch_axes=patch_axes,
        patch_filter=patch_filter,
        save_file=str(out_file),
        verbose=True,
    )

    validate_patch_array(X, "X patches", extreme_value_cutoff=extreme_value_cutoff)
    validate_patch_array(Y, "Y patches", extreme_value_cutoff=extreme_value_cutoff)

    save_patch_metadata(
        patch_file=out_file,
        variant_name=variant_name,
        axes=XY_axes,
        X_shape=X.shape,
        Y_shape=Y.shape,
        n_pairs_used=len(raw_data),
        n_patches_total=len(X),
        patch_size=patch_size,
        n_patches_per_image=n_patches_per_image,
        patch_filter_threshold=patch_filter_threshold,
        extra_summary=extra_summary or {},
    )

    return X, Y, XY_axes


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def normalize_pairs_jointly(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.8,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    x_batch = np.asarray(x_batch, dtype=np.float32)
    y_batch = np.asarray(y_batch, dtype=np.float32)

    x_norm = np.empty_like(x_batch, dtype=np.float32)
    y_norm = np.empty_like(y_batch, dtype=np.float32)

    for i in range(len(x_batch)):
        pair_vals = np.concatenate([x_batch[i].ravel(), y_batch[i].ravel()])
        lo = np.percentile(pair_vals, pmin)
        hi = np.percentile(pair_vals, pmax)
        x_norm[i] = np.clip((x_batch[i] - lo) / (hi - lo + eps), 0, 1)
        y_norm[i] = np.clip((y_batch[i] - lo) / (hi - lo + eps), 0, 1)

    return x_norm, y_norm


def visualize_patch_pairs(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_show: int = 5,
    random_seed: Optional[int] = None,
    normalize: bool = True,
    title: Optional[str] = None,
    figsize=(12, 5),
):
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Nothing to visualize.")

    n_show = min(n_show, len(X))
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(X), size=n_show, replace=False)

    X_show = X[idx]
    Y_show = Y[idx]

    if normalize:
        X_show, Y_show = normalize_pairs_jointly(X_show, Y_show)

    plt.figure(figsize=figsize)
    plot_some(X_show, Y_show)
    plt.suptitle(title or f"{n_show} random patch pairs")
    plt.show()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def run_patch_generation(
    *,
    care_root: Path,
    care_subdirs: Sequence[str],
    source_dirname: str,
    target_dirname: str,
    pattern: str,
    dapi_channel_index: int,
    axes: str,
    patch_size,
    n_patches_per_image,
    patch_axes,
    patch_filter_threshold: float,
    patch_dirname: str,
    merge_all_samples: bool,
    merged_non_dapi_name: str,
    merged_dapi_name: str,
    max_images_per_sample: Optional[int] = None,
    fraction_images_per_sample: Optional[float] = None,
    sampling_seed: int = 42,
    min_source_max: float = 0.0,
    min_source_std: float = 1e-6,
    min_target_max: float = 0.0,
    min_target_std: float = 1e-6,
    extreme_value_cutoff: float = 1e6,
    check_half_plane_artifacts: bool = True,
    check_signal_consistency: bool = True,
    check_low_information_target: bool = True,
    signal_std_threshold: float = 3e-3,
    empty_std_threshold: float = 1e-6,
    signal_max_threshold: float = 0.0,
    empty_max_threshold: float = 0.0,
    target_robust_range_floor: float = 1e-3,
    min_target_to_source_robust_range_ratio: float = 0.12,
    min_target_to_source_std_ratio: float = 0.15,
    pair_selection_mode: str = "signal_biased",
    signal_score_use_target: bool = True,
    max_pair_repeats: int = 3,
    drop_low_score_fraction: float = 0.20,
    min_pairs_to_keep_after_signal_bias: int = 8,
    signal_score_pmin: float = 1.0,
    signal_score_pmax: float = 99.0,
    signal_foreground_fraction: float = 0.10,
    signal_weight_std: float = 1.5,
    signal_weight_robust_range: float = 1.5,
    signal_weight_foreground: float = 3.0,
    signal_weight_max: float = 0.25,
):
    filter_settings = {
        "min_source_max": min_source_max,
        "min_source_std": min_source_std,
        "min_target_max": min_target_max,
        "min_target_std": min_target_std,
        "extreme_value_cutoff": extreme_value_cutoff,
        "check_half_plane_artifacts": check_half_plane_artifacts,
        "check_signal_consistency": check_signal_consistency,
        "check_low_information_target": check_low_information_target,
        "signal_std_threshold": signal_std_threshold,
        "empty_std_threshold": empty_std_threshold,
        "signal_max_threshold": signal_max_threshold,
        "empty_max_threshold": empty_max_threshold,
        "target_robust_range_floor": target_robust_range_floor,
        "min_target_to_source_robust_range_ratio": min_target_to_source_robust_range_ratio,
        "min_target_to_source_std_ratio": min_target_to_source_std_ratio,
    }

    sample_dirs = find_all_samples(
        care_root,
        care_subdirs,
        source_dirname,
        target_dirname,
    )

    print("=" * 90)
    print("Starting CARE patch generation")
    print("=" * 90)
    print("Samples detected:", len(sample_dirs))

    all_patch_files_non_dapi: list[Path] = []
    all_patch_files_dapi: list[Path] = []
    merged_X_non_dapi: list[np.ndarray] = []
    merged_Y_non_dapi: list[np.ndarray] = []
    merged_X_dapi: list[np.ndarray] = []
    merged_Y_dapi: list[np.ndarray] = []
    axes_ref_non_dapi = None
    axes_ref_dapi = None
    sample_summaries: list[dict] = []

    for sample_idx, sample_dir in enumerate(sample_dirs, start=1):
        sample_name = sample_dir.name
        sample_group_name = "__".join(sample_dir.parent.relative_to(care_root).parts)

        print("\n" + "=" * 90)
        print(f"[Sample {sample_idx}/{len(sample_dirs)}] {sample_name}")

        file_pairs = collect_pairs(sample_dir, source_dirname, target_dirname, pattern)
        non_dapi_pairs_all, dapi_pairs_all, excluded_count = split_pairs_excluding_dapi(
            file_pairs, dapi_channel_index=dapi_channel_index
        )

        non_dapi_pairs_sampled = subsample_pairs(
            non_dapi_pairs_all,
            max_images=max_images_per_sample,
            fraction=fraction_images_per_sample,
            seed=sampling_seed,
        )
        dapi_pairs_sampled = subsample_pairs(
            dapi_pairs_all,
            max_images=max_images_per_sample,
            fraction=fraction_images_per_sample,
            seed=sampling_seed,
        )

        non_dapi_pairs, removed_non_dapi_info = filter_pairs_with_usable_images(
            non_dapi_pairs_sampled,
            filter_settings=filter_settings,
        )
        dapi_pairs, removed_dapi_info = filter_pairs_with_usable_images(
            dapi_pairs_sampled,
            filter_settings=filter_settings,
        )

        non_dapi_pairs_for_patching, non_dapi_signal_summary = apply_signal_bias_to_pairs(
            non_dapi_pairs,
            pair_selection_mode=pair_selection_mode,
            signal_score_use_target=signal_score_use_target,
            max_pair_repeats=max_pair_repeats,
            drop_low_score_fraction=drop_low_score_fraction,
            min_pairs_to_keep_after_signal_bias=min_pairs_to_keep_after_signal_bias,
            signal_score_pmin=signal_score_pmin,
            signal_score_pmax=signal_score_pmax,
            signal_foreground_fraction=signal_foreground_fraction,
            signal_weight_std=signal_weight_std,
            signal_weight_robust_range=signal_weight_robust_range,
            signal_weight_foreground=signal_weight_foreground,
            signal_weight_max=signal_weight_max,
        )
        dapi_pairs_for_patching, dapi_signal_summary = apply_signal_bias_to_pairs(
            dapi_pairs,
            pair_selection_mode=pair_selection_mode,
            signal_score_use_target=signal_score_use_target,
            max_pair_repeats=max_pair_repeats,
            drop_low_score_fraction=drop_low_score_fraction,
            min_pairs_to_keep_after_signal_bias=min_pairs_to_keep_after_signal_bias,
            signal_score_pmin=signal_score_pmin,
            signal_score_pmax=signal_score_pmax,
            signal_foreground_fraction=signal_foreground_fraction,
            signal_weight_std=signal_weight_std,
            signal_weight_robust_range=signal_weight_robust_range,
            signal_weight_foreground=signal_weight_foreground,
            signal_weight_max=signal_weight_max,
        )

        print("NON_DAPI usable pairs:", len(non_dapi_pairs))
        print("NON_DAPI pairs for patching:", len(non_dapi_pairs_for_patching))
        print("DAPI usable pairs:", len(dapi_pairs))
        print("DAPI pairs for patching:", len(dapi_pairs_for_patching))
        if excluded_count:
            print("Excluded without _chN tag:", excluded_count)

        out_dir = sample_dir / patch_dirname
        out_file_non_dapi = out_dir / f"{sample_group_name}__{sample_name}__NON_DAPI__train_patches.npz"
        out_file_dapi = out_dir / f"{sample_group_name}__{sample_name}__DAPI_ONLY__train_patches.npz"

        sample_summary = {
            "sample_name": sample_name,
            "sample_dir": str(sample_dir),
            "n_non_dapi_pairs": len(non_dapi_pairs),
            "n_dapi_pairs": len(dapi_pairs),
            "n_non_dapi_pairs_for_patching": len(non_dapi_pairs_for_patching),
            "n_dapi_pairs_for_patching": len(dapi_pairs_for_patching),
            "n_excluded_pairs": excluded_count,
            "removed_non_dapi_examples": removed_non_dapi_info[:10],
            "removed_dapi_examples": removed_dapi_info[:10],
            "non_dapi_signal_bias_summary": non_dapi_signal_summary,
            "dapi_signal_bias_summary": dapi_signal_summary,
            "n_non_dapi_patches": 0,
            "n_dapi_patches": 0,
            "non_dapi_file": None,
            "dapi_file": None,
        }

        if non_dapi_pairs_for_patching:
            raw_data_non_dapi = rawdata_from_pairs(non_dapi_pairs_for_patching, axes=axes)
            Xn, Yn, axes_n = make_patches_for_variant(
                raw_data=raw_data_non_dapi,
                out_file=out_file_non_dapi,
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_axes=patch_axes,
                patch_filter_threshold=patch_filter_threshold,
                extreme_value_cutoff=extreme_value_cutoff,
                variant_name="NON_DAPI",
                extra_summary=sample_summary,
            )
            all_patch_files_non_dapi.append(out_file_non_dapi)
            sample_summary["n_non_dapi_patches"] = int(len(Xn))
            sample_summary["non_dapi_file"] = str(out_file_non_dapi)

            if merge_all_samples:
                merged_X_non_dapi.append(Xn)
                merged_Y_non_dapi.append(Yn)
                if axes_ref_non_dapi is None:
                    axes_ref_non_dapi = axes_n

        if dapi_pairs_for_patching:
            raw_data_dapi = rawdata_from_pairs(dapi_pairs_for_patching, axes=axes)
            Xd, Yd, axes_d = make_patches_for_variant(
                raw_data=raw_data_dapi,
                out_file=out_file_dapi,
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_axes=patch_axes,
                patch_filter_threshold=patch_filter_threshold,
                extreme_value_cutoff=extreme_value_cutoff,
                variant_name="DAPI_ONLY",
                extra_summary=sample_summary,
            )
            all_patch_files_dapi.append(out_file_dapi)
            sample_summary["n_dapi_patches"] = int(len(Xd))
            sample_summary["dapi_file"] = str(out_file_dapi)

            if merge_all_samples:
                merged_X_dapi.append(Xd)
                merged_Y_dapi.append(Yd)
                if axes_ref_dapi is None:
                    axes_ref_dapi = axes_d

        sample_summaries.append(sample_summary)

    merged_non_dapi_file = None
    merged_dapi_file = None
    merge_timestamp = timestamp_for_filename()

    if merge_all_samples:
        merged_out_dir = care_root / patch_dirname
        merged_out_dir.mkdir(parents=True, exist_ok=True)

        if merged_X_non_dapi:
            merged_non_dapi_file = merged_out_dir / append_timestamp_to_filename(
                merged_non_dapi_name, merge_timestamp
            )
            X = np.concatenate(merged_X_non_dapi, axis=0)
            Y = np.concatenate(merged_Y_non_dapi, axis=0)
            validate_patch_array(X, "Merged NON_DAPI X")
            validate_patch_array(Y, "Merged NON_DAPI Y")
            np.savez_compressed(merged_non_dapi_file, X=X, Y=Y, axes=axes_ref_non_dapi)

        if merged_X_dapi:
            merged_dapi_file = merged_out_dir / append_timestamp_to_filename(
                merged_dapi_name, merge_timestamp
            )
            X = np.concatenate(merged_X_dapi, axis=0)
            Y = np.concatenate(merged_Y_dapi, axis=0)
            validate_patch_array(X, "Merged DAPI X")
            validate_patch_array(Y, "Merged DAPI Y")
            np.savez_compressed(merged_dapi_file, X=X, Y=Y, axes=axes_ref_dapi)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "sample_summaries": sample_summaries,
        "all_patch_files_non_dapi": [str(p) for p in all_patch_files_non_dapi],
        "all_patch_files_dapi": [str(p) for p in all_patch_files_dapi],
        "merged_non_dapi_file": str(merged_non_dapi_file) if merged_non_dapi_file else None,
        "merged_dapi_file": str(merged_dapi_file) if merged_dapi_file else None,
    }

    metadata_dir = care_root / patch_dirname
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / "patch_generation_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)

    print("\nPatch generation complete.")
    print("Metadata file:", metadata_file)

    return {
        "sample_dirs": sample_dirs,
        "all_patch_files_non_dapi": all_patch_files_non_dapi,
        "all_patch_files_dapi": all_patch_files_dapi,
        "merged_non_dapi_file": merged_non_dapi_file,
        "merged_dapi_file": merged_dapi_file,
        "sample_summaries": sample_summaries,
        "metadata": metadata,
        "metadata_file": metadata_file,
    }

def normalize_pairs_jointly(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.8,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Percentile-normalize each input-target pair jointly for visualization.
    """
    x_batch = np.asarray(x_batch, dtype=np.float32)
    y_batch = np.asarray(y_batch, dtype=np.float32)

    x_norm = np.empty_like(x_batch, dtype=np.float32)
    y_norm = np.empty_like(y_batch, dtype=np.float32)

    for i in range(len(x_batch)):
        pair_vals = np.concatenate([x_batch[i].ravel(), y_batch[i].ravel()])
        lo = np.percentile(pair_vals, pmin)
        hi = np.percentile(pair_vals, pmax)

        x_norm[i] = np.clip((x_batch[i] - lo) / (hi - lo + eps), 0, 1)
        y_norm[i] = np.clip((y_batch[i] - lo) / (hi - lo + eps), 0, 1)

    return x_norm, y_norm


def visualize_patch_pairs(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_show: int = 5,
    random_seed: Optional[int] = None,
    normalize: bool = True,
    title: Optional[str] = None,
    figsize=(12, 5),
):
    """
    Visualize random patch pairs from one dataset.
    """
    if len(X) == 0:
        raise ValueError("X is empty; nothing to visualize.")
    if len(Y) == 0:
        raise ValueError("Y is empty; nothing to visualize.")

    n_show = min(n_show, len(X))

    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(X), size=n_show, replace=False)

    X_show = X[idx]
    Y_show = Y[idx]

    if normalize:
        X_show, Y_show = normalize_pairs_jointly(X_show, Y_show)

    plt.figure(figsize=figsize)
    plot_some(X_show, Y_show)

    if title is None:
        title = f"{n_show} random patch pairs"

    if normalize:
        title = f"{title}\n(top: input, bottom: target; jointly normalized per pair)"
    else:
        title = f"{title}\n(top: input, bottom: target)"

    plt.suptitle(title)
    plt.show()


def visualize_saved_patches_across_samples(
    patch_files: Sequence[Path | str],
    *,
    n_show_per_sample: int = 3,
    variant_name: str = "NON_DAPI",
    random_seed: Optional[int] = None,
    normalize: bool = True,
    figsize=(12, 5),
):
    """
    Show example patch pairs for every saved sample patch file.
    """
    patch_files = [Path(p) for p in patch_files]

    if not patch_files:
        print(f"No patch files found for {variant_name}.")
        return

    base_rng = np.random.default_rng(random_seed)

    for patch_file in patch_files:
        data = np.load(patch_file)
        X = data["X"]
        Y = data["Y"]

        sample_title = patch_file.stem
        seed_i = int(base_rng.integers(0, 2**32 - 1)) if random_seed is not None else None

        visualize_patch_pairs(
            X,
            Y,
            n_show=n_show_per_sample,
            random_seed=seed_i,
            normalize=normalize,
            title=f"{variant_name}: {sample_title}",
            figsize=figsize,
        )
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


def print_error(msg: str) -> None:
    print(color_text("[ERROR] ", T.RED, bold=True) + msg)


def print_success(msg: str) -> None:
    print(color_text("[DONE] ", T.GREEN, bold=True) + msg)


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

    if arr.dtype == np.object_:
        return False, "object dtype"

    if not np.all(np.isfinite(arr)):
        return False, "contains NaN/Inf"

    arr = np.asarray(arr, dtype=np.float32)

    if not np.all(np.isfinite(arr)):
        return False, "contains NaN/Inf after float32 conversion"

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

    halves = {"left": left, "right": right, "top": top, "bottom": bottom}
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
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

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
    info = {"source_reason": None, "target_reason": None, "pair_reason": None}

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
        print_warn(f"Could not access {err.filename}: {err}")

    for base in search_roots:
        if not base.exists():
            print_warn(f"{base} does not exist, skipping.")
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


# -----------------------------------------------------------------------------
# Read + validate pairs first
# -----------------------------------------------------------------------------

def read_image_as_float32(
    path: Path,
    *,
    extreme_value_cutoff: float = 1e6,
) -> np.ndarray:
    arr = imread(path)
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError("empty array")
    if arr.dtype == np.object_:
        raise ValueError("object dtype")
    if not np.all(np.isfinite(arr)):
        raise ValueError("contains NaN/Inf before float32 conversion")

    arr = np.asarray(arr, dtype=np.float32)

    if not np.all(np.isfinite(arr)):
        raise ValueError("contains NaN/Inf after float32 conversion")

    absmax = float(np.max(np.abs(arr)))
    if absmax > extreme_value_cutoff:
        raise ValueError(
            f"extreme image values after float32 conversion: "
            f"absmax={absmax:.6g} > cutoff={extreme_value_cutoff:.6g}"
        )

    return arr


def filter_pairs_with_usable_images(
    file_pairs: list[tuple[Path, Path]],
    *,
    filter_settings: dict,
) -> tuple[list[tuple[Path, Path, np.ndarray, np.ndarray]], list[dict]]:
    kept_pairs: list[tuple[Path, Path, np.ndarray, np.ndarray]] = []
    removed_info: list[dict] = []

    extreme_value_cutoff = filter_settings["extreme_value_cutoff"]

    for src, tgt in file_pairs:
        try:
            x = read_image_as_float32(src, extreme_value_cutoff=extreme_value_cutoff)
            y = read_image_as_float32(tgt, extreme_value_cutoff=extreme_value_cutoff)
        except Exception as e:
            removed_info.append(
                {
                    "source": str(src),
                    "target": str(tgt),
                    "source_reason": None,
                    "target_reason": None,
                    "pair_reason": f"read/image-level validation error: {e}",
                }
            )
            continue

        ok, info = pair_is_usable(x, y, **filter_settings)

        if ok:
            kept_pairs.append((src, tgt, x, y))
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

    metadata_file = patch_file.with_suffix(".json")

    with open(metadata_file, "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)

    print_info(f"Saved patch metadata: {metadata_file}")
    return metadata_file


def make_patches_for_variant(
    *,
    pairs: list[tuple[Path, Path, np.ndarray, np.ndarray]],
    out_file: Path,
    axes: str,
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

    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    dropped_pairs: list[dict] = []
    XY_axes_ref = None

    print_subsection(f"{variant_name}: patch creation", color=T.MAGENTA)
    print_info(f"Input image pairs: {len(pairs)}")
    print_info(f"Patch size: {patch_size}")
    print_info(f"Patches per image: {n_patches_per_image}")
    print_info(f"Patch filter threshold: {patch_filter_threshold}")

    for pair_idx, (src, tgt, x, y) in enumerate(pairs, start=1):
        def gen():
            yield x, y, axes, None

        raw_data_single = RawData(
            generator=gen,
            size=1,
            description="single_pair_generator",
        )

        try:
            X_i, Y_i, XY_axes = create_patches(
                raw_data=raw_data_single,
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_axes=patch_axes,
                patch_filter=patch_filter,
                save_file=None,
                verbose=False,
            )

            validate_patch_array(
                X_i,
                "single-pair X patches",
                extreme_value_cutoff=extreme_value_cutoff,
            )
            validate_patch_array(
                Y_i,
                "single-pair Y patches",
                extreme_value_cutoff=extreme_value_cutoff,
            )

        except Exception as e:
            dropped_pairs.append(
                {
                    "source": str(src),
                    "target": str(tgt),
                    "reason": str(e),
                }
            )
            print_warn(f"Dropping bad {variant_name} image pair {pair_idx}/{len(pairs)}")
            print_warn(f"  source: {src}")
            print_warn(f"  target: {tgt}")
            print_warn(f"  reason: {e}")
            continue

        X_parts.append(X_i)
        Y_parts.append(Y_i)

        if XY_axes_ref is None:
            XY_axes_ref = XY_axes

    if not X_parts:
        raise RuntimeError(f"{variant_name}: no valid image pairs left after patch creation.")

    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)

    validate_patch_array(X, "X patches", extreme_value_cutoff=extreme_value_cutoff)
    validate_patch_array(Y, "Y patches", extreme_value_cutoff=extreme_value_cutoff)

    np.savez_compressed(out_file, X=X, Y=Y, axes=XY_axes_ref)

    summary = dict(extra_summary or {})
    summary["n_pairs_before_patch_creation"] = len(pairs)
    summary["n_pairs_used_after_patch_creation"] = len(X_parts)
    summary["n_pairs_dropped_during_patch_creation"] = len(dropped_pairs)
    summary["dropped_pairs_during_patch_creation"] = dropped_pairs[:50]

    save_patch_metadata(
        patch_file=out_file,
        variant_name=variant_name,
        axes=XY_axes_ref,
        X_shape=X.shape,
        Y_shape=Y.shape,
        n_pairs_used=len(X_parts),
        n_patches_total=len(X),
        patch_size=patch_size,
        n_patches_per_image=n_patches_per_image,
        patch_filter_threshold=patch_filter_threshold,
        extra_summary=summary,
    )

    print_success(f"{variant_name} patch file saved: {out_file}")
    print_info(f"{variant_name} patches created: {len(X)}")
    print_info(f"{variant_name} image pairs used: {len(X_parts)}")
    print_info(f"{variant_name} image pairs dropped during patch creation: {len(dropped_pairs)}")

    return X, Y, XY_axes_ref, dropped_pairs


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
    patch_files = [Path(p) for p in patch_files]

    if not patch_files:
        print_warn(f"No patch files found for {variant_name}.")
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
    
    search_roots = [care_root / d for d in care_subdirs] if care_subdirs else [care_root]
    
    print_section("CARE patch generation setup")
    print_info(f"CARE root: {care_root}")
    print_info(f"Subdirs: {list(care_subdirs) if care_subdirs else 'ALL'}")
    print_info(f"Source dir name: {source_dirname}")
    print_info(f"Target dir name: {target_dirname}")
    print_info(f"Pattern: {pattern}")
    print_info(f"DAPI channel index: {dapi_channel_index}")
    print_info(f"Samples detected: {len(sample_dirs)}")
    
    print_subsection("Search roots")
    for i, root_dir in enumerate(search_roots, start=1):
        status = "exists" if root_dir.exists() else "missing"
        print_info(f"{i:03d}: {root_dir} ({status})")
    
    print_subsection("Detected sample directories")
    if not sample_dirs:
        print_warn(
            "No valid samples found. A valid sample folder must contain both "
            f"'{source_dirname}' and '{target_dirname}'."
        )
    else:
        for i, sample_dir in enumerate(sample_dirs, start=1):
            try:
                rel_sample = sample_dir.relative_to(care_root)
            except ValueError:
                rel_sample = sample_dir
    
            print_info(f"{i:03d}: {rel_sample}")
    
            src_root = sample_dir / source_dirname
            tgt_root = sample_dir / target_dirname
    
            if not src_root.exists() or not tgt_root.exists():
                print_warn("     source or target folder missing")
                continue
    
            src_files = list(src_root.glob(pattern))
            tgt_files = list(tgt_root.glob(pattern))
    
            src_rel_dirs = {f.relative_to(src_root).parent for f in src_files}
            tgt_rel_dirs = {f.relative_to(tgt_root).parent for f in tgt_files}
    
            valid_subdirs = sorted(src_rel_dirs & tgt_rel_dirs)
    
            if not valid_subdirs:
                print_warn("     no matching source/target subdirectories found")
            else:
                print_info(f"     matching data subdirectories: {len(valid_subdirs)}")
                for subdir in valid_subdirs[:10]:
                    print_info(f"       - {subdir}")
    
                if len(valid_subdirs) > 10:
                    print_info(f"       ... plus {len(valid_subdirs) - 10} more")
        
    print_subsection("Patch settings")
    print_info(f"Axes: {axes}")
    print_info(f"Patch axes: {patch_axes}")
    print_info(f"Patch size: {patch_size}")
    print_info(f"Patches per image: {n_patches_per_image}")
    print_info(f"Patch filter threshold: {patch_filter_threshold}")
    
    print_subsection("Sampling settings")
    print_info(f"Max images per sample: {max_images_per_sample}")
    print_info(f"Fraction images per sample: {fraction_images_per_sample}")
    print_info(f"Sampling seed: {sampling_seed}")
    
    print_subsection("Output settings")
    print_info(f"Patch directory name: {patch_dirname}")
    print_info(f"Merge all samples: {merge_all_samples}")
    print_info(f"Merged NON_DAPI name: {merged_non_dapi_name}")
    print_info(f"Merged DAPI name: {merged_dapi_name}")
    
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
        rel_parent = sample_dir.parent.relative_to(care_root)
        sample_group_name = "__".join(rel_parent.parts) if rel_parent.parts else "root"

        print_section(f"Sample {sample_idx}/{len(sample_dirs)}: {sample_name}", color=T.CYAN)
        print_info(f"Sample directory: {sample_dir}")

        try:
            file_pairs = collect_pairs(sample_dir, source_dirname, target_dirname, pattern)
        except Exception as e:
            print_error(f"Could not collect pairs for sample {sample_name}: {e}")
            continue

        non_dapi_pairs_all, dapi_pairs_all, excluded_count = split_pairs_excluding_dapi(
            file_pairs,
            dapi_channel_index=dapi_channel_index,
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

        print_subsection("Pair discovery")
        print_info(f"Total matched pairs: {len(file_pairs)}")
        print_info(f"NON_DAPI pairs found: {len(non_dapi_pairs_all)}")
        print_info(f"DAPI pairs found: {len(dapi_pairs_all)}")
        print_info(f"NON_DAPI pairs sampled: {len(non_dapi_pairs_sampled)}")
        print_info(f"DAPI pairs sampled: {len(dapi_pairs_sampled)}")

        if excluded_count:
            print_warn(f"Excluded files without _chN tag: {excluded_count}")

        non_dapi_pairs, removed_non_dapi_info = filter_pairs_with_usable_images(
            non_dapi_pairs_sampled,
            filter_settings=filter_settings,
        )
        dapi_pairs, removed_dapi_info = filter_pairs_with_usable_images(
            dapi_pairs_sampled,
            filter_settings=filter_settings,
        )

        print_subsection("Image-level filtering")
        print_info(f"NON_DAPI usable pairs before patch creation: {len(non_dapi_pairs)}")
        print_info(f"DAPI usable pairs before patch creation: {len(dapi_pairs)}")
        print_info(f"NON_DAPI removed by image/pair filters: {len(removed_non_dapi_info)}")
        print_info(f"DAPI removed by image/pair filters: {len(removed_dapi_info)}")

        out_dir = sample_dir / patch_dirname
        out_file_non_dapi = out_dir / f"{sample_group_name}__{sample_name}__NON_DAPI__train_patches.npz"
        out_file_dapi = out_dir / f"{sample_group_name}__{sample_name}__DAPI_ONLY__train_patches.npz"

        sample_summary = {
            "sample_name": sample_name,
            "sample_dir": str(sample_dir),
            "n_total_pairs_found": len(file_pairs),
            "n_non_dapi_pairs_before_filtering": len(non_dapi_pairs_sampled),
            "n_dapi_pairs_before_filtering": len(dapi_pairs_sampled),
            "n_non_dapi_pairs_used": 0,
            "n_dapi_pairs_used": 0,
            "n_non_dapi_pairs_dropped_during_patch_creation": 0,
            "n_dapi_pairs_dropped_during_patch_creation": 0,
            "n_excluded_pairs_no_channel_tag": excluded_count,
            "removed_non_dapi_examples": removed_non_dapi_info[:10],
            "removed_dapi_examples": removed_dapi_info[:10],
            "dropped_non_dapi_pairs_during_patch_creation": [],
            "dropped_dapi_pairs_during_patch_creation": [],
            "n_non_dapi_patches": 0,
            "n_dapi_patches": 0,
            "non_dapi_file": None,
            "dapi_file": None,
            "non_dapi_error": None,
            "dapi_error": None,
        }

        if non_dapi_pairs:
            try:
                Xn, Yn, axes_n, dropped_non_dapi_pairs = make_patches_for_variant(
                    pairs=non_dapi_pairs,
                    out_file=out_file_non_dapi,
                    axes=axes,
                    patch_size=patch_size,
                    n_patches_per_image=n_patches_per_image,
                    patch_axes=patch_axes,
                    patch_filter_threshold=patch_filter_threshold,
                    extreme_value_cutoff=extreme_value_cutoff,
                    variant_name="NON_DAPI",
                    extra_summary=sample_summary,
                )
            except Exception as e:
                sample_summary["non_dapi_error"] = str(e)
                print_warn(f"Skipping NON_DAPI patches for sample: {sample_name}")
                print_warn(f"Reason: {e}")
            else:
                all_patch_files_non_dapi.append(out_file_non_dapi)
                sample_summary["n_non_dapi_patches"] = int(len(Xn))
                sample_summary["non_dapi_file"] = str(out_file_non_dapi)
                sample_summary["n_non_dapi_pairs_used"] = int(len(non_dapi_pairs) - len(dropped_non_dapi_pairs))
                sample_summary["n_non_dapi_pairs_dropped_during_patch_creation"] = int(len(dropped_non_dapi_pairs))
                sample_summary["dropped_non_dapi_pairs_during_patch_creation"] = dropped_non_dapi_pairs[:20]

                if merge_all_samples:
                    merged_X_non_dapi.append(Xn)
                    merged_Y_non_dapi.append(Yn)
                    if axes_ref_non_dapi is None:
                        axes_ref_non_dapi = axes_n
        else:
            print_warn("No NON_DAPI pairs left after filtering.")

        if dapi_pairs:
            try:
                Xd, Yd, axes_d, dropped_dapi_pairs = make_patches_for_variant(
                    pairs=dapi_pairs,
                    out_file=out_file_dapi,
                    axes=axes,
                    patch_size=patch_size,
                    n_patches_per_image=n_patches_per_image,
                    patch_axes=patch_axes,
                    patch_filter_threshold=patch_filter_threshold,
                    extreme_value_cutoff=extreme_value_cutoff,
                    variant_name="DAPI_ONLY",
                    extra_summary=sample_summary,
                )
            except Exception as e:
                sample_summary["dapi_error"] = str(e)
                print_warn(f"Skipping DAPI patches for sample: {sample_name}")
                print_warn(f"Reason: {e}")
            else:
                all_patch_files_dapi.append(out_file_dapi)
                sample_summary["n_dapi_patches"] = int(len(Xd))
                sample_summary["dapi_file"] = str(out_file_dapi)
                sample_summary["n_dapi_pairs_used"] = int(len(dapi_pairs) - len(dropped_dapi_pairs))
                sample_summary["n_dapi_pairs_dropped_during_patch_creation"] = int(len(dropped_dapi_pairs))
                sample_summary["dropped_dapi_pairs_during_patch_creation"] = dropped_dapi_pairs[:20]

                if merge_all_samples:
                    merged_X_dapi.append(Xd)
                    merged_Y_dapi.append(Yd)
                    if axes_ref_dapi is None:
                        axes_ref_dapi = axes_d
        else:
            print_warn("No DAPI pairs left after filtering.")

        print_subsection("Sample summary", color=T.GREEN)
        print_info(f"NON_DAPI patches: {sample_summary['n_non_dapi_patches']}")
        print_info(f"DAPI patches: {sample_summary['n_dapi_patches']}")
        print_info(f"NON_DAPI pairs used: {sample_summary['n_non_dapi_pairs_used']}")
        print_info(f"DAPI pairs used: {sample_summary['n_dapi_pairs_used']}")
        print_info(
            f"NON_DAPI pairs dropped during patch creation: "
            f"{sample_summary['n_non_dapi_pairs_dropped_during_patch_creation']}"
        )
        print_info(
            f"DAPI pairs dropped during patch creation: "
            f"{sample_summary['n_dapi_pairs_dropped_during_patch_creation']}"
        )

        sample_summaries.append(sample_summary)

    merged_non_dapi_file = None
    merged_dapi_file = None
    merge_timestamp = timestamp_for_filename()

    if merge_all_samples:
        print_section("Merging samples", color=T.MAGENTA)

        merged_out_dir = care_root / patch_dirname
        merged_out_dir.mkdir(parents=True, exist_ok=True)

        if merged_X_non_dapi:
            merged_non_dapi_file = merged_out_dir / append_timestamp_to_filename(
                merged_non_dapi_name,
                merge_timestamp,
            )
            X = np.concatenate(merged_X_non_dapi, axis=0)
            Y = np.concatenate(merged_Y_non_dapi, axis=0)
            validate_patch_array(X, "Merged NON_DAPI X")
            validate_patch_array(Y, "Merged NON_DAPI Y")
            np.savez_compressed(merged_non_dapi_file, X=X, Y=Y, axes=axes_ref_non_dapi)

            print_success(f"Merged NON_DAPI file saved: {merged_non_dapi_file}")
            print_info(f"Merged NON_DAPI patches: {len(X)}")

            save_patch_metadata(
                patch_file=merged_non_dapi_file,
                variant_name="MERGED_NON_DAPI",
                axes=axes_ref_non_dapi,
                X_shape=X.shape,
                Y_shape=Y.shape,
                n_pairs_used=sum(s["n_non_dapi_pairs_used"] for s in sample_summaries),
                n_patches_total=len(X),
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_filter_threshold=patch_filter_threshold,
                extra_summary={
                    "merged_from_files": [str(p) for p in all_patch_files_non_dapi],
                    "sample_summaries": sample_summaries,
                },
            )
        else:
            print_warn("No NON_DAPI patches available for merging.")

        if merged_X_dapi:
            merged_dapi_file = merged_out_dir / append_timestamp_to_filename(
                merged_dapi_name,
                merge_timestamp,
            )
            X = np.concatenate(merged_X_dapi, axis=0)
            Y = np.concatenate(merged_Y_dapi, axis=0)
            validate_patch_array(X, "Merged DAPI X")
            validate_patch_array(Y, "Merged DAPI Y")
            np.savez_compressed(merged_dapi_file, X=X, Y=Y, axes=axes_ref_dapi)

            print_success(f"Merged DAPI file saved: {merged_dapi_file}")
            print_info(f"Merged DAPI patches: {len(X)}")

            save_patch_metadata(
                patch_file=merged_dapi_file,
                variant_name="MERGED_DAPI_ONLY",
                axes=axes_ref_dapi,
                X_shape=X.shape,
                Y_shape=Y.shape,
                n_pairs_used=sum(s["n_dapi_pairs_used"] for s in sample_summaries),
                n_patches_total=len(X),
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_filter_threshold=patch_filter_threshold,
                extra_summary={
                    "merged_from_files": [str(p) for p in all_patch_files_dapi],
                    "sample_summaries": sample_summaries,
                },
            )
        else:
            print_warn("No DAPI patches available for merging.")

    total_non_dapi_patches = int(sum(s["n_non_dapi_patches"] for s in sample_summaries))
    total_dapi_patches = int(sum(s["n_dapi_patches"] for s in sample_summaries))
    total_patches = total_non_dapi_patches + total_dapi_patches

    total_non_dapi_pairs_used = int(sum(s["n_non_dapi_pairs_used"] for s in sample_summaries))
    total_dapi_pairs_used = int(sum(s["n_dapi_pairs_used"] for s in sample_summaries))
    total_non_dapi_pairs_dropped = int(
        sum(s["n_non_dapi_pairs_dropped_during_patch_creation"] for s in sample_summaries)
    )
    total_dapi_pairs_dropped = int(
        sum(s["n_dapi_pairs_dropped_during_patch_creation"] for s in sample_summaries)
    )

    run_id = merge_timestamp  # reuse the same timestamp used for merged files
    
    metadata = {
        "run_id": run_id,  # <-- consistent identifier across all outputs
        "timestamp": datetime.now().isoformat(),  # exact time (human-readable)
    
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
    
        "total_non_dapi_patches": total_non_dapi_patches,
        "total_dapi_patches": total_dapi_patches,
        "total_patches": total_patches,
    
        "total_non_dapi_pairs_used": total_non_dapi_pairs_used,
        "total_dapi_pairs_used": total_dapi_pairs_used,
    
        "total_non_dapi_pairs_dropped_during_patch_creation": total_non_dapi_pairs_dropped,
        "total_dapi_pairs_dropped_during_patch_creation": total_dapi_pairs_dropped,
    
        "sample_summaries": sample_summaries,
    
        "all_patch_files_non_dapi": [str(p) for p in all_patch_files_non_dapi],
        "all_patch_files_dapi": [str(p) for p in all_patch_files_dapi],
    
        "merged_non_dapi_file": str(merged_non_dapi_file) if merged_non_dapi_file else None,
        "merged_dapi_file": str(merged_dapi_file) if merged_dapi_file else None,
    
        "settings": {
            "care_root": str(care_root),
            "care_subdirs": list(care_subdirs),
            "source_dirname": source_dirname,
            "target_dirname": target_dirname,
            "pattern": pattern,
            "dapi_channel_index": dapi_channel_index,
            "axes": axes,
            "patch_size": patch_size_to_jsonable(patch_size),
            "n_patches_per_image": n_patches_per_image,
            "patch_axes": patch_axes,
            "patch_filter_threshold": patch_filter_threshold,
            "patch_dirname": patch_dirname,
            "merge_all_samples": merge_all_samples,
            "merged_non_dapi_name": merged_non_dapi_name,
            "merged_dapi_name": merged_dapi_name,
            "max_images_per_sample": max_images_per_sample,
            "fraction_images_per_sample": fraction_images_per_sample,
            "sampling_seed": sampling_seed,
            "filter_settings": filter_settings,
        },
    }

    metadata_dir = care_root / patch_dirname
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / f"patch_generation_metadata__{run_id}.json"

    with open(metadata_file, "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)

    print_section("Patch generation complete", color=T.GREEN)
    print_info(f"Total NON_DAPI patches: {total_non_dapi_patches}")
    print_info(f"Total DAPI patches: {total_dapi_patches}")
    print_info(f"Total patches: {total_patches}")
    print_info(f"Total NON_DAPI image pairs used: {total_non_dapi_pairs_used}")
    print_info(f"Total DAPI image pairs used: {total_dapi_pairs_used}")
    print_info(f"Total NON_DAPI image pairs dropped during patch creation: {total_non_dapi_pairs_dropped}")
    print_info(f"Total DAPI image pairs dropped during patch creation: {total_dapi_pairs_dropped}")
    print_info(f"Metadata file: {metadata_file}")
    print_info(f"Merged NON_DAPI file: {merged_non_dapi_file}")
    print_info(f"Merged DAPI file: {merged_dapi_file}")

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
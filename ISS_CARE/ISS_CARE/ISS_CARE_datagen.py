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
SITE_RE = re.compile(r"_s(\d+)_", re.IGNORECASE)


def image_is_usable(
    arr: np.ndarray,
    *,
    min_max: float = 0.0,
    min_std: float = 1e-6,
    extreme_value_cutoff: float = 1e6,
) -> tuple[bool, str]:
    """
    Decide whether a single image is usable for training patch generation.
    """
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


def filter_pairs_with_usable_images(
    file_pairs: list[tuple[Path, Path]],
    *,
    min_source_max: float = 0.0,
    min_source_std: float = 1e-6,
    min_target_max: float = 0.0,
    min_target_std: float = 1e-6,
    extreme_value_cutoff: float = 1e6,
) -> tuple[list[tuple[Path, Path]], list[dict]]:
    """
    Remove source/target pairs where either image is unusable.
    """
    kept_pairs: list[tuple[Path, Path]] = []
    removed_info: list[dict] = []

    for src, tgt in file_pairs:
        x = imread(src)
        y = imread(tgt)

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

        if x_ok and y_ok:
            kept_pairs.append((src, tgt))
        else:
            removed_info.append(
                {
                    "source": str(src),
                    "target": str(tgt),
                    "source_reason": None if x_ok else x_reason,
                    "target_reason": None if y_ok else y_reason,
                }
            )

    return kept_pairs, removed_info


def find_all_samples(
    root: Path,
    subdirs: Sequence[str],
    source_dirname: str,
    target_dirname: str,
) -> list[Path]:
    """
    Find all sample directories under root (or under the specified subdirs).
    """
    if subdirs:
        search_roots = [root / d for d in subdirs]
    else:
        search_roots = [root]

    sample_dirs: list[Path] = []

    def _onerror(err):
        print(f"WARNING: Could not access {err.filename}: {err}")

    for base in search_roots:
        if not base.exists():
            print(f"WARNING: {base} does not exist, skipping.")
            continue

        for dirpath, dirnames, _filenames in os.walk(base, topdown=True, onerror=_onerror):
            dirpath = Path(dirpath)

            if source_dirname in dirnames and target_dirname in dirnames:
                sample_dirs.append(dirpath)
                dirnames[:] = []
                continue

    return sorted(set(sample_dirs))


def get_channel_from_name(p: Path) -> Optional[int]:
    """Parse '_chN.tif' from filename. Returns int or None if not found."""
    m = CH_RE.search(p.name)
    return int(m.group(1)) if m else None


def get_site_from_name(p: Path) -> Optional[int]:
    """Parse '_sN_' from filename. Returns int or None if not found."""
    m = SITE_RE.search(p.name)
    return int(m.group(1)) if m else None


def collect_pairs(
    sample_dir: Path,
    source_dirname: str,
    target_dirname: str,
    pattern: str,
) -> list[tuple[Path, Path]]:
    """
    Collect (source_file, target_file) pairs matched by relative path under
    source_dirname/ and target_dirname/.
    """
    src_root = sample_dir / source_dirname
    tgt_root = sample_dir / target_dirname

    src_files = sorted(src_root.glob(pattern))
    tgt_files = sorted(tgt_root.glob(pattern))

    if not src_files:
        raise RuntimeError(f"No SOURCE files found: {src_root} pattern={pattern}")
    if not tgt_files:
        raise RuntimeError(f"No TARGET files found: {tgt_root} pattern={pattern}")

    tgt_rel_set = {f.relative_to(tgt_root) for f in tgt_files}

    file_pairs: list[tuple[Path, Path]] = []
    for source_file in src_files:
        rel = source_file.relative_to(src_root)
        if rel in tgt_rel_set:
            file_pairs.append((source_file, tgt_root / rel))

    if not file_pairs:
        example = src_files[0].relative_to(src_root)
        raise RuntimeError(
            f"Found {source_dirname} and {target_dirname} files, but none match by relative path.\n"
            f"Example {source_dirname} relative path: {example}\n"
            f"Check that {source_dirname}/ and {target_dirname}/ have identical subfolder structure under them."
        )

    return file_pairs


def subsample_pairs(
    file_pairs: list[tuple[Path, Path]],
    max_images: Optional[int] = None,
    fraction: Optional[float] = None,
    seed: int = 42,
) -> list[tuple[Path, Path]]:
    """
    Subsample a list of (source, target) pairs.
    """
    n = len(file_pairs)

    if max_images is not None:
        k = min(max_images, n)
    elif fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in the interval (0, 1].")
        k = max(1, int(round(n * fraction)))
    else:
        return file_pairs

    rng = random.Random(seed)
    return rng.sample(file_pairs, k)


def split_pairs_excluding_dapi(
    file_pairs: list[tuple[Path, Path]],
    dapi_channel_index: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], int]:
    """
    Split pairs into NON_DAPI and DAPI_ONLY.

    Files without a '_chN' pattern are excluded entirely.
    """
    non_dapi: list[tuple[Path, Path]] = []
    dapi: list[tuple[Path, Path]] = []
    excluded = 0

    for source_file, target_file in file_pairs:
        ch = get_channel_from_name(source_file)

        if ch is None:
            excluded += 1
            continue

        if ch == dapi_channel_index:
            dapi.append((source_file, target_file))
        else:
            non_dapi.append((source_file, target_file))

    return non_dapi, dapi, excluded


def rawdata_from_pairs(source_target_pairs: list[tuple[Path, Path]], axes: str) -> RawData:
    """
    Build RawData from a generator so images are not all loaded into memory at once.
    """

    def gen():
        for source_path, target_path in source_target_pairs:
            yield imread(source_path), imread(target_path), axes, None

    return RawData(
        generator=gen,
        size=len(source_target_pairs),
        description="paired_generator",
    )


def validate_patch_array(
    arr: np.ndarray,
    name: str,
    *,
    extreme_value_cutoff: float = 1e6,
) -> None:
    """
    Validate patch arrays after create_patches().
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError(f"{name} is empty after patch creation.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf after patch creation.")

    absmax = float(np.max(np.abs(arr)))
    if absmax > extreme_value_cutoff:
        raise ValueError(
            f"{name} contains extreme values after patch creation "
            f"(absmax={absmax:.6g}, cutoff={extreme_value_cutoff:.6g})."
        )


def patch_filter_from_threshold(patch_filter_threshold: float):
    """Build the csbdeep patch filter or return None."""
    if patch_filter_threshold and patch_filter_threshold > 0:
        return no_background_patches(patch_filter_threshold)
    return None


def make_patches_for_variant(
    *,
    raw_data: RawData,
    out_file: Path,
    patch_size,
    n_patches_per_image,
    patch_axes,
    patch_filter_threshold: float,
    extreme_value_cutoff: float = 1e6,
):
    """
    Create and save training patches for one model variant.
    """
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

    print(f"Post-create_patches summary for: {out_file.name}")
    print(
        f"  X -> dtype: {X.dtype}, shape: {X.shape}, "
        f"min: {np.min(X):.6g}, max: {np.max(X):.6g}, absmax: {np.max(np.abs(X)):.6g}"
    )
    print(
        f"  Y -> dtype: {Y.dtype}, shape: {Y.shape}, "
        f"min: {np.min(Y):.6g}, max: {np.max(Y):.6g}, absmax: {np.max(np.abs(Y)):.6g}"
    )

    try:
        validate_patch_array(X, "X patches", extreme_value_cutoff=extreme_value_cutoff)
        validate_patch_array(Y, "Y patches", extreme_value_cutoff=extreme_value_cutoff)
    except Exception:
        if out_file.exists():
            try:
                out_file.unlink()
                print("Removed invalid patch file:", out_file)
            except OSError:
                print("WARNING: Could not remove invalid patch file:", out_file)
        raise

    return X, Y, XY_axes


def chunk_is_usable_for_patching(
    pairs: list[tuple[Path, Path]],
    *,
    axes: str,
    patch_size,
    n_patches_per_image,
    patch_axes,
    patch_filter_threshold: float,
    extreme_value_cutoff: float = 1e6,
) -> tuple[bool, Optional[str]]:
    """
    Test whether a chunk of pairs can generate valid patches.
    """
    if not pairs:
        return True, None

    patch_filter = patch_filter_from_threshold(patch_filter_threshold)

    try:
        raw = rawdata_from_pairs(pairs, axes=axes)

        X, Y, _ = create_patches(
            raw_data=raw,
            patch_size=patch_size,
            n_patches_per_image=n_patches_per_image,
            patch_axes=patch_axes,
            patch_filter=patch_filter,
            save_file=None,
            verbose=False,
        )

        validate_patch_array(X, "X patches", extreme_value_cutoff=extreme_value_cutoff)
        validate_patch_array(Y, "Y patches", extreme_value_cutoff=extreme_value_cutoff)

        return True, None

    except Exception as e:
        return False, str(e)


def group_pairs_by_site(
    pairs: list[tuple[Path, Path]],
) -> tuple[dict[int, list[tuple[Path, Path]]], list[tuple[Path, Path]]]:
    """
    Group pairs by site index parsed from the source filename.
    """
    grouped: dict[int, list[tuple[Path, Path]]] = {}
    ungrouped: list[tuple[Path, Path]] = []

    for src, tgt in pairs:
        site = get_site_from_name(src)
        if site is None:
            ungrouped.append((src, tgt))
        else:
            grouped.setdefault(site, []).append((src, tgt))

    return grouped, ungrouped


def filter_pairs_with_patch_validation_by_site(
    pairs: list[tuple[Path, Path]],
    *,
    axes: str,
    patch_size,
    n_patches_per_image,
    patch_axes,
    patch_filter_threshold: float,
    extreme_value_cutoff: float = 1e6,
) -> tuple[list[tuple[Path, Path]], list[dict]]:
    """
    Always run patch-level validation, but do it in batches of sites.

    Strategy
    --------
    1. Group pairs by site.
    2. Test batches of sites together.
    3. If a batch passes, keep all those sites.
    4. If a batch fails, split the batch of sites.
    5. When one failing site is isolated, remove that whole site.
    """
    good_pairs: list[tuple[Path, Path]] = []
    removed_sites: list[dict] = []

    grouped, ungrouped = group_pairs_by_site(pairs)
    site_items = sorted(grouped.items())

    def _recurse_site_batches(site_batch: list[tuple[int, list[tuple[Path, Path]]]]) -> None:
        if not site_batch:
            return

        batch_pairs: list[tuple[Path, Path]] = []
        for _site, site_pairs in site_batch:
            batch_pairs.extend(site_pairs)

        ok, reason = chunk_is_usable_for_patching(
            batch_pairs,
            axes=axes,
            patch_size=patch_size,
            n_patches_per_image=n_patches_per_image,
            patch_axes=patch_axes,
            patch_filter_threshold=patch_filter_threshold,
            extreme_value_cutoff=extreme_value_cutoff,
        )

        if ok:
            good_pairs.extend(batch_pairs)
            return

        if len(site_batch) == 1:
            site, site_pairs = site_batch[0]
            removed_sites.append(
                {
                    "site": site,
                    "n_pairs": len(site_pairs),
                    "example_source": str(site_pairs[0][0]),
                    "example_target": str(site_pairs[0][1]),
                    "reason": reason,
                }
            )
            return

        mid = len(site_batch) // 2
        _recurse_site_batches(site_batch[:mid])
        _recurse_site_batches(site_batch[mid:])

    _recurse_site_batches(site_items)

    if ungrouped:
        ok, reason = chunk_is_usable_for_patching(
            ungrouped,
            axes=axes,
            patch_size=patch_size,
            n_patches_per_image=n_patches_per_image,
            patch_axes=patch_axes,
            patch_filter_threshold=patch_filter_threshold,
            extreme_value_cutoff=extreme_value_cutoff,
        )

        if ok:
            good_pairs.extend(ungrouped)
        else:
            removed_sites.append(
                {
                    "site": None,
                    "n_pairs": len(ungrouped),
                    "example_source": str(ungrouped[0][0]),
                    "example_target": str(ungrouped[0][1]),
                    "reason": reason,
                }
            )

    return good_pairs, removed_sites


def normalize_pairs_jointly(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    pmin: float = 1,
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


def sample_group_name_from_sample_dir(sample_dir: Path, care_root: Path) -> str:
    """
    Build a flattened group name from the sample parent path under care_root.
    """
    return "__".join(sample_dir.parent.relative_to(care_root).parts)


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
):
    """
    Run the full patch generation workflow.

    Also builds and saves metadata automatically at the end.

    Returns a dictionary containing generated files, arrays for merging,
    per-sample summary information, and the saved metadata path.
    """
    all_patch_files_non_dapi: list[Path] = []
    all_patch_files_dapi: list[Path] = []

    merged_X_non_dapi: list[np.ndarray] = []
    merged_Y_non_dapi: list[np.ndarray] = []
    merged_X_dapi: list[np.ndarray] = []
    merged_Y_dapi: list[np.ndarray] = []

    axes_ref_non_dapi = None
    axes_ref_dapi = None

    sample_summaries = []

    sample_dirs = find_all_samples(
        care_root,
        care_subdirs,
        source_dirname,
        target_dirname,
    )
    n_samples = len(sample_dirs)

    print("\n" + "=" * 90)
    print("Starting CARE patch generation")
    print("=" * 90)
    print("Dataset root:", care_root)
    print(
        "Search subdirectories:",
        care_subdirs if care_subdirs else "[all subdirectories under dataset root]",
    )
    print("Source folder name:", source_dirname)
    print("Target folder name:", target_dirname)
    print("File pattern:", pattern)
    print("DAPI channel index (0-based):", dapi_channel_index)
    print("Max images per sample:", max_images_per_sample)
    print("Fraction of images per sample:", fraction_images_per_sample)
    print("Patch filter threshold:", patch_filter_threshold)
    print("Minimum source max intensity:", min_source_max)
    print("Minimum source std:", min_source_std)
    print("Minimum target max intensity:", min_target_max)
    print("Minimum target std:", min_target_std)
    print("Extreme value cutoff:", extreme_value_cutoff)
    print("Sample directories detected:", n_samples)

    for sample_idx, sample_dir in enumerate(sample_dirs, start=1):
        sample_name = sample_dir.name
        sample_group_name = sample_group_name_from_sample_dir(sample_dir, care_root)

        print("\n" + "=" * 90)
        print(f"[Sample {sample_idx}/{n_samples}] Processing: {sample_name}")
        print(f"Samples remaining after this one: {n_samples - sample_idx}")
        print("Sample directory:", sample_dir)
        print("Sample group:", sample_group_name)

        print("Stage 1/4: Matching source and target files")
        file_pairs = collect_pairs(sample_dir, source_dirname, target_dirname, pattern)
        n_pairs_before_sampling = len(file_pairs)

        non_dapi_pairs_all, dapi_pairs_all, excluded_count = split_pairs_excluding_dapi(
            file_pairs, dapi_channel_index=dapi_channel_index
        )

        print("Stage 2/4: Sampling and image-level filtering")
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
            min_source_max=min_source_max,
            min_source_std=min_source_std,
            min_target_max=min_target_max,
            min_target_std=min_target_std,
            extreme_value_cutoff=extreme_value_cutoff,
        )
        dapi_pairs, removed_dapi_info = filter_pairs_with_usable_images(
            dapi_pairs_sampled,
            min_source_max=min_source_max,
            min_source_std=min_source_std,
            min_target_max=min_target_max,
            min_target_std=min_target_std,
            extreme_value_cutoff=extreme_value_cutoff,
        )

        removed_non_dapi = len(removed_non_dapi_info)
        removed_dapi = len(removed_dapi_info)

        print("Pair summary:")
        print("  Total matched source/target pairs:", n_pairs_before_sampling)
        print("  Available before sampling:")
        print("    NON_DAPI:", len(non_dapi_pairs_all))
        print("    DAPI_ONLY:", len(dapi_pairs_all))
        print("  After sampling:")
        print("    NON_DAPI:", len(non_dapi_pairs_sampled))
        print("    DAPI_ONLY:", len(dapi_pairs_sampled))
        print("  After image-quality filtering:")
        print("    NON_DAPI:", len(non_dapi_pairs))
        print("    DAPI_ONLY:", len(dapi_pairs))
        print("Removed during image-quality filtering:")
        print("  NON_DAPI removed:", removed_non_dapi)
        print("  DAPI_ONLY removed:", removed_dapi)

        if removed_non_dapi_info[:5]:
            print("Examples of removed NON_DAPI pairs:")
            for item in removed_non_dapi_info[:5]:
                print(
                    "  -",
                    Path(item["source"]).name,
                    "| source:", item["source_reason"],
                    "| target:", item["target_reason"],
                )

        if removed_dapi_info[:5]:
            print("Examples of removed DAPI_ONLY pairs:")
            for item in removed_dapi_info[:5]:
                print(
                    "  -",
                    Path(item["source"]).name,
                    "| source:", item["source_reason"],
                    "| target:", item["target_reason"],
                )

        if excluded_count:
            print(
                f"Excluded {excluded_count} file(s) because the filename did not contain a '_chN' channel tag."
            )

        print("Stage 3/4: Patch-level validation by batched site checks")
        patch_bad_non_dapi_info: list[dict] = []
        patch_bad_dapi_info: list[dict] = []

        if non_dapi_pairs:
            grouped_non_dapi, ungrouped_non_dapi = group_pairs_by_site(non_dapi_pairs)
            print(
                f"Checking NON_DAPI pairs by batched site validation "
                f"({len(non_dapi_pairs)} pairs across {len(grouped_non_dapi)} parsed sites"
                f"{', plus ungrouped files' if ungrouped_non_dapi else ''})..."
            )
            non_dapi_pairs, patch_bad_non_dapi_info = filter_pairs_with_patch_validation_by_site(
                non_dapi_pairs,
                axes=axes,
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_axes=patch_axes,
                patch_filter_threshold=patch_filter_threshold,
                extreme_value_cutoff=extreme_value_cutoff,
            )

            if patch_bad_non_dapi_info:
                print("Removed whole NON_DAPI sites due to patch-level validation failure:")
                for item in patch_bad_non_dapi_info[:10]:
                    site_label = f"s{item['site']}" if item["site"] is not None else "[no parsed site]"
                    print(f"  - {site_label} ({item['n_pairs']} pairs)")

        if dapi_pairs:
            grouped_dapi, ungrouped_dapi = group_pairs_by_site(dapi_pairs)
            print(
                f"Checking DAPI_ONLY pairs by batched site validation "
                f"({len(dapi_pairs)} pairs across {len(grouped_dapi)} parsed sites"
                f"{', plus ungrouped files' if ungrouped_dapi else ''})..."
            )
            dapi_pairs, patch_bad_dapi_info = filter_pairs_with_patch_validation_by_site(
                dapi_pairs,
                axes=axes,
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_axes=patch_axes,
                patch_filter_threshold=patch_filter_threshold,
                extreme_value_cutoff=extreme_value_cutoff,
            )

            if patch_bad_dapi_info:
                print("Removed whole DAPI_ONLY sites due to patch-level validation failure:")
                for item in patch_bad_dapi_info[:10]:
                    site_label = f"s{item['site']}" if item["site"] is not None else "[no parsed site]"
                    print(f"  - {site_label} ({item['n_pairs']} pairs)")

        n_pairs_after_sampling = len(non_dapi_pairs) + len(dapi_pairs)

        if file_pairs:
            print("Example matched pair:")
            print("  Source:", file_pairs[0][0])
            print("  Target:", file_pairs[0][1])

        print("Pairs going into final patch generation:")
        print("  NON_DAPI:", len(non_dapi_pairs))
        print("  DAPI_ONLY:", len(dapi_pairs))
        print("  Total:", n_pairs_after_sampling)

        print("Stage 4/4: Final patch generation")
        raw_data_non_dapi = rawdata_from_pairs(non_dapi_pairs, axes=axes) if non_dapi_pairs else None
        raw_data_dapi = rawdata_from_pairs(dapi_pairs, axes=axes) if dapi_pairs else None

        out_dir = sample_dir / patch_dirname
        out_file_non_dapi = out_dir / f"{sample_group_name}__{sample_name}__NON_DAPI__train_patches.npz"
        out_file_dapi = out_dir / f"{sample_group_name}__{sample_name}__DAPI_ONLY__train_patches.npz"

        sample_summary = {
            "sample_name": sample_name,
            "sample_dir": str(sample_dir),
            "sample_group_name": sample_group_name,
            "n_pairs_before_sampling": n_pairs_before_sampling,
            "n_non_dapi_pairs_before_sampling": len(non_dapi_pairs_all),
            "n_dapi_pairs_before_sampling": len(dapi_pairs_all),
            "n_non_dapi_pairs_after_sampling": len(non_dapi_pairs_sampled),
            "n_dapi_pairs_after_sampling": len(dapi_pairs_sampled),
            "n_pairs_after_sampling": n_pairs_after_sampling,
            "n_non_dapi_pairs": len(non_dapi_pairs),
            "n_dapi_pairs": len(dapi_pairs),
            "n_excluded_pairs": excluded_count,
            "n_removed_non_dapi_pairs": removed_non_dapi,
            "n_removed_dapi_pairs": removed_dapi,
            "n_removed_non_dapi_patch_sites": len(patch_bad_non_dapi_info),
            "n_removed_dapi_patch_sites": len(patch_bad_dapi_info),
            "removed_non_dapi_examples": removed_non_dapi_info[:10],
            "removed_dapi_examples": removed_dapi_info[:10],
            "removed_non_dapi_patch_examples": patch_bad_non_dapi_info[:10],
            "removed_dapi_patch_examples": patch_bad_dapi_info[:10],
            "non_dapi_file": None,
            "dapi_file": None,
        }

        if raw_data_non_dapi is None:
            print("Skipping NON_DAPI: no usable non-DAPI pairs remained after filtering.")
        else:
            print("Creating NON_DAPI patches...")
            print("Output file:", out_file_non_dapi)
            Xn, Yn, axes_n = make_patches_for_variant(
                raw_data=raw_data_non_dapi,
                out_file=out_file_non_dapi,
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_axes=patch_axes,
                patch_filter_threshold=patch_filter_threshold,
                extreme_value_cutoff=extreme_value_cutoff,
            )
            print("Finished NON_DAPI patch creation.")
            print("  X shape:", Xn.shape)
            print("  Y shape:", Yn.shape)
            print("  Axes:", axes_n)
            all_patch_files_non_dapi.append(out_file_non_dapi)
            sample_summary["non_dapi_file"] = str(out_file_non_dapi)

            if merge_all_samples:
                merged_X_non_dapi.append(Xn)
                merged_Y_non_dapi.append(Yn)
                if axes_ref_non_dapi is None:
                    axes_ref_non_dapi = axes_n

        if raw_data_dapi is None:
            print("Skipping DAPI_ONLY: no usable DAPI pairs remained after filtering.")
        else:
            print("Creating DAPI_ONLY patches...")
            print("Output file:", out_file_dapi)
            Xd, Yd, axes_d = make_patches_for_variant(
                raw_data=raw_data_dapi,
                out_file=out_file_dapi,
                patch_size=patch_size,
                n_patches_per_image=n_patches_per_image,
                patch_axes=patch_axes,
                patch_filter_threshold=patch_filter_threshold,
                extreme_value_cutoff=extreme_value_cutoff,
            )
            print("Finished DAPI_ONLY patch creation.")
            print("  X shape:", Xd.shape)
            print("  Y shape:", Yd.shape)
            print("  Axes:", axes_d)
            all_patch_files_dapi.append(out_file_dapi)
            sample_summary["dapi_file"] = str(out_file_dapi)

            if merge_all_samples:
                merged_X_dapi.append(Xd)
                merged_Y_dapi.append(Yd)
                if axes_ref_dapi is None:
                    axes_ref_dapi = axes_d

        sample_summaries.append(sample_summary)

    merged_non_dapi_file = None
    merged_dapi_file = None

    if merge_all_samples:
        merged_out_dir = care_root / patch_dirname
        merged_out_dir.mkdir(parents=True, exist_ok=True)

        if merged_X_non_dapi:
            merged_non_dapi_file = merged_out_dir / merged_non_dapi_name
            X = np.concatenate(merged_X_non_dapi, axis=0)
            Y = np.concatenate(merged_Y_non_dapi, axis=0)

            validate_patch_array(X, "Merged X patches", extreme_value_cutoff=extreme_value_cutoff)
            validate_patch_array(Y, "Merged Y patches", extreme_value_cutoff=extreme_value_cutoff)

            np.savez_compressed(merged_non_dapi_file, X=X, Y=Y, axes=axes_ref_non_dapi)

            print("\n" + "=" * 90)
            print("Saved merged NON_DAPI patch file")
            print("File:", merged_non_dapi_file)
            print("  X shape:", X.shape)
            print("  Y shape:", Y.shape)
            print("  Axes:", axes_ref_non_dapi)

        if merged_X_dapi:
            merged_dapi_file = merged_out_dir / merged_dapi_name
            X = np.concatenate(merged_X_dapi, axis=0)
            Y = np.concatenate(merged_Y_dapi, axis=0)

            validate_patch_array(X, "Merged DAPI X patches", extreme_value_cutoff=extreme_value_cutoff)
            validate_patch_array(Y, "Merged DAPI Y patches", extreme_value_cutoff=extreme_value_cutoff)

            np.savez_compressed(merged_dapi_file, X=X, Y=Y, axes=axes_ref_dapi)

            print("\n" + "=" * 90)
            print("Saved merged DAPI_ONLY patch file")
            print("File:", merged_dapi_file)
            print("  X shape:", X.shape)
            print("  Y shape:", Y.shape)
            print("  Axes:", axes_ref_dapi)

    print("\n" + "=" * 90)
    print("Patch generation complete")
    print("=" * 90)

    print("\nPer-sample NON_DAPI patch files:")
    for p in all_patch_files_non_dapi:
        print(" -", p)

    print("\nPer-sample DAPI_ONLY patch files:")
    for p in all_patch_files_dapi:
        print(" -", p)

    metadata = build_run_metadata(
        care_root=care_root,
        care_subdirs=care_subdirs,
        source_dirname=source_dirname,
        target_dirname=target_dirname,
        pattern=pattern,
        axes=axes,
        patch_axes=patch_axes,
        dapi_channel_index=dapi_channel_index,
        patch_size=patch_size,
        n_patches_per_image=n_patches_per_image,
        patch_filter_threshold=patch_filter_threshold,
        patch_dirname=patch_dirname,
        merge_all_samples=merge_all_samples,
        merged_non_dapi_name=merged_non_dapi_name,
        merged_dapi_name=merged_dapi_name,
        max_images_per_sample=max_images_per_sample,
        fraction_images_per_sample=fraction_images_per_sample,
        min_source_max=min_source_max,
        min_source_std=min_source_std,
        min_target_max=min_target_max,
        min_target_std=min_target_std,
        extreme_value_cutoff=extreme_value_cutoff,
        sample_dirs=sample_dirs,
        all_patch_files_non_dapi=all_patch_files_non_dapi,
        all_patch_files_dapi=all_patch_files_dapi,
        merged_non_dapi_file=merged_non_dapi_file,
        merged_dapi_file=merged_dapi_file,
        sample_summaries=sample_summaries,
    )

    metadata_file = save_run_metadata(
        metadata,
        care_root=care_root,
        patch_dirname=patch_dirname,
    )

    return {
        "sample_dirs": sample_dirs,
        "all_patch_files_non_dapi": all_patch_files_non_dapi,
        "all_patch_files_dapi": all_patch_files_dapi,
        "merged_X_non_dapi": merged_X_non_dapi,
        "merged_Y_non_dapi": merged_Y_non_dapi,
        "merged_X_dapi": merged_X_dapi,
        "merged_Y_dapi": merged_Y_dapi,
        "axes_ref_non_dapi": axes_ref_non_dapi,
        "axes_ref_dapi": axes_ref_dapi,
        "merged_non_dapi_file": merged_non_dapi_file,
        "merged_dapi_file": merged_dapi_file,
        "sample_summaries": sample_summaries,
        "metadata": metadata,
        "metadata_file": metadata_file,
    }


def build_run_metadata(
    *,
    care_root: Path,
    care_subdirs: Sequence[str],
    source_dirname: str,
    target_dirname: str,
    pattern: str,
    axes: str,
    patch_axes: str,
    dapi_channel_index: int,
    patch_size,
    n_patches_per_image,
    patch_filter_threshold: float,
    patch_dirname: str,
    merge_all_samples: bool,
    merged_non_dapi_name: str,
    merged_dapi_name: str,
    max_images_per_sample: Optional[int],
    fraction_images_per_sample: Optional[float],
    min_source_max: float,
    min_source_std: float,
    min_target_max: float,
    min_target_std: float,
    extreme_value_cutoff: float,
    sample_dirs: Sequence[Path],
    all_patch_files_non_dapi: Sequence[Path],
    all_patch_files_dapi: Sequence[Path],
    merged_non_dapi_file: Optional[Path],
    merged_dapi_file: Optional[Path],
    sample_summaries: Optional[list[dict]] = None,
) -> dict:
    """
    Build metadata dictionary for the patch generation run.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "dataset": {
            "CARE_ROOT": str(care_root),
            "CARE_SUBDIRS": list(care_subdirs),
            "SOURCE_DIRNAME": source_dirname,
            "TARGET_DIRNAME": target_dirname,
            "PATTERN": pattern,
            "num_samples": len(sample_dirs),
            "sample_dirs": [str(p) for p in sample_dirs],
        },
        "sampling": {
            "MAX_IMAGES_PER_SAMPLE": max_images_per_sample,
            "FRACTION_IMAGES_PER_SAMPLE": fraction_images_per_sample,
        },
        "data_interpretation": {
            "AXES": axes,
            "PATCH_AXES": patch_axes,
            "DAPI_CHANNEL_INDEX": dapi_channel_index,
        },
        "patch_parameters": {
            "PATCH_SIZE": patch_size,
            "N_PATCHES_PER_IMAGE": n_patches_per_image,
            "PATCH_FILTER_THRESHOLD": patch_filter_threshold,
            "PATCH_DIRNAME": patch_dirname,
        },
        "image_filtering": {
            "MIN_SOURCE_MAX": min_source_max,
            "MIN_SOURCE_STD": min_source_std,
            "MIN_TARGET_MAX": min_target_max,
            "MIN_TARGET_STD": min_target_std,
            "EXTREME_VALUE_CUTOFF": extreme_value_cutoff,
        },
        "merge_settings": {
            "MERGE_ALL_SAMPLES": merge_all_samples,
            "MERGED_NON_DAPI_NAME": merged_non_dapi_name,
            "MERGED_DAPI_NAME": merged_dapi_name,
        },
        "generated_files": {
            "per_sample_non_dapi": [str(p) for p in all_patch_files_non_dapi],
            "per_sample_dapi": [str(p) for p in all_patch_files_dapi],
            "merged_non_dapi": str(merged_non_dapi_file) if merged_non_dapi_file is not None else None,
            "merged_dapi": str(merged_dapi_file) if merged_dapi_file is not None else None,
        },
        "sample_summaries": sample_summaries if sample_summaries is not None else [],
    }


def save_run_metadata(
    metadata: dict,
    *,
    care_root: Path,
    patch_dirname: str,
    filename: str = "patch_generation_metadata.json",
) -> Path:
    """
    Save metadata JSON file under care_root / patch_dirname.
    """
    metadata_dir = care_root / patch_dirname
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = metadata_dir / filename
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved run metadata to:", metadata_file)
    return metadata_file
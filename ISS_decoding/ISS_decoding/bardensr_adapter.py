"""SpaceTx adapters and memory-bounded tiling for Bardensr decoding."""

from __future__ import annotations

from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version as distribution_version
from math import ceil
import re

import numpy as np
import pandas as pd

from starfish.types import Axes, Coordinates, Features


BARDENSR_DEFAULT_KWARGS = {
    "method": "singleshot",
    "noisefloor": 0.05,
    "peak_threshold": 0.72,
    "peak_threshold_fraction": None,
    "poolsize": (1, 1, 1),
    "tile_size": (512, 512),
    "overlap": None,
    "normalize_frames": True,
    "l1_penalty": 0.0,
    "psf_radius": (0, 0, 0),
    "iterations": 100,
    "estimate_codebook_gain": True,
    "estimate_colormixing": False,
    "estimate_phasing": False,
    "device": "auto",
    "z_projection": "max",
}


def installed_bardensr_version():
    """Return the installed optional Bardensr version without importing TensorFlow."""
    try:
        return distribution_version("bardensr")
    except PackageNotFoundError:
        return None


def _integer_tuple(value, length, name, *, positive=False):
    values = (value,) * length if np.isscalar(value) else tuple(value)
    if len(values) != length:
        raise ValueError(
            f"Bardensr '{name}' must be a scalar or contain {length} values."
        )
    normalized = tuple(int(item) for item in values)
    minimum = 1 if positive else 0
    if any(item < minimum for item in normalized) or any(
        original != converted for original, converted in zip(values, normalized)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"Bardensr '{name}' must contain {qualifier} integers.")
    return normalized


def effective_bardensr_kwargs(overrides=None):
    """Return validated Bardensr settings after applying user overrides."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(BARDENSR_DEFAULT_KWARGS))
    if unknown:
        raise ValueError(f"Unknown Bardensr setting(s): {unknown}")

    settings = dict(BARDENSR_DEFAULT_KWARGS)
    settings.update(overrides)

    method = str(settings["method"]).strip().lower()
    if method not in {"singleshot", "iterative"}:
        raise ValueError("Bardensr 'method' must be 'singleshot' or 'iterative'.")
    settings["method"] = method

    # Match the thresholds demonstrated in Bardensr's official example.  The
    # iterative density scale is data-dependent, so its default is relative to
    # each memory-bounded tile rather than the singleshot correlation cutoff.
    threshold_was_set = "peak_threshold" in overrides
    fraction_was_set = "peak_threshold_fraction" in overrides
    if method == "iterative" and not threshold_was_set and not fraction_was_set:
        settings["peak_threshold"] = None
        settings["peak_threshold_fraction"] = 0.1

    threshold = settings["peak_threshold"]
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "Bardensr 'peak_threshold' must be finite and non-negative or None."
            )
        settings["peak_threshold"] = threshold

    threshold_fraction = settings["peak_threshold_fraction"]
    if threshold_fraction is not None:
        threshold_fraction = float(threshold_fraction)
        if not np.isfinite(threshold_fraction) or not 0 < threshold_fraction <= 1:
            raise ValueError(
                "Bardensr 'peak_threshold_fraction' must be in (0, 1] or None."
            )
        settings["peak_threshold_fraction"] = threshold_fraction
    if (threshold is None) == (threshold_fraction is None):
        raise ValueError(
            "Set exactly one of Bardensr 'peak_threshold' and "
            "'peak_threshold_fraction'."
        )

    noisefloor = float(settings["noisefloor"])
    if not np.isfinite(noisefloor) or noisefloor <= 0:
        raise ValueError("Bardensr 'noisefloor' must be finite and positive.")
    settings["noisefloor"] = noisefloor

    l1_penalty = float(settings["l1_penalty"])
    if not np.isfinite(l1_penalty) or l1_penalty < 0:
        raise ValueError(
            "Bardensr 'l1_penalty' must be finite and non-negative."
        )
    settings["l1_penalty"] = l1_penalty

    iterations = settings["iterations"]
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations <= 0
    ):
        raise ValueError("Bardensr 'iterations' must be a positive integer.")

    settings["poolsize"] = _integer_tuple(
        settings["poolsize"], 3, "poolsize"
    )
    settings["psf_radius"] = _integer_tuple(
        settings["psf_radius"], 3, "psf_radius"
    )
    settings["tile_size"] = _integer_tuple(
        settings["tile_size"], 2, "tile_size", positive=True
    )

    overlap = settings["overlap"]
    if overlap is None:
        peak_halo = max(settings["poolsize"][1:])
        psf_halo = (
            ceil(3 * max(settings["psf_radius"][1:]))
            if method == "iterative"
            else 0
        )
        overlap = max(peak_halo, psf_halo)
    elif not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("Bardensr 'overlap' must be a non-negative integer or None.")
    settings["overlap"] = overlap
    if any(size <= 2 * overlap for size in settings["tile_size"]):
        raise ValueError("Bardensr 'tile_size' must be greater than twice the overlap.")

    for name in (
        "normalize_frames",
        "estimate_codebook_gain",
        "estimate_colormixing",
        "estimate_phasing",
    ):
        if not isinstance(settings[name], (bool, np.bool_)):
            raise ValueError(f"Bardensr '{name}' must be boolean.")
        settings[name] = bool(settings[name])

    if settings["estimate_phasing"]:
        raise ValueError(
            "Bardensr phasing estimation is not implemented by the upstream decoder."
        )
    if method == "singleshot" and settings["estimate_colormixing"]:
        raise ValueError(
            "Bardensr 'estimate_colormixing' is available only with method='iterative'."
        )

    device = str(settings["device"]).strip().lower()
    if not device:
        raise ValueError("Bardensr 'device' cannot be empty.")
    settings["device"] = device

    z_projection = str(settings["z_projection"]).strip().lower()
    if z_projection not in {"max", "mean"}:
        raise ValueError("Bardensr 'z_projection' must be 'max' or 'mean'.")
    settings["z_projection"] = z_projection
    return settings


def format_spacetx_codebook_for_bardensr(codebook):
    """Convert a SpaceTx codebook to Bardensr ``frames x targets`` order."""
    expected_dims = (Features.TARGET, Axes.ROUND.value, Axes.CH.value)
    if not all(dim in codebook.dims for dim in expected_dims):
        raise ValueError(
            "The SpaceTx codebook must have target, round, and channel dimensions."
        )

    ordered = codebook.transpose(*expected_dims)
    barcodes = np.asarray(ordered.values, dtype=np.float32)
    if barcodes.ndim != 3:
        raise ValueError(
            "The SpaceTx codebook must have shape (targets, rounds, channels)."
        )
    if not np.isfinite(barcodes).all():
        raise ValueError("The SpaceTx codebook contains non-finite values.")
    if not np.all(np.isclose(barcodes, 0) | np.isclose(barcodes, 1)):
        raise ValueError("Bardensr requires a binary one-hot codebook.")
    if not np.allclose(barcodes.sum(axis=2), 1):
        raise ValueError(
            "Bardensr requires exactly one active channel for every target and round."
        )

    target_names = np.asarray(ordered.coords[Features.TARGET].values).astype(str)
    frames_by_targets = barcodes.transpose(1, 2, 0).reshape(-1, barcodes.shape[0])
    return frames_by_targets, target_names


def format_spacetx_image_for_bardensr(image_stack, z_projection="max"):
    """Convert a Starfish ImageStack to Bardensr ``frames x 1 x y x x`` order."""
    z_projection = str(z_projection).strip().lower()
    if z_projection not in {"max", "mean"}:
        raise ValueError("z_projection must be 'max' or 'mean'.")

    expected_dims = (
        Axes.ROUND.value,
        Axes.CH.value,
        Axes.ZPLANE.value,
        Axes.Y.value,
        Axes.X.value,
    )
    data = image_stack.xarray
    if not all(dim in data.dims for dim in expected_dims):
        raise ValueError(
            "The SpaceTx image must have round, channel, z, y, and x dimensions."
        )
    ordered = np.asarray(data.transpose(*expected_dims).values, dtype=np.float32)
    projected = (
        ordered.max(axis=2, keepdims=True)
        if z_projection == "max"
        else ordered.mean(axis=2, keepdims=True)
    )
    frames = projected.reshape(-1, 1, projected.shape[-2], projected.shape[-1])
    if not np.isfinite(frames).all():
        raise ValueError("The SpaceTx image contains non-finite values.")
    if (frames < 0).any():
        raise ValueError("Bardensr requires non-negative image intensities.")
    return frames


def _load_bardensr_runtime():
    try:
        import bardensr
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "Bardensr is not installed. Install ISS_decoding with the 'bardensr' "
            "extra or create the environment from ISS_decoding.yml."
        ) from exc
    return bardensr, tf


def _resolve_device(tf, requested):
    normalized = str(requested).strip().lower().lstrip("/")
    gpus = tf.config.list_physical_devices("GPU")
    if normalized == "auto":
        return "/GPU:0" if gpus else "/CPU:0"
    if normalized in {"cpu", "cpu:0"}:
        return "/CPU:0"
    match = re.fullmatch(r"(?:gpu|cuda)(?::(\d+))?", normalized)
    if match:
        index = int(match.group(1) or 0)
        if index >= len(gpus):
            raise RuntimeError(
                f"Bardensr GPU {index} was requested, but TensorFlow found "
                f"{len(gpus)} GPU device(s)."
            )
        return f"/GPU:{index}"
    raise ValueError(
        "Bardensr 'device' must be 'auto', 'cpu', 'gpu', 'cuda', or a GPU index."
    )


def _core_windows(length, input_size, overlap):
    core_size = input_size - 2 * overlap
    for core_start in range(0, length, core_size):
        core_stop = min(length, core_start + core_size)
        read_start = max(0, core_start - overlap)
        read_stop = min(length, core_stop + overlap)
        yield core_start, core_stop, read_start, read_stop


def _normalize_frames(images):
    minima = images.min(axis=(1, 2, 3), keepdims=True)
    shifted = images - minima
    maxima = shifted.max(axis=(1, 2, 3), keepdims=True)
    return np.divide(
        shifted,
        maxima,
        out=np.zeros_like(shifted, dtype=np.float32),
        where=maxima > 0,
    )


def _empty_decoded_table():
    columns = {
        Features.SPOT_ID: "int64",
        Axes.X.value: "float64",
        Axes.Y.value: "float64",
        Axes.ZPLANE.value: "int64",
        "xc": "float64",
        "yc": "float64",
        "zc": "float64",
        Features.TARGET: "object",
        "candidate_target": "object",
        "target_id": "int64",
        "assignment_class": "object",
        Features.PASSES_THRESHOLDS: "bool",
        "bardensr_evidence": "float64",
        "bardensr_peak_threshold": "float64",
        "bardensr_tile_max": "float64",
        "bardensr_tile": "object",
        "bardensr_method": "object",
        "decoder": "object",
        "spot_detector": "object",
    }
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})


def _physical_axis(image_stack, coordinate, expected_length):
    try:
        values = np.asarray(image_stack.xarray[coordinate.value].values, dtype=float)
    except (AttributeError, KeyError):
        return None
    return values if values.ndim == 1 and len(values) == expected_length else None


def decode_bardensr_array(images, codebook, target_names, *, rounds, settings):
    """Jointly detect and decode a Bardensr array in overlapping image tiles."""
    images = np.asarray(images, dtype=np.float32)
    codebook = np.asarray(codebook, dtype=np.float32)
    target_names = np.asarray(target_names).astype(str)
    if images.ndim != 4:
        raise ValueError("Bardensr images must have shape (frames, z, y, x).")
    if codebook.ndim != 2 or images.shape[0] != codebook.shape[0]:
        raise ValueError("Bardensr image and codebook frame dimensions differ.")
    if target_names.shape != (codebook.shape[1],):
        raise ValueError("Bardensr target names must contain one value per target.")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValueError("Bardensr 'rounds' must be a positive integer.")
    if images.shape[0] % rounds:
        raise ValueError("Bardensr image frames must divide evenly into rounds.")
    if not np.isfinite(images).all() or (images < 0).any():
        raise ValueError("Bardensr images must contain finite, non-negative values.")

    settings = effective_bardensr_kwargs(settings)
    if settings["normalize_frames"]:
        images = _normalize_frames(images)

    bardensr, tf = _load_bardensr_runtime()
    device = _resolve_device(tf, settings["device"])
    tile_height, tile_width = settings["tile_size"]
    overlap = settings["overlap"]
    rows = []

    y_windows = list(_core_windows(images.shape[2], tile_height, overlap))
    x_windows = list(_core_windows(images.shape[3], tile_width, overlap))
    for row_index, (core_y0, core_y1, read_y0, read_y1) in enumerate(y_windows):
        for column_index, (core_x0, core_x1, read_x0, read_x1) in enumerate(x_windows):
            image_tile = images[:, :, read_y0:read_y1, read_x0:read_x1]
            device_context = tf.device(device) if hasattr(tf, "device") else nullcontext()
            with device_context:
                if settings["method"] == "singleshot":
                    density = bardensr.spot_calling.estimate_density_singleshot(
                        image_tile,
                        codebook,
                        noisefloor=settings["noisefloor"],
                    )
                else:
                    density, _diagnostics = (
                        bardensr.spot_calling.estimate_density_iterative(
                            image_tile,
                            codebook,
                            l1_penalty=settings["l1_penalty"],
                            psf_radius=settings["psf_radius"],
                            iterations=settings["iterations"],
                            estimate_codebook_gain=settings["estimate_codebook_gain"],
                            rounds=rounds,
                            estimate_colormixing=settings["estimate_colormixing"],
                            estimate_phasing=False,
                        )
                    )

                density = np.asarray(density)
                tile_max = float(density.max()) if density.size else 0.0
                threshold = settings["peak_threshold"]
                if threshold is None:
                    threshold = tile_max * settings["peak_threshold_fraction"]
                peaks = bardensr.spot_calling.find_peaks(
                    density,
                    thresh=threshold,
                    poolsize=settings["poolsize"],
                )

            required_columns = {"m0", "m1", "m2", "j"}
            if not required_columns.issubset(peaks.columns):
                raise ValueError(
                    "Bardensr peak output is missing required coordinate columns."
                )
            for peak in peaks.itertuples(index=False):
                local_z = int(peak.m0)
                local_y = int(peak.m1)
                local_x = int(peak.m2)
                target_id = int(peak.j)
                global_y = read_y0 + local_y
                global_x = read_x0 + local_x
                if not (
                    core_y0 <= global_y < core_y1
                    and core_x0 <= global_x < core_x1
                ):
                    continue
                rows.append(
                    {
                        Axes.X.value: float(global_x),
                        Axes.Y.value: float(global_y),
                        Axes.ZPLANE.value: local_z,
                        Features.TARGET: target_names[target_id],
                        "candidate_target": target_names[target_id],
                        "target_id": target_id,
                        "assignment_class": "gene",
                        Features.PASSES_THRESHOLDS: True,
                        "bardensr_evidence": float(
                            density[local_z, local_y, local_x, target_id]
                        ),
                        "bardensr_peak_threshold": float(threshold),
                        "bardensr_tile_max": tile_max,
                        "bardensr_tile": f"{row_index}_{column_index}",
                        "bardensr_method": settings["method"],
                        "decoder": "bardensr",
                        "spot_detector": "bardensr_joint",
                    }
                )

    if not rows:
        result = _empty_decoded_table()
    else:
        result = pd.DataFrame(rows)
        result.insert(0, Features.SPOT_ID, np.arange(len(result), dtype=int))
        result["xc"] = np.nan
        result["yc"] = np.nan
        result["zc"] = np.nan
    result.attrs["bardensr_device"] = device
    result.attrs["bardensr_method"] = settings["method"]
    return result


def decode_imagestack_with_bardensr(image_stack, codebook, bardensr_kwargs=None):
    """Decode a preprocessed Starfish ImageStack with tiled Bardensr."""
    settings = effective_bardensr_kwargs(bardensr_kwargs)
    images = format_spacetx_image_for_bardensr(
        image_stack,
        z_projection=settings["z_projection"],
    )
    frames_by_targets, target_names = format_spacetx_codebook_for_bardensr(codebook)
    rounds = int(codebook.sizes[Axes.ROUND.value])
    result = decode_bardensr_array(
        images,
        frames_by_targets,
        target_names,
        rounds=rounds,
        settings=settings,
    )

    x_coordinates = _physical_axis(image_stack, Coordinates.X, images.shape[3])
    y_coordinates = _physical_axis(image_stack, Coordinates.Y, images.shape[2])
    z_coordinates = _physical_axis(
        image_stack,
        Coordinates.Z,
        image_stack.xarray.sizes[Axes.ZPLANE.value],
    )
    if len(result):
        if x_coordinates is not None:
            result["xc"] = np.interp(
                result[Axes.X.value], np.arange(images.shape[3]), x_coordinates
            )
        if y_coordinates is not None:
            result["yc"] = np.interp(
                result[Axes.Y.value], np.arange(images.shape[2]), y_coordinates
            )
        if z_coordinates is not None and len(z_coordinates):
            result["zc"] = float(z_coordinates[0])
    return result

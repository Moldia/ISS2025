"""Adapters and memory-bounded tiling for ISTDECO image-level decoding."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as distribution_version
from math import ceil

import numpy as np
import pandas as pd

from starfish.types import Axes, Coordinates, Features


ISTDECO_DEFAULT_KWARGS = {
    "sigma": 1.2,
    "background": 1e-8,
    "scale": 1.0,
    "niter": 75,
    "acceleration": 1.0,
    "suppress_radius": 1,
    "tile_size": (512, 512),
    "overlap": None,
    "intensity_percentile": 99.0,
    "intensity_threshold": None,
    "quality_threshold": 0.5,
    "device": "auto",
    "z_projection": "max",
}


def installed_istdeco_version():
    """Return the installed optional ISTDECO version without importing PyTorch."""
    try:
        return distribution_version("istdeco")
    except PackageNotFoundError:
        return None


def _positive_pair(value, name):
    values = (value, value) if np.isscalar(value) else tuple(value)
    if len(values) != 2:
        raise ValueError(f"ISTDECO '{name}' must be a scalar or a pair.")
    normalized = tuple(int(item) for item in values)
    if any(item <= 0 for item in normalized) or any(
        original != converted for original, converted in zip(values, normalized)
    ):
        raise ValueError(f"ISTDECO '{name}' must contain positive integers.")
    return normalized


def _sigma_pair(value):
    values = (value, value) if np.isscalar(value) else tuple(value)
    if len(values) != 2:
        raise ValueError("ISTDECO 'sigma' must be a scalar or a pair.")
    normalized = tuple(float(item) for item in values)
    if any(not np.isfinite(item) or item < 0 for item in normalized):
        raise ValueError("ISTDECO 'sigma' must contain finite, non-negative values.")
    return normalized


def effective_istdeco_kwargs(overrides=None):
    """Return validated ISTDECO settings after applying user overrides."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(ISTDECO_DEFAULT_KWARGS))
    if unknown:
        raise ValueError(f"Unknown ISTDECO setting(s): {unknown}")

    settings = dict(ISTDECO_DEFAULT_KWARGS)
    settings.update(overrides)
    settings["sigma"] = _sigma_pair(settings["sigma"])
    settings["tile_size"] = _positive_pair(settings["tile_size"], "tile_size")

    for name in ("background", "scale", "acceleration"):
        value = float(settings[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"ISTDECO '{name}' must be finite and positive.")
        settings[name] = value

    niter = settings["niter"]
    if not isinstance(niter, int) or isinstance(niter, bool) or niter <= 0:
        raise ValueError("ISTDECO 'niter' must be a positive integer.")

    suppress_radius = settings["suppress_radius"]
    if (
        suppress_radius is not None
        and (
            not isinstance(suppress_radius, int)
            or isinstance(suppress_radius, bool)
            or suppress_radius < 0
        )
    ):
        raise ValueError(
            "ISTDECO 'suppress_radius' must be a non-negative integer or None."
        )

    overlap = settings["overlap"]
    if overlap is None:
        suppress_halo = ceil((suppress_radius or 0) / settings["scale"])
        overlap = max(ceil(3 * max(settings["sigma"])), suppress_halo)
    elif not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("ISTDECO 'overlap' must be a non-negative integer or None.")
    settings["overlap"] = overlap
    if any(size <= 2 * overlap for size in settings["tile_size"]):
        raise ValueError("ISTDECO 'tile_size' must be greater than twice the overlap.")

    percentile = settings["intensity_percentile"]
    if percentile is not None:
        percentile = float(percentile)
        if not np.isfinite(percentile) or not 0 <= percentile <= 100:
            raise ValueError(
                "ISTDECO 'intensity_percentile' must be between 0 and 100 or None."
            )
        settings["intensity_percentile"] = percentile

    threshold = settings["intensity_threshold"]
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "ISTDECO 'intensity_threshold' must be finite and non-negative."
            )
        settings["intensity_threshold"] = threshold
    elif percentile is None:
        raise ValueError(
            "Set either ISTDECO 'intensity_threshold' or 'intensity_percentile'."
        )

    quality_threshold = float(settings["quality_threshold"])
    if not np.isfinite(quality_threshold) or quality_threshold < 0:
        raise ValueError(
            "ISTDECO 'quality_threshold' must be finite and non-negative."
        )
    settings["quality_threshold"] = quality_threshold

    device = str(settings["device"]).strip().lower()
    if not device:
        raise ValueError("ISTDECO 'device' cannot be empty.")
    settings["device"] = device

    z_projection = str(settings["z_projection"]).strip().lower()
    if z_projection not in {"max", "mean"}:
        raise ValueError("ISTDECO 'z_projection' must be 'max' or 'mean'.")
    settings["z_projection"] = z_projection
    return settings


def format_spacetx_codebook_for_istdeco(codebook):
    """Return a one-hot SpaceTx codebook as ISTDECO ``codes x rounds x channels``."""
    expected_dims = (Features.TARGET, Axes.ROUND.value, Axes.CH.value)
    if not all(dim in codebook.dims for dim in expected_dims):
        raise ValueError(
            "The SpaceTx codebook must have target, round, and channel dimensions."
        )

    ordered = codebook.transpose(*expected_dims)
    barcodes = np.asarray(ordered.values, dtype=np.float32)
    if barcodes.ndim != 3:
        raise ValueError(
            "The SpaceTx codebook must have shape (barcodes, rounds, channels)."
        )
    if not np.isfinite(barcodes).all():
        raise ValueError("The SpaceTx codebook contains non-finite values.")
    if not np.all(np.isclose(barcodes, 0) | np.isclose(barcodes, 1)):
        raise ValueError("ISTDECO requires a binary one-hot codebook.")
    if not np.allclose(barcodes.sum(axis=2), 1):
        raise ValueError(
            "ISTDECO requires exactly one active channel for every barcode and round."
        )

    target_names = np.asarray(ordered.coords[Features.TARGET].values).astype(str)
    return barcodes, target_names


def format_spacetx_image_for_istdeco(image_stack, z_projection="max"):
    """Convert a Starfish ImageStack into ISTDECO ``rounds x channels x y x``."""
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
    images = ordered.max(axis=2) if z_projection == "max" else ordered.mean(axis=2)
    if not np.isfinite(images).all():
        raise ValueError("The SpaceTx image contains non-finite values.")
    if (images < 0).any():
        raise ValueError("ISTDECO requires non-negative image intensities.")
    return images


def _resolve_device(requested):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required by ISTDECO. Install the CUDA/CPU build appropriate "
            "for this machine, then install ISS_decoding with the 'istdeco' extra."
        ) from exc

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid ISTDECO device: {requested!r}.") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"ISTDECO device {requested!r} was requested, but CUDA is not available."
        )
    return str(device)


def _load_istdeco_class():
    try:
        from istdeco import ISTDeco
    except ImportError as exc:
        raise ImportError(
            "ISTDECO is not installed. Install ISS_decoding with the 'istdeco' "
            "extra or create the environment from ISS_decoding.yml."
        ) from exc
    return ISTDeco


def _core_windows(length, input_size, overlap):
    core_size = input_size - 2 * overlap
    for core_start in range(0, length, core_size):
        core_stop = min(length, core_start + core_size)
        read_start = max(0, core_start - overlap)
        read_stop = min(length, core_stop + overlap)
        yield core_start, core_stop, read_start, read_stop


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
        "istdeco_intensity": "float64",
        "istdeco_quality": "float64",
        "istdeco_intensity_threshold": "float64",
        "istdeco_tile": "object",
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


def decode_istdeco_array(images, barcodes, target_names, *, settings):
    """Decode an ISTDECO-formatted array in overlapping, memory-bounded tiles."""
    images = np.asarray(images, dtype=np.float32)
    barcodes = np.asarray(barcodes, dtype=np.float32)
    target_names = np.asarray(target_names).astype(str)
    if images.ndim != 4:
        raise ValueError("ISTDECO images must have shape (rounds, channels, y, x).")
    if barcodes.ndim != 3 or images.shape[:2] != barcodes.shape[1:]:
        raise ValueError("ISTDECO image and codebook round/channel dimensions differ.")
    if target_names.shape != (barcodes.shape[0],):
        raise ValueError("ISTDECO target names must contain one value per barcode.")
    if not np.isfinite(images).all() or (images < 0).any():
        raise ValueError("ISTDECO images must contain finite, non-negative values.")

    settings = effective_istdeco_kwargs(settings)
    threshold = settings["intensity_threshold"]
    if threshold is None:
        threshold = float(np.percentile(images, settings["intensity_percentile"]))

    device = _resolve_device(settings["device"])
    model_class = _load_istdeco_class()
    tile_height, tile_width = settings["tile_size"]
    overlap = settings["overlap"]
    scale = settings["scale"]
    rows = []

    y_windows = list(_core_windows(images.shape[2], tile_height, overlap))
    x_windows = list(_core_windows(images.shape[3], tile_width, overlap))
    for row_index, (core_y0, core_y1, read_y0, read_y1) in enumerate(y_windows):
        for column_index, (core_x0, core_x1, read_x0, read_x1) in enumerate(x_windows):
            image_tile = images[:, :, read_y0:read_y1, read_x0:read_x1]
            model = model_class(
                image_tile,
                barcodes,
                sigma=settings["sigma"],
                b=settings["background"],
                scale=scale,
            ).to(device)
            intensity, quality, _loss = model.run(
                niter=settings["niter"],
                acc=settings["acceleration"],
                suppress_radius=settings["suppress_radius"],
            )
            code_index, local_y, local_x = np.where(
                (intensity > threshold) & (quality > settings["quality_threshold"])
            )
            output_height, output_width = intensity.shape[-2:]
            input_height, input_width = image_tile.shape[-2:]
            y_factor = (
                (input_height - 1) / (output_height - 1)
                if input_height > 1 and output_height > 1
                else 0.0
            )
            x_factor = (
                (input_width - 1) / (output_width - 1)
                if input_width > 1 and output_width > 1
                else 0.0
            )
            global_y = read_y0 + local_y.astype(float) * y_factor
            global_x = read_x0 + local_x.astype(float) * x_factor
            owned = (
                (global_y >= core_y0)
                & (global_y < core_y1)
                & (global_x >= core_x0)
                & (global_x < core_x1)
            )
            for code, y_value, x_value, local_y_value, local_x_value in zip(
                code_index[owned],
                global_y[owned],
                global_x[owned],
                local_y[owned],
                local_x[owned],
            ):
                rows.append(
                    {
                        Axes.X.value: x_value,
                        Axes.Y.value: y_value,
                        Axes.ZPLANE.value: 0,
                        Features.TARGET: target_names[code],
                        "candidate_target": target_names[code],
                        "target_id": int(code),
                        "assignment_class": "gene",
                        Features.PASSES_THRESHOLDS: True,
                        "istdeco_intensity": float(intensity[code, local_y_value, local_x_value]),
                        "istdeco_quality": float(quality[code, local_y_value, local_x_value]),
                        "istdeco_tile": f"{row_index}_{column_index}",
                        "decoder": "istdeco",
                        "spot_detector": "istdeco_joint",
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
    result["istdeco_intensity_threshold"] = threshold
    result.attrs["istdeco_intensity_threshold"] = threshold
    result.attrs["istdeco_device"] = device
    return result


def decode_imagestack_with_istdeco(image_stack, codebook, istdeco_kwargs=None):
    """Decode a preprocessed Starfish ImageStack with tiled ISTDECO."""
    settings = effective_istdeco_kwargs(istdeco_kwargs)
    images = format_spacetx_image_for_istdeco(
        image_stack,
        z_projection=settings["z_projection"],
    )
    barcodes, target_names = format_spacetx_codebook_for_istdeco(codebook)
    result = decode_istdeco_array(
        images,
        barcodes,
        target_names,
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

"""SpaceTx adapters and standardized outputs for Graph-ISS decoding."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as distribution_version

import numpy as np
import pandas as pd

from starfish.types import Axes, Coordinates, Features


GRAPHISS_DEFAULT_KWARGS = {
    "h": 0.05,
    "radius": 3,
    "candidate_probability_threshold": None,
    "graph_radius": 3.0,
    "transition_radius": 4.0,
    "spatial_decay": 0.33,
    "search_mode": "prior",
    "quality_distance_scale": 3.0,
    "quality_threshold": None,
    "min_signal_probability": None,
    "max_distance": None,
    "normalize_frames": True,
    "z_projection": "max",
    "verbose": True,
    "max_candidates": None,
    "max_component_size": None,
}


def installed_graphiss_version():
    """Return the optional Graph-ISS version without loading its model weights."""
    try:
        return distribution_version("graph-iss")
    except PackageNotFoundError:
        return None


def _optional_nonnegative(value, name, *, upper=None):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value < 0 or (upper is not None and value > upper):
        suffix = f" and at most {upper}" if upper is not None else ""
        raise ValueError(
            f"Graph-ISS '{name}' must be finite, non-negative{suffix}, or None."
        )
    return value


def effective_graphiss_kwargs(overrides=None):
    """Return validated Graph-ISS settings after applying user overrides."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(GRAPHISS_DEFAULT_KWARGS))
    if unknown:
        raise ValueError(f"Unknown Graph-ISS setting(s): {unknown}")

    settings = dict(GRAPHISS_DEFAULT_KWARGS)
    settings.update(overrides)

    h = float(settings["h"])
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Graph-ISS 'h' must be finite and positive.")
    settings["h"] = h

    radius = settings["radius"]
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
        raise ValueError("Graph-ISS 'radius' must be a positive integer.")

    for name in (
        "graph_radius",
        "transition_radius",
        "spatial_decay",
        "quality_distance_scale",
    ):
        value = float(settings[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Graph-ISS '{name}' must be finite and positive.")
        settings[name] = value

    settings["candidate_probability_threshold"] = _optional_nonnegative(
        settings["candidate_probability_threshold"],
        "candidate_probability_threshold",
        upper=1,
    )
    settings["quality_threshold"] = _optional_nonnegative(
        settings["quality_threshold"], "quality_threshold"
    )
    settings["min_signal_probability"] = _optional_nonnegative(
        settings["min_signal_probability"], "min_signal_probability", upper=1
    )
    settings["max_distance"] = _optional_nonnegative(
        settings["max_distance"], "max_distance"
    )

    search_mode = str(settings["search_mode"]).strip().lower()
    if search_mode not in {"prior", "blind"}:
        raise ValueError("Graph-ISS 'search_mode' must be 'prior' or 'blind'.")
    settings["search_mode"] = search_mode

    if not isinstance(settings["normalize_frames"], (bool, np.bool_)):
        raise ValueError("Graph-ISS 'normalize_frames' must be boolean.")
    settings["normalize_frames"] = bool(settings["normalize_frames"])

    z_projection = str(settings["z_projection"]).strip().lower()
    if z_projection not in {"max", "mean"}:
        raise ValueError("Graph-ISS 'z_projection' must be 'max' or 'mean'.")
    settings["z_projection"] = z_projection

    if not isinstance(settings["verbose"], (bool, np.bool_)):
        raise ValueError("Graph-ISS 'verbose' must be boolean.")
    settings["verbose"] = bool(settings["verbose"])

    for name in ("max_candidates", "max_component_size"):
        value = settings[name]
        if value is not None:
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value < 1
            ):
                raise ValueError(
                    f"Graph-ISS '{name}' must be a positive integer or None."
                )
            settings[name] = int(value)
    return settings


def format_spacetx_codebook_for_graphiss(codebook):
    """Convert a SpaceTx codebook to Graph-ISS target x round x channel order."""
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
        raise ValueError("Graph-ISS requires a binary one-hot codebook.")
    if not np.allclose(barcodes.sum(axis=2), 1):
        raise ValueError(
            "Graph-ISS requires exactly one active channel for every target and round."
        )
    target_names = np.asarray(ordered.coords[Features.TARGET].values).astype(str)
    return barcodes, target_names


def format_spacetx_image_for_graphiss(image_stack, z_projection="max"):
    """Convert a Starfish ImageStack to Graph-ISS round x channel x y x x order."""
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
        raise ValueError("Graph-ISS requires non-negative image intensities.")
    return images


def _normalize_frames(images):
    minima = images.min(axis=(2, 3), keepdims=True)
    shifted = images - minima
    maxima = shifted.max(axis=(2, 3), keepdims=True)
    return np.divide(
        shifted,
        maxima,
        out=np.zeros_like(shifted, dtype=np.float32),
        where=maxima > 0,
    )


def _load_graphiss_decoder():
    try:
        from graph_iss import decode
    except ImportError as exc:
        raise ImportError(
            "Graph-ISS is not installed. Install ISS_decoding with the 'graphiss' "
            "extra or create the environment from ISS_decoding.yml."
        ) from exc
    return decode


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
        "graphiss_sequence": "object",
        "graphiss_ambiguous_target": "bool",
        "graphiss_signal_probability_sum": "float64",
        "graphiss_signal_probability_mean": "float64",
        "graphiss_signal_probability_min": "float64",
        "graphiss_max_distance": "float64",
        "graphiss_transition_probability_product": "float64",
        "graphiss_spatial_quality": "float64",
        "graphiss_quality": "float64",
        "graphiss_search_mode": "object",
        "graphiss_path_candidate_indices": "object",
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


def decode_graphiss_array(images, barcodes, target_names, *, settings):
    """Jointly detect and decode a Graph-ISS-formatted image array."""
    images = np.asarray(images, dtype=np.float32)
    barcodes = np.asarray(barcodes, dtype=np.float32)
    target_names = np.asarray(target_names).astype(str)
    if images.ndim != 4:
        raise ValueError("Graph-ISS images must have shape (rounds, channels, y, x).")
    if barcodes.ndim != 3 or images.shape[:2] != barcodes.shape[1:]:
        raise ValueError("Graph-ISS image and codebook round/channel dimensions differ.")
    if target_names.shape != (barcodes.shape[0],):
        raise ValueError("Graph-ISS target names must contain one value per barcode.")
    if not np.isfinite(images).all() or (images < 0).any():
        raise ValueError("Graph-ISS images must contain finite, non-negative values.")

    settings = effective_graphiss_kwargs(settings)
    if settings["normalize_frames"]:
        images = _normalize_frames(images)

    decode = _load_graphiss_decoder()
    raw = decode(
        images,
        barcodes,
        target_names,
        h=settings["h"],
        radius=settings["radius"],
        candidate_probability_threshold=settings["candidate_probability_threshold"],
        graph_radius=settings["graph_radius"],
        transition_radius=settings["transition_radius"],
        spatial_decay=settings["spatial_decay"],
        search_mode=settings["search_mode"],
        quality_distance_scale=settings["quality_distance_scale"],
        verbose=settings["verbose"],
        max_candidates=settings["max_candidates"],
        max_component_size=settings["max_component_size"],
    )
    diagnostics = dict(raw.attrs.get("diagnostics", {}))
    if raw.empty:
        result = _empty_decoded_table()
        result.attrs["graphiss_diagnostics"] = diagnostics
        return result

    result = pd.DataFrame(
        {
            Features.SPOT_ID: raw["spot_id"].astype(int),
            Axes.X.value: raw["x"].astype(float),
            Axes.Y.value: raw["y"].astype(float),
            Axes.ZPLANE.value: raw["z"].round().astype(int),
            "target_id": raw["target_id"].astype(int),
            "candidate_target": raw["target"].where(
                raw["target_id"] >= 0,
                "sequence:" + raw["sequence"].astype(str),
            ),
            "graphiss_sequence": raw["sequence"].astype(str),
            "graphiss_ambiguous_target": raw["ambiguous_target"].astype(bool),
            "graphiss_signal_probability_sum": raw[
                "signal_probability_sum"
            ].astype(float),
            "graphiss_signal_probability_mean": raw[
                "signal_probability_mean"
            ].astype(float),
            "graphiss_signal_probability_min": raw[
                "signal_probability_min"
            ].astype(float),
            "graphiss_max_distance": raw["max_distance"].astype(float),
            "graphiss_transition_probability_product": raw[
                "transition_probability_product"
            ].astype(float),
            "graphiss_spatial_quality": raw["spatial_quality"].astype(float),
            "graphiss_quality": raw["quality"].astype(float),
            "graphiss_search_mode": settings["search_mode"],
            "graphiss_path_candidate_indices": raw["path_candidate_indices"],
        }
    )
    passes = pd.Series(True, index=result.index)
    if settings["quality_threshold"] is not None:
        passes &= result["graphiss_quality"] >= settings["quality_threshold"]
    if settings["min_signal_probability"] is not None:
        passes &= (
            result["graphiss_signal_probability_min"]
            >= settings["min_signal_probability"]
        )
    if settings["max_distance"] is not None:
        passes &= result["graphiss_max_distance"] <= settings["max_distance"]

    is_gene = result["target_id"] >= 0
    is_ambiguous = result["graphiss_ambiguous_target"]
    passes &= is_gene & ~is_ambiguous
    result[Features.PASSES_THRESHOLDS] = passes
    result[Features.TARGET] = result["candidate_target"].where(passes)
    result["assignment_class"] = np.select(
        [~is_gene, is_ambiguous, ~passes],
        ["unexpected_sequence", "ambiguous_code", "low_confidence"],
        default="gene",
    )
    result["decoder"] = "graphiss"
    result["spot_detector"] = "graphiss_joint"
    result["xc"] = np.nan
    result["yc"] = np.nan
    result["zc"] = np.nan
    result.attrs["graphiss_diagnostics"] = diagnostics
    return result


def decode_imagestack_with_graphiss(image_stack, codebook, graphiss_kwargs=None):
    """Decode a preprocessed Starfish ImageStack with Graph-ISS."""
    settings = effective_graphiss_kwargs(graphiss_kwargs)
    images = format_spacetx_image_for_graphiss(
        image_stack, z_projection=settings["z_projection"]
    )
    barcodes, target_names = format_spacetx_codebook_for_graphiss(codebook)
    result = decode_graphiss_array(
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

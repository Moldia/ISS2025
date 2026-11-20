"""RCA-composite registration-reference utilities for ISS preprocessing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile


_NUCLEAR_CHANNEL_NAMES = {"dapi", "hoechst", "nuclei", "nuclear"}


def _normalise_name(value) -> str:
    return str(value).strip().casefold()


def _resolve_named_channels(channels, channel_names):
    """Resolve a mixture of channel names and indices to integer indices."""
    names = list(channel_names)
    lookup = {_normalise_name(name): index for index, name in enumerate(names)}
    resolved = []
    for channel in channels:
        if isinstance(channel, (int, np.integer)):
            index = int(channel)
        else:
            key = _normalise_name(channel)
            if key not in lookup:
                raise ValueError(
                    f"Unknown channel name {channel!r}; available names are {names}"
                )
            index = lookup[key]
        resolved.append(index)
    return resolved


def resolve_rca_channels(
    channels,
    *,
    n_channels: int,
    nuclear_channel: Optional[int] = None,
    channel_names: Optional[Sequence[str]] = None,
) -> list[int]:
    """Resolve and validate channels used to build an RCA composite.

    When ``channels`` is ``None``, all channels except ``nuclear_channel`` are
    selected.  This is how the preprocessing pipeline guarantees that DAPI is
    not included implicitly.  Passing the DAPI index explicitly is allowed.
    """
    n_channels = int(n_channels)
    if n_channels < 1:
        raise ValueError("n_channels must be at least 1")

    if channels is None:
        resolved = [
            index
            for index in range(n_channels)
            if nuclear_channel is None or index != int(nuclear_channel)
        ]
    else:
        requested = [channels] if isinstance(channels, (str, int, np.integer)) else list(channels)
        if channel_names is not None:
            resolved = _resolve_named_channels(requested, channel_names)
        else:
            if any(not isinstance(value, (int, np.integer)) for value in requested):
                raise ValueError(
                    "Channel names require channel_names metadata; use integer indices "
                    "for unnamed image arrays"
                )
            resolved = [int(value) for value in requested]

    # Preserve acquisition order while rejecting duplicates and invalid values.
    unique = list(dict.fromkeys(resolved))
    invalid = [index for index in unique if not 0 <= index < n_channels]
    if invalid:
        raise ValueError(
            f"RCA channel index/indices {invalid} out of range for "
            f"n_channels={n_channels}"
        )
    if not unique:
        raise ValueError(
            "No RCA channels remain after excluding the nuclear channel; "
            "provide rca_channels explicitly"
        )
    return unique


def _iter_selected_images(images, channels, channel_axis, channel_names):
    """Yield selected 2D channel arrays without stacking them in memory."""
    if isinstance(images, Mapping):
        keys = list(images)
        if channels is None:
            selected = [
                key
                for key in keys
                if _normalise_name(key) not in _NUCLEAR_CHANNEL_NAMES
            ]
        else:
            requested = [channels] if isinstance(channels, (str, int, np.integer)) else list(channels)
            name_lookup = {_normalise_name(key): key for key in keys}
            selected = []
            for channel in requested:
                if channel in images:
                    selected.append(channel)
                elif _normalise_name(channel) in name_lookup:
                    selected.append(name_lookup[_normalise_name(channel)])
                else:
                    raise ValueError(
                        f"Unknown channel {channel!r}; available channels are {keys}"
                    )
        for key in selected:
            yield np.asarray(images[key])
        return

    if isinstance(images, np.ndarray):
        array = np.asarray(images)
        if array.ndim == 2:
            if channels not in (None, [0], (0,), 0):
                raise ValueError("A 2D image contains only channel 0")
            yield array
            return
        if array.ndim != 3:
            raise ValueError(
                f"images must be 2D or 3D; received shape {array.shape}"
            )
        channel_first = np.moveaxis(array, int(channel_axis), 0)
        if channel_names is not None and len(channel_names) != channel_first.shape[0]:
            raise ValueError("channel_names length does not match the channel axis")
        if channels is None:
            if channel_names is None:
                raise ValueError(
                    "channels must be specified for an unnamed multichannel "
                    "array so a nuclear channel is not included accidentally"
                )
            else:
                selected = [
                    index
                    for index, name in enumerate(channel_names)
                    if _normalise_name(name) not in _NUCLEAR_CHANNEL_NAMES
                ]
        elif channel_names is not None:
            requested = [channels] if isinstance(channels, (str, int, np.integer)) else list(channels)
            selected = _resolve_named_channels(requested, channel_names)
        else:
            requested = [channels] if isinstance(channels, (int, np.integer)) else list(channels)
            if any(not isinstance(value, (int, np.integer)) for value in requested):
                raise ValueError("Named channel selection requires channel_names")
            selected = [int(value) for value in requested]
        for index in selected:
            if not 0 <= int(index) < channel_first.shape[0]:
                raise ValueError(
                    f"Channel index {index} out of range for "
                    f"{channel_first.shape[0]} channels"
                )
            yield channel_first[int(index)]
        return

    if not isinstance(images, Iterable):
        raise TypeError("images must be an array, mapping, or iterable of 2D arrays")

    if channel_names is not None:
        requested = (
            [
                index
                for index, name in enumerate(channel_names)
                if _normalise_name(name) not in _NUCLEAR_CHANNEL_NAMES
            ]
            if channels is None
            else _resolve_named_channels(
                [channels] if isinstance(channels, (str, int, np.integer)) else list(channels),
                channel_names,
            )
        )
        selected = set(requested)
    elif channels is None:
        selected = None
    else:
        requested = [channels] if isinstance(channels, (int, np.integer)) else list(channels)
        if any(not isinstance(value, (int, np.integer)) for value in requested):
            raise ValueError("Named channel selection requires channel_names")
        selected = {int(value) for value in requested}

    for index, image in enumerate(images):
        if selected is None or index in selected:
            yield np.asarray(image)


def _robust_limits(image, percentiles):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 0.0
    low, high = np.percentile(finite, percentiles)
    return float(low), float(high)


def _cast_composite(composite, output_dtype, *, normalized):
    if output_dtype is None:
        return composite.astype(np.float32, copy=False)
    dtype = np.dtype(output_dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if normalized:
            composite = composite * info.max
        composite = np.clip(composite, info.min, info.max)
    return composite.astype(dtype, copy=False)


def generate_rca_composite(
    images,
    channels=None,
    method="normalized_max",
    background_subtraction=True,
    normalization_percentiles=(1, 99.9),
    gaussian_sigma=None,
    output_dtype=None,
    *,
    channel_axis=0,
    channel_names=None,
):
    """Generate one 2D registration reference from RCA fluorescence channels.

    ``images`` may be a channel-first/last 3D array, a mapping from channel
    names to 2D images, or an iterable of 2D images.  Iterables are consumed one
    channel at a time so large microscopy channels need not be stacked.

    Named mappings and arrays with ``channel_names`` exclude common nuclear
    names (including DAPI) when ``channels`` is omitted.  Unnamed multichannel
    arrays require an explicit channel selection so DAPI cannot be included by
    accident. An iterable may contain an already-selected set of 2D channels.
    """
    method = str(method).strip().lower()
    if method not in {"max", "sum", "normalized_max"}:
        raise ValueError(
            "method must be one of 'max', 'sum', or 'normalized_max'"
        )
    try:
        low_percentile, high_percentile = map(float, normalization_percentiles)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalization_percentiles must contain two numbers") from exc
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError(
            "normalization_percentiles must satisfy 0 <= low < high <= 100"
        )

    composite = None
    shape = None
    n_selected = 0
    for image in _iter_selected_images(
        images, channels, channel_axis, channel_names
    ):
        if image.ndim != 2:
            raise ValueError(
                f"Each selected channel must be 2D; received shape {image.shape}"
            )
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(
                f"All channels must have the same shape; expected {shape}, "
                f"received {image.shape}"
            )

        working = np.asarray(image, dtype=np.float32)
        low, high = _robust_limits(
            working, (low_percentile, high_percentile)
        )
        if method == "normalized_max":
            background = low if background_subtraction else 0.0
            scale = high - background
            if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
                processed = np.zeros_like(working, dtype=np.float32)
            else:
                processed = (working - background) / scale
                np.nan_to_num(processed, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
                np.clip(processed, 0.0, 1.0, out=processed)
        else:
            processed = working
            if background_subtraction:
                processed = working - low
                np.maximum(processed, 0.0, out=processed)
            np.nan_to_num(processed, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if composite is None:
            composite = processed.copy()
        elif method == "sum":
            composite += processed
        else:
            np.maximum(composite, processed, out=composite)
        n_selected += 1

    if n_selected == 0 or composite is None:
        raise ValueError("No image channels were selected for the RCA composite")

    if gaussian_sigma is not None:
        sigma = float(gaussian_sigma)
        if sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")
        if sigma > 0:
            from scipy.ndimage import gaussian_filter

            composite = gaussian_filter(composite, sigma=sigma)

    return _cast_composite(
        composite,
        output_dtype,
        normalized=(method == "normalized_max"),
    )


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _ome_images(tiff):
    if not tiff.ome_metadata:
        raise ValueError(f"Input is not an OME-TIFF: {tiff.filehandle.path}")
    ome = tifffile.xml2dict(tiff.ome_metadata).get("OME", {})
    images = _as_list(ome.get("Image"))
    if len(images) != len(tiff.series):
        raise ValueError(
            "OME Image metadata count does not match the number of TIFF series"
        )
    return images


def _channel_names_from_pixels(pixels, n_channels):
    records = _as_list(pixels.get("Channel"))
    names = []
    for index in range(n_channels):
        record = records[index] if index < len(records) else {}
        names.append(record.get("Name", f"Channel {index}"))
    return names


def read_ome_channel_names(path):
    """Return channel names from the first series of an OME-TIFF, if present."""
    with tifffile.TiffFile(path) as tiff:
        images = _ome_images(tiff)
        pixels = images[0].get("Pixels", {})
        n_channels = int(pixels.get("SizeC", 0))
        records = _as_list(pixels.get("Channel"))
        if not records or not any(record.get("Name") for record in records):
            return None
        return _channel_names_from_pixels(pixels, n_channels)


def _series_as_cyx(series):
    array = series.asarray()
    axes = series.axes
    if "C" not in axes or "Y" not in axes or "X" not in axes:
        raise ValueError(
            f"OME series axes must include C, Y, and X; received {axes!r}"
        )

    selection = []
    retained_axes = []
    for axis, size in zip(axes, array.shape):
        if axis in "CYX":
            selection.append(slice(None))
            retained_axes.append(axis)
        elif size == 1:
            selection.append(0)
        else:
            raise ValueError(
                "RCA composite generation expects 2D channel images; "
                f"axis {axis!r} has non-singleton size {size}"
            )
    array = array[tuple(selection)]
    order = [retained_axes.index(axis) for axis in "CYX"]
    return np.transpose(array, order)


def _series_metadata(ome_image, n_channels, composite_name):
    pixels = ome_image.get("Pixels", {})
    metadata = {
        "axes": "CYX",
        "Channel": {
            "Name": _channel_names_from_pixels(pixels, n_channels)
            + [str(composite_name)]
        },
    }

    pixel_metadata = {}
    for key in (
        "PhysicalSizeX",
        "PhysicalSizeXUnit",
        "PhysicalSizeY",
        "PhysicalSizeYUnit",
    ):
        if key in pixels:
            pixel_metadata[key] = pixels[key]
    if pixel_metadata:
        metadata["Pixels"] = pixel_metadata

    planes = _as_list(pixels.get("Plane"))
    if planes:
        by_channel = {}
        for plane in planes:
            by_channel.setdefault(int(plane.get("TheC", 0)), plane)
        fallback = planes[0]
        source_planes = [by_channel.get(index, fallback) for index in range(n_channels)]
        source_planes.append(source_planes[0])
        plane_metadata = {}
        for key in ("PositionX", "PositionXUnit", "PositionY", "PositionYUnit"):
            if all(key in plane for plane in source_planes):
                plane_metadata[key] = [plane[key] for plane in source_planes]
        if plane_metadata:
            metadata["Plane"] = plane_metadata
    return metadata


def create_rca_augmented_ome(
    input_path,
    output_path,
    *,
    rca_channels: Sequence[int],
    method="normalized_max",
    background_subtraction=True,
    normalization_percentiles=(1, 99.9),
    gaussian_sigma=None,
    composite_name="RCA composite",
):
    """Copy an OME-TIFF while appending one RCA-composite channel per tile.

    The current ISS preprocessing format stores each microscope tile as one
    OME Image/series with ``CYX`` data.  This function retains the original
    channels, stage positions, and physical pixel sizes and appends a final
    channel that ASHLAR can use through its standard processing path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    try:
        with tifffile.TiffFile(input_path) as source:
            ome_images = _ome_images(source)
            if not source.series:
                raise ValueError(f"OME-TIFF contains no image series: {input_path}")

            with tifffile.TiffWriter(partial_path, bigtiff=True, ome=True) as target:
                expected_channels = None
                for index, (series, ome_image) in enumerate(zip(source.series, ome_images)):
                    channel_first = _series_as_cyx(series)
                    n_channels = int(channel_first.shape[0])
                    if expected_channels is None:
                        expected_channels = n_channels
                    elif n_channels != expected_channels:
                        raise ValueError(
                            "All OME series must have the same channel count; "
                            f"series 0 has {expected_channels}, series {index} has {n_channels}"
                        )
                    invalid = [
                        channel
                        for channel in rca_channels
                        if not 0 <= int(channel) < n_channels
                    ]
                    if invalid:
                        raise ValueError(
                            f"RCA channels {invalid} are invalid for {n_channels} channels"
                        )

                    composite = generate_rca_composite(
                        channel_first,
                        channels=rca_channels,
                        method=method,
                        background_subtraction=background_subtraction,
                        normalization_percentiles=normalization_percentiles,
                        gaussian_sigma=gaussian_sigma,
                        output_dtype=channel_first.dtype,
                    )
                    augmented = np.concatenate(
                        [channel_first, composite[np.newaxis, ...]], axis=0
                    )
                    metadata = _series_metadata(
                        ome_image, n_channels, composite_name
                    )
                    target.write(
                        augmented,
                        metadata=metadata,
                        photometric="minisblack",
                    )

        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    return output_path


def estimate_residual_translation(reference, moving, upsample_factor=10):
    """Return the subpixel ``(y, x)`` shift needed to align ``moving``."""
    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError("reference and moving must be same-shaped 2D images")
    if int(upsample_factor) < 1:
        raise ValueError("upsample_factor must be at least 1")

    reference = np.nan_to_num(reference - np.nanmean(reference))
    moving = np.nan_to_num(moving - np.nanmean(moving))
    product = np.fft.fftn(reference) * np.fft.fftn(moving).conj()
    magnitude = np.abs(product)
    product /= np.maximum(magnitude, np.finfo(float).eps)
    correlation = np.fft.ifftn(product)
    absolute = np.abs(correlation)
    peak_index = np.unravel_index(np.argmax(absolute), absolute.shape)
    shifts = np.asarray(peak_index, dtype=float)
    for axis, size in enumerate(reference.shape):
        if shifts[axis] > size // 2:
            shifts[axis] -= size

        # Quadratic interpolation around the discrete peak gives a stable
        # subpixel residual without adding another image-registration backend.
        center = list(peak_index)
        before = list(peak_index)
        after = list(peak_index)
        before[axis] = (before[axis] - 1) % size
        after[axis] = (after[axis] + 1) % size
        y0 = absolute[tuple(before)]
        y1 = absolute[tuple(center)]
        y2 = absolute[tuple(after)]
        denominator = y0 - 2 * y1 + y2
        if abs(denominator) > np.finfo(float).eps:
            delta = 0.5 * (y0 - y2) / denominator
            resolution = 1.0 / int(upsample_factor)
            shifts[axis] += np.round(delta / resolution) * resolution

    peak = correlation[peak_index]
    error = max(0.0, 1.0 - float(np.abs(peak)))
    phase = float(np.angle(peak))
    return shifts, error, phase


def normalized_cross_correlation(reference, moving):
    """Calculate zero-mean normalized cross-correlation for two 2D images."""
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError("reference and moving must be same-shaped 2D images")
    valid = np.isfinite(reference) & np.isfinite(moving)
    if not np.any(valid):
        return float("nan")
    left = reference[valid] - np.mean(reference[valid])
    right = moving[valid] - np.mean(moving[valid])
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def registration_overlay(reference, moving):
    """Create an RGB overlay (reference=magenta, moving=green)."""
    def scale(image):
        image = np.asarray(image, dtype=np.float32)
        low, high = _robust_limits(image, (1, 99.9))
        if high <= low:
            return np.zeros_like(image, dtype=np.float32)
        result = (image - low) / (high - low)
        np.clip(result, 0, 1, out=result)
        return result

    ref = scale(reference)
    mov = scale(moving)
    return np.stack([ref, mov, ref], axis=-1)

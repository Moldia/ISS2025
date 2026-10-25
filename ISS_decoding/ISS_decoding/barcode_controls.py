"""Shared negative-control barcode generation for ISS decoders."""

from __future__ import annotations

from math import ceil

import numpy as np

from starfish import Codebook
from starfish.types import Axes, Features


FAKE_BARCODE_PREFIX = "__fake_barcode_"
FAKE_BARCODE_DEFAULTS = {
    "fake_barcode_fraction": 0.0,
    "fake_barcode_seed": 0,
}


def validate_fake_barcode_settings(settings, decoder_name):
    """Validate and normalize shared fake-barcode settings in ``settings``."""
    fraction = settings["fake_barcode_fraction"]
    if isinstance(fraction, (bool, np.bool_)):
        raise ValueError(
            f"{decoder_name} 'fake_barcode_fraction' must be finite and non-negative."
        )
    try:
        fraction = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{decoder_name} 'fake_barcode_fraction' must be finite and non-negative."
        ) from exc
    if not np.isfinite(fraction) or fraction < 0:
        raise ValueError(
            f"{decoder_name} 'fake_barcode_fraction' must be finite and non-negative."
        )

    seed = settings["fake_barcode_seed"]
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or seed < 0
    ):
        raise ValueError(
            f"{decoder_name} 'fake_barcode_seed' must be a non-negative integer."
        )

    settings["fake_barcode_fraction"] = fraction
    settings["fake_barcode_seed"] = int(seed)
    return settings


def _sequence_ids(barcodes, channels):
    sequences = barcodes.argmax(axis=2).astype(np.int64)
    identifiers = np.zeros(len(sequences), dtype=np.int64)
    for round_values in sequences.T:
        identifiers = identifiers * channels + round_values
    return identifiers


def _sample_unused_ids(total_space, used_ids, count, seed):
    available = total_space - len(used_ids)
    if count > available:
        raise ValueError(
            f"Requested {count} fake barcodes, but only {available} unused codewords "
            "exist in the code space."
        )
    if count == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    # Exhaustive enumeration is deterministic and efficient for the small code
    # spaces typical of ISS. Rejection sampling avoids materializing very large
    # spaces when only a small control panel is requested.
    if total_space <= 1_000_000 or available <= max(100_000, count * 10):
        unused = np.fromiter(
            (identifier for identifier in range(total_space) if identifier not in used_ids),
            dtype=np.int64,
            count=available,
        )
        return rng.choice(unused, size=count, replace=False)

    selected = set()
    while len(selected) < count:
        batch_size = max(64, 2 * (count - len(selected)))
        candidates = rng.integers(0, total_space, size=batch_size, dtype=np.int64)
        selected.update(
            int(identifier)
            for identifier in candidates
            if int(identifier) not in used_ids
        )
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    if len(selected_array) > count:
        selected_array = rng.choice(selected_array, size=count, replace=False)
    return selected_array


def augment_one_hot_barcodes(
    barcodes,
    target_names,
    *,
    fraction=0.0,
    seed=0,
    prefix=FAKE_BARCODE_PREFIX,
):
    """Append reproducible unused one-hot codewords as negative controls.

    Parameters
    ----------
    barcodes
        Target x round x channel binary array.
    target_names
        One name per target.
    fraction
        Requested fake controls relative to the number of true targets. The
        number is rounded up.
    seed
        Random seed used to sample unused codewords.
    prefix
        Prefix for generated target names.

    Returns
    -------
    augmented_barcodes, augmented_names, fake_mask
        ``fake_mask`` is aligned to the augmented target axis.
    """
    barcodes = np.asarray(barcodes, dtype=np.float32)
    target_names = np.asarray(target_names).astype(str)
    if barcodes.ndim != 3:
        raise ValueError("Barcodes must have shape (targets, rounds, channels).")
    if target_names.shape != (barcodes.shape[0],):
        raise ValueError("Target names must contain one value per barcode.")

    if isinstance(fraction, (bool, np.bool_)):
        raise ValueError("Fake barcode fraction must be finite and non-negative.")
    try:
        fraction = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Fake barcode fraction must be finite and non-negative."
        ) from exc
    if not np.isfinite(fraction) or fraction < 0:
        raise ValueError("Fake barcode fraction must be finite and non-negative.")
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or seed < 0
    ):
        raise ValueError("Fake barcode seed must be a non-negative integer.")
    seed = int(seed)
    fake_count = int(ceil(len(barcodes) * fraction)) if fraction > 0 else 0
    if fake_count == 0:
        return barcodes, target_names, np.zeros(len(barcodes), dtype=bool)

    if not np.isfinite(barcodes).all() or not np.all(
        np.isclose(barcodes, 0) | np.isclose(barcodes, 1)
    ):
        raise ValueError("Fake barcode generation requires a finite binary codebook.")
    if not np.allclose(barcodes.sum(axis=2), 1):
        raise ValueError(
            "Fake barcode generation requires exactly one active channel for "
            "every target and round."
        )

    rounds, channels = barcodes.shape[1:]
    total_space = channels**rounds
    if total_space > np.iinfo(np.int64).max:
        raise ValueError(
            "The barcode code space is too large for deterministic control generation."
        )
    used_ids = set(_sequence_ids(barcodes, channels).tolist())
    selected_ids = _sample_unused_ids(total_space, used_ids, fake_count, seed)

    fake_barcodes = np.zeros((fake_count, rounds, channels), dtype=np.float32)
    for fake_index, identifier in enumerate(selected_ids):
        value = int(identifier)
        sequence = np.empty(rounds, dtype=int)
        for round_index in range(rounds - 1, -1, -1):
            sequence[round_index] = value % channels
            value //= channels
        fake_barcodes[fake_index, np.arange(rounds), sequence] = 1

    existing_names = set(target_names.tolist())
    fake_names = []
    name_index = 1
    while len(fake_names) < fake_count:
        candidate = f"{prefix}{name_index:04d}"
        if candidate not in existing_names:
            fake_names.append(candidate)
        name_index += 1

    return (
        np.concatenate((barcodes, fake_barcodes), axis=0),
        np.concatenate((target_names, np.asarray(fake_names))),
        np.concatenate(
            (
                np.zeros(len(barcodes), dtype=bool),
                np.ones(fake_count, dtype=bool),
            )
        ),
    )


def augment_spacetx_codebook(codebook, *, fraction=0.0, seed=0):
    """Return an in-memory Starfish codebook augmented with fake controls."""
    expected_dims = (Features.TARGET, Axes.ROUND.value, Axes.CH.value)
    if not all(dim in codebook.dims for dim in expected_dims):
        raise ValueError(
            "The SpaceTx codebook must have target, round, and channel dimensions."
        )
    ordered = codebook.transpose(*expected_dims)
    barcodes, names, fake_mask = augment_one_hot_barcodes(
        ordered.values,
        ordered.coords[Features.TARGET].values,
        fraction=fraction,
        seed=seed,
    )
    augmented = Codebook.from_numpy(
        names,
        n_round=barcodes.shape[1],
        n_channel=barcodes.shape[2],
        data=barcodes.astype(np.uint8),
    )
    return augmented, names, fake_mask


def fake_name_set(target_names, fake_mask):
    """Return generated control names as a set for output annotation."""
    names = np.asarray(target_names).astype(str)
    mask = np.asarray(fake_mask, dtype=bool)
    if mask.shape != names.shape:
        raise ValueError("Fake barcode mask must align with target names.")
    return set(names[mask].tolist())

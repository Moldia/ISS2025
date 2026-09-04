"""Adapters between Starfish/SpaceTx objects and PoSTcode's NumPy inputs."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import (
    transfer_physical_coords_to_intensity_table,
)
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match
from starfish.types import Axes, Features


@dataclass(frozen=True)
class PostcodeInputs:
    """PoSTcode arrays plus the Starfish table that carries spot coordinates."""

    spot_intensities: np.ndarray
    barcodes: np.ndarray
    target_names: np.ndarray
    intensity_table: Any


def format_spacetx_codebook_for_postcode(codebook):
    """Convert a Starfish ``Codebook`` from ``K x R x C`` to ``K x C x R``."""
    expected_dims = (Features.TARGET, Axes.ROUND.value, Axes.CH.value)
    if not all(dim in codebook.dims for dim in expected_dims):
        raise ValueError(
            "The SpaceTx codebook must have target, round, and channel dimensions."
        )

    ordered = codebook.transpose(*expected_dims)
    starfish_barcodes = np.asarray(ordered.values)
    if starfish_barcodes.ndim != 3:
        raise ValueError(
            "The SpaceTx codebook must have shape (barcodes, rounds, channels)."
        )
    if not np.isfinite(starfish_barcodes).all():
        raise ValueError("The SpaceTx codebook contains non-finite values.")
    if not np.all(np.isclose(starfish_barcodes, 0) | np.isclose(starfish_barcodes, 1)):
        raise ValueError("PoSTcode requires a binary one-hot codebook.")
    if not np.allclose(starfish_barcodes.sum(axis=2), 1):
        raise ValueError(
            "PoSTcode requires exactly one active channel for every barcode and round."
        )

    barcodes = np.swapaxes(starfish_barcodes, 1, 2).astype(np.float32, copy=False)
    target_names = np.asarray(ordered.coords[Features.TARGET].values).astype(str)
    return barcodes, target_names


def format_intensity_table_for_postcode(intensity_table):
    """Convert Starfish spot traces from ``N x R x C`` to PoSTcode's ``N x C x R``."""
    expected_dims = (Features.AXIS, Axes.ROUND.value, Axes.CH.value)
    if not all(dim in intensity_table.dims for dim in expected_dims):
        raise ValueError(
            "The Starfish intensity table must have feature, round, and channel dimensions."
        )

    ordered = intensity_table.transpose(*expected_dims)
    starfish_spots = np.asarray(ordered.values, dtype=np.float32)
    if np.isinf(starfish_spots).any():
        raise ValueError("The Starfish intensity table contains infinite values.")
    starfish_spots = np.nan_to_num(starfish_spots, nan=0.0)
    return np.swapaxes(starfish_spots, 1, 2)


def format_starfish_spots_for_postcode(spot_results):
    """Build exact-match Starfish traces and convert them into PoSTcode spot inputs."""
    intensity_table = build_spot_traces_exact_match(spot_results)
    transfer_physical_coords_to_intensity_table(
        intensity_table=intensity_table,
        spots=spot_results,
    )
    return format_intensity_table_for_postcode(intensity_table), intensity_table


def prepare_postcode_inputs(spot_results, codebook):
    """Prepare the image-derived traces and SpaceTx codebook required by PoSTcode."""
    spot_intensities, intensity_table = format_starfish_spots_for_postcode(spot_results)
    barcodes, target_names = format_spacetx_codebook_for_postcode(codebook)
    if spot_intensities.shape[1:] != barcodes.shape[1:]:
        raise ValueError(
            "Spot traces and codebook have different channel/round dimensions: "
            f"{spot_intensities.shape[1:]} != {barcodes.shape[1:]}."
        )
    return PostcodeInputs(
        spot_intensities=spot_intensities,
        barcodes=barcodes,
        target_names=target_names,
        intensity_table=intensity_table,
    )


def summarize_postcode_output(
    output,
    target_names,
    probability_threshold=None,
):
    """Return reversible per-spot assignments from PoSTcode posterior probabilities.

    ``target`` is populated only when the winning class is a gene and it passes
    ``probability_threshold``. ``candidate_target`` always records the most likely
    gene, even when background or the aggregate infeasible class wins.
    """
    class_probs = np.asarray(output["class_probs"], dtype=float)
    if class_probs.ndim != 2:
        raise ValueError("PoSTcode class probabilities must be a two-dimensional array.")
    if not np.isfinite(class_probs).all():
        raise ValueError("PoSTcode class probabilities contain non-finite values.")
    if probability_threshold is not None and not 0 <= probability_threshold <= 1:
        raise ValueError("probability_threshold must be between 0 and 1.")

    target_names = np.asarray(target_names).astype(str)
    gene_indices = np.atleast_1d(output["class_ind"]["genes"]).astype(int)
    if target_names.shape[0] != gene_indices.shape[0]:
        raise ValueError("The number of target names does not match the decoded genes.")
    if gene_indices.size == 0:
        raise ValueError("PoSTcode output does not contain any gene classes.")

    class_count = class_probs.shape[1]

    def normalized_indices(class_name):
        indices = np.atleast_1d(output["class_ind"][class_name]).astype(int)
        if np.any((indices < 0) | (indices >= class_count)):
            raise ValueError(f"PoSTcode {class_name!r} class index is out of bounds.")
        return indices

    if np.any((gene_indices < 0) | (gene_indices >= class_count)):
        raise ValueError("A PoSTcode gene class index is out of bounds.")
    if np.unique(gene_indices).size != gene_indices.size:
        raise ValueError("PoSTcode gene class indices must be unique.")

    background_indices = normalized_indices("bkg")
    infeasible_indices = normalized_indices("inf")
    nan_indices = normalized_indices("nan")

    gene_probs = class_probs[:, gene_indices]
    gene_order = np.argsort(-gene_probs, axis=1, kind="stable")
    row_indices = np.arange(class_probs.shape[0])
    best_gene_positions = gene_order[:, 0]
    best_gene_probability = gene_probs[row_indices, best_gene_positions]
    candidate_target = target_names[best_gene_positions]

    if gene_indices.size > 1:
        second_gene_positions = gene_order[:, 1]
        second_gene = target_names[second_gene_positions].astype(object)
        second_gene_probability = gene_probs[row_indices, second_gene_positions]
        gene_probability_margin = best_gene_probability - second_gene_probability
    else:
        second_gene = np.full(class_probs.shape[0], None, dtype=object)
        second_gene_probability = np.full(class_probs.shape[0], np.nan)
        gene_probability_margin = np.full(class_probs.shape[0], np.nan)

    winning_indices = class_probs.argmax(axis=1)
    assignment_probability = class_probs[row_indices, winning_indices]
    assignment_class = np.full(class_probs.shape[0], "unknown", dtype=object)
    gene_winner = np.isin(winning_indices, gene_indices)
    assignment_class[gene_winner] = "gene"
    assignment_class[np.isin(winning_indices, background_indices)] = "background"
    assignment_class[np.isin(winning_indices, infeasible_indices)] = "infeasible"
    assignment_class[np.isin(winning_indices, nan_indices)] = "nan"
    if np.any(assignment_class == "unknown"):
        raise ValueError("PoSTcode output contains a class without a declared class type.")

    passes_thresholds = gene_winner.copy()
    if probability_threshold is not None:
        passes_thresholds &= assignment_probability >= probability_threshold

    accepted_target = np.full(class_probs.shape[0], None, dtype=object)
    accepted_target[passes_thresholds] = candidate_target[passes_thresholds]

    def aggregate_probability(indices):
        if indices.size == 0:
            return np.zeros(class_probs.shape[0], dtype=float)
        return class_probs[:, indices].sum(axis=1)

    return pd.DataFrame(
        {
            "target": accepted_target,
            "candidate_target": candidate_target,
            "assignment_class": assignment_class.astype(str),
            "passes_thresholds": passes_thresholds,
            "assignment_probability": assignment_probability,
            "best_gene_probability": best_gene_probability,
            "background_probability": aggregate_probability(background_indices),
            "infeasible_probability": aggregate_probability(infeasible_indices),
            "nan_probability": aggregate_probability(nan_indices),
            "second_gene": second_gene,
            "second_gene_probability": second_gene_probability,
            "gene_probability_margin": gene_probability_margin,
        }
    )


def postcode_output_to_decoded_table(
    output,
    intensity_table,
    target_names,
    probability_threshold=None,
):
    """Attach PoSTcode assignments to a Starfish ``DecodedIntensityTable``."""
    class_probs = np.asarray(output["class_probs"])
    if class_probs.ndim != 2 or class_probs.shape[0] != intensity_table.shape[0]:
        raise ValueError(
            "PoSTcode class probabilities must have one row per Starfish spot."
        )
    target_names = np.asarray(target_names).astype(str)
    summary = summarize_postcode_output(
        output,
        target_names,
        probability_threshold=probability_threshold,
    )
    probabilities = summary["assignment_probability"].to_numpy()
    class_names = summary["assignment_class"].to_numpy()
    passes_threshold = summary["passes_thresholds"].to_numpy()
    targets = summary["target"].fillna("nan").to_numpy(dtype=str)

    decoded = DecodedIntensityTable.from_intensity_table(
        intensity_table,
        targets=(Features.AXIS, targets.astype(str)),
        distances=(Features.AXIS, 1 - probabilities),
        passes_threshold=(Features.AXIS, passes_threshold),
    )
    return decoded, probabilities, class_names.astype(str)

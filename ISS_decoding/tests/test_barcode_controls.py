import numpy as np
import pytest
from starfish import Codebook

from ISS_decoding.barcode_controls import (
    augment_one_hot_barcodes,
    augment_spacetx_codebook,
    validate_fake_barcode_settings,
)


def test_shared_controls_are_reproducible_unique_and_rounded_up():
    barcodes = np.array(
        [
            [[1, 0], [1, 0], [1, 0]],
            [[0, 1], [0, 1], [0, 1]],
            [[1, 0], [0, 1], [1, 0]],
        ],
        dtype=np.float32,
    )
    names = np.array(["a", "b", "c"])

    first = augment_one_hot_barcodes(barcodes, names, fraction=0.1, seed=11)
    second = augment_one_hot_barcodes(barcodes, names, fraction=0.1, seed=11)

    augmented, augmented_names, fake_mask = first
    assert augmented.shape == (4, 3, 2)
    assert fake_mask.tolist() == [False, False, False, True]
    assert augmented_names[-1] == "__fake_barcode_0001"
    assert len(np.unique(augmented.reshape(4, -1), axis=0)) == 4
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_shared_controls_reject_exhausted_code_space():
    sequences = np.array(
        [
            [[1, 0], [1, 0]],
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, 1], [0, 1]],
        ],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="only 0 unused codewords"):
        augment_one_hot_barcodes(
            sequences,
            np.array(["a", "b", "c", "d"]),
            fraction=0.1,
            seed=0,
        )


def test_spacetx_augmentation_does_not_mutate_original_codebook():
    original = Codebook.from_numpy(
        ["gene_a"],
        n_round=2,
        n_channel=2,
        data=np.array([[[1, 0], [1, 0]]], dtype=np.uint8),
    )
    original_values = original.values.copy()

    augmented, names, fake_mask = augment_spacetx_codebook(
        original, fraction=1.0, seed=4
    )

    np.testing.assert_array_equal(original.values, original_values)
    assert augmented.shape == (2, 2, 2)
    assert names.tolist() == ["gene_a", "__fake_barcode_0001"]
    assert fake_mask.tolist() == [False, True]


@pytest.mark.parametrize(
    "settings",
    [
        {"fake_barcode_fraction": -0.1, "fake_barcode_seed": 0},
        {"fake_barcode_fraction": np.nan, "fake_barcode_seed": 0},
        {"fake_barcode_fraction": 0.1, "fake_barcode_seed": -1},
        {"fake_barcode_fraction": 0.1, "fake_barcode_seed": 1.5},
    ],
)
def test_shared_control_settings_are_validated(settings):
    with pytest.raises(ValueError, match="fake_barcode"):
        validate_fake_barcode_settings(settings, "Decoder")

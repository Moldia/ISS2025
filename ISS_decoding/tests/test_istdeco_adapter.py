import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from starfish.types import Axes, Coordinates, Features

from ISS_decoding import istdeco_adapter


def make_codebook():
    return xr.DataArray(
        np.array(
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ],
            dtype=np.float32,
        ),
        dims=(Features.TARGET, Axes.ROUND.value, Axes.CH.value),
        coords={Features.TARGET: ["gene_a", "gene_b"]},
    )


class FakeImageStack:
    def __init__(self, values):
        rounds, channels, zplanes, height, width = values.shape
        self.xarray = xr.DataArray(
            values,
            dims=(
                Axes.ROUND.value,
                Axes.CH.value,
                Axes.ZPLANE.value,
                Axes.Y.value,
                Axes.X.value,
            ),
            coords={
                Coordinates.X.value: (Axes.X.value, np.arange(width) * 0.5 + 10),
                Coordinates.Y.value: (Axes.Y.value, np.arange(height) * 0.5 + 20),
                Coordinates.Z.value: (Axes.ZPLANE.value, np.arange(zplanes)),
            },
        )


def test_spacetx_adapters_preserve_istdeco_axis_order():
    codebook, names = istdeco_adapter.format_spacetx_codebook_for_istdeco(
        make_codebook()
    )
    assert codebook.shape == (2, 2, 2)
    assert names.tolist() == ["gene_a", "gene_b"]
    np.testing.assert_array_equal(codebook[0], [[1, 0], [0, 1]])

    image = np.arange(2 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 2, 3, 4)
    formatted = istdeco_adapter.format_spacetx_image_for_istdeco(
        FakeImageStack(image), z_projection="max"
    )
    np.testing.assert_array_equal(formatted, image.max(axis=2))


def test_istdeco_settings_are_validated_and_overlap_is_automatic():
    settings = istdeco_adapter.effective_istdeco_kwargs(
        {
            "sigma": 2,
            "tile_size": 64,
            "device": "CPU",
            "fake_barcode_fraction": 0.1,
            "fake_barcode_seed": 17,
        }
    )
    assert settings["sigma"] == (2.0, 2.0)
    assert settings["tile_size"] == (64, 64)
    assert settings["overlap"] == 6
    assert settings["device"] == "cpu"
    assert settings["fake_barcode_fraction"] == 0.1
    assert settings["fake_barcode_seed"] == 17


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unexpected": 1}, "Unknown ISTDECO"),
        ({"sigma": -1}, "sigma"),
        ({"tile_size": 8, "overlap": 4}, "greater than twice"),
        ({"intensity_percentile": 101}, "between 0 and 100"),
        ({"quality_threshold": -0.1}, "quality_threshold"),
        ({"z_projection": "sum"}, "z_projection"),
        ({"fake_barcode_fraction": -0.1}, "fake_barcode_fraction"),
        ({"fake_barcode_fraction": np.nan}, "fake_barcode_fraction"),
        ({"fake_barcode_seed": -1}, "fake_barcode_seed"),
        ({"fake_barcode_seed": 1.5}, "fake_barcode_seed"),
    ],
)
def test_invalid_istdeco_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        istdeco_adapter.effective_istdeco_kwargs(overrides)


def test_fake_barcodes_are_reproducible_unique_and_not_true_codes():
    true_barcodes, true_names = istdeco_adapter.format_spacetx_codebook_for_istdeco(
        make_codebook()
    )

    augmented, names, fake_mask = (
        istdeco_adapter.augment_istdeco_codebook_with_fake_barcodes(
            true_barcodes,
            true_names,
            fraction=0.5,
            seed=7,
        )
    )
    repeated, repeated_names, repeated_mask = (
        istdeco_adapter.augment_istdeco_codebook_with_fake_barcodes(
            true_barcodes,
            true_names,
            fraction=0.5,
            seed=7,
        )
    )

    assert augmented.shape == (3, 2, 2)
    assert fake_mask.tolist() == [False, False, True]
    assert names[:2].tolist() == ["gene_a", "gene_b"]
    assert names[2].startswith(istdeco_adapter.ISTDECO_FAKE_BARCODE_PREFIX)
    flattened = augmented.reshape(len(augmented), -1)
    assert len(np.unique(flattened, axis=0)) == len(augmented)
    np.testing.assert_array_equal(repeated, augmented)
    np.testing.assert_array_equal(repeated_names, names)
    np.testing.assert_array_equal(repeated_mask, fake_mask)


def test_fake_barcode_generation_rejects_requests_larger_than_unused_space():
    true_barcodes, true_names = istdeco_adapter.format_spacetx_codebook_for_istdeco(
        make_codebook()
    )

    with pytest.raises(ValueError, match="only 2 unused codewords"):
        istdeco_adapter.augment_istdeco_codebook_with_fake_barcodes(
            true_barcodes,
            true_names,
            fraction=1.5,
            seed=0,
        )


def test_tiling_crops_halos_and_avoids_duplicate_seam_spots(monkeypatch):
    class FakeISTDeco:
        def __init__(self, image, codebook, **_kwargs):
            self.shape = image.shape
            self.code_count = codebook.shape[0]
            self.image = image

        def to(self, _device):
            return self

        def run(self, **_kwargs):
            output_shape = (self.code_count, *self.shape[-2:])
            intensity = np.zeros(output_shape, dtype=np.float32)
            quality = np.zeros(output_shape, dtype=np.float32)
            detected_y, detected_x = np.where(self.image[0, 0] == 10)
            intensity[0, detected_y, detected_x] = 10
            quality[0, detected_y, detected_x] = 0.9
            return intensity, quality, np.array([1.0])

    monkeypatch.setattr(istdeco_adapter, "_load_istdeco_class", lambda: FakeISTDeco)
    monkeypatch.setattr(istdeco_adapter, "_resolve_device", lambda _device: "cpu")
    images = np.ones((2, 2, 12, 12), dtype=np.float32)
    images[0, 0, 4, 4] = 10
    result = istdeco_adapter.decode_istdeco_array(
        images,
        make_codebook().values,
        np.array(["gene_a", "gene_b"]),
        settings={
            "tile_size": (8, 8),
            "overlap": 2,
            "intensity_threshold": 5,
            "quality_threshold": 0.5,
            "niter": 1,
        },
    )

    assert len(result) == 1
    assert result.loc[0, "target"] == "gene_a"
    assert result.loc[0, "istdeco_tile"] == "1_1"
    assert result.loc[0, "x"] == 4
    assert result.loc[0, "y"] == 4
    assert result.attrs["istdeco_intensity_threshold"] == 5


def test_fake_barcode_calls_are_marked_as_negative_controls(monkeypatch):
    class FakeISTDeco:
        def __init__(self, image, codebook, **_kwargs):
            self.shape = image.shape
            self.code_count = codebook.shape[0]

        def to(self, _device):
            return self

        def run(self, **_kwargs):
            output_shape = (self.code_count, *self.shape[-2:])
            intensity = np.zeros(output_shape, dtype=np.float32)
            quality = np.zeros(output_shape, dtype=np.float32)
            intensity[-1, 2, 3] = 10
            quality[-1, 2, 3] = 0.9
            return intensity, quality, np.array([1.0])

    monkeypatch.setattr(istdeco_adapter, "_load_istdeco_class", lambda: FakeISTDeco)
    monkeypatch.setattr(istdeco_adapter, "_resolve_device", lambda _device: "cpu")
    result = istdeco_adapter.decode_istdeco_array(
        np.ones((2, 2, 6, 7), dtype=np.float32),
        make_codebook().values,
        np.array(["gene_a", "gene_b"]),
        settings={
            "tile_size": (8, 8),
            "overlap": 0,
            "intensity_threshold": 5,
            "quality_threshold": 0.5,
            "niter": 1,
            "fake_barcode_fraction": 0.5,
            "fake_barcode_seed": 7,
        },
    )

    assert len(result) == 1
    assert result.loc[0, "target"].startswith(
        istdeco_adapter.ISTDECO_FAKE_BARCODE_PREFIX
    )
    assert result.loc[0, "assignment_class"] == "fake_barcode"
    assert bool(result.loc[0, "istdeco_is_fake_barcode"])
    assert result.attrs["istdeco_true_barcode_count"] == 2
    assert result.attrs["istdeco_fake_barcode_count"] == 1


def test_imagestack_decoder_adds_physical_coordinates(monkeypatch):
    monkeypatch.setattr(
        istdeco_adapter,
        "decode_istdeco_array",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                Features.SPOT_ID: [0],
                Axes.X.value: [2.0],
                Axes.Y.value: [3.0],
                Axes.ZPLANE.value: [0],
                "target": ["gene_a"],
                "xc": [np.nan],
                "yc": [np.nan],
                "zc": [np.nan],
            }
        ),
    )
    stack = FakeImageStack(np.ones((2, 2, 1, 6, 7), dtype=np.float32))
    result = istdeco_adapter.decode_imagestack_with_istdeco(stack, make_codebook())
    assert result.loc[0, "xc"] == 11
    assert result.loc[0, "yc"] == 21.5
    assert result.loc[0, "zc"] == 0

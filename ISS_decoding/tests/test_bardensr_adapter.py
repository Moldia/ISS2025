from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from starfish.types import Axes, Coordinates, Features

from ISS_decoding import bardensr_adapter


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


def test_spacetx_adapters_preserve_bardensr_frame_order():
    codebook, names = bardensr_adapter.format_spacetx_codebook_for_bardensr(
        make_codebook()
    )
    assert codebook.shape == (4, 2)
    assert names.tolist() == ["gene_a", "gene_b"]
    np.testing.assert_array_equal(
        codebook,
        [[1, 0], [0, 1], [0, 1], [1, 0]],
    )

    image = np.arange(2 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 2, 3, 4)
    formatted = bardensr_adapter.format_spacetx_image_for_bardensr(
        FakeImageStack(image), z_projection="max"
    )
    assert formatted.shape == (4, 1, 3, 4)
    np.testing.assert_array_equal(formatted[:, 0], image.max(axis=2).reshape(4, 3, 4))


def test_bardensr_method_defaults_and_automatic_overlap():
    singleshot = bardensr_adapter.effective_bardensr_kwargs({"tile_size": 64})
    assert singleshot["method"] == "singleshot"
    assert singleshot["peak_threshold"] == 0.72
    assert singleshot["peak_threshold_fraction"] is None
    assert singleshot["overlap"] == 1

    iterative = bardensr_adapter.effective_bardensr_kwargs(
        {"method": "iterative", "psf_radius": (0, 2, 2), "tile_size": 64}
    )
    assert iterative["peak_threshold"] is None
    assert iterative["peak_threshold_fraction"] == 0.1
    assert iterative["overlap"] == 6


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unexpected": 1}, "Unknown Bardensr"),
        ({"method": "other"}, "method"),
        ({"peak_threshold": 0.5, "peak_threshold_fraction": 0.1}, "exactly one"),
        ({"noisefloor": 0}, "noisefloor"),
        ({"tile_size": 8, "overlap": 4}, "greater than twice"),
        ({"poolsize": (1, -1, 1)}, "poolsize"),
        ({"estimate_phasing": True}, "not implemented"),
        ({"z_projection": "sum"}, "z_projection"),
    ],
)
def test_invalid_bardensr_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        bardensr_adapter.effective_bardensr_kwargs(overrides)


def test_tiling_crops_halos_and_avoids_duplicate_seam_spots(monkeypatch):
    class FakeSpotCalling:
        @staticmethod
        def estimate_density_singleshot(image, codebook, noisefloor):
            del noisefloor
            density = np.zeros((*image.shape[1:], codebook.shape[1]), dtype=np.float32)
            z, y, x = np.where(image[0] == 1)
            density[z, y, x, 0] = 0.9
            return density

        @staticmethod
        def find_peaks(density, thresh, poolsize):
            del poolsize
            z, y, x, target = np.where(density > thresh)
            return pd.DataFrame({"m0": z, "m1": y, "m2": x, "j": target})

    fake_bardensr = SimpleNamespace(spot_calling=FakeSpotCalling())
    fake_tf = SimpleNamespace(
        config=SimpleNamespace(list_physical_devices=lambda _kind: []),
        device=lambda _device: nullcontext(),
    )
    monkeypatch.setattr(
        bardensr_adapter,
        "_load_bardensr_runtime",
        lambda: (fake_bardensr, fake_tf),
    )

    images = np.zeros((4, 1, 12, 12), dtype=np.float32)
    images[0, 0, 4, 4] = 10
    result = bardensr_adapter.decode_bardensr_array(
        images,
        np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        np.array(["gene_a", "gene_b"]),
        rounds=2,
        settings={
            "tile_size": (8, 8),
            "overlap": 2,
            "peak_threshold": 0.5,
        },
    )

    assert len(result) == 1
    assert result.loc[0, "target"] == "gene_a"
    assert result.loc[0, "bardensr_tile"] == "1_1"
    assert result.loc[0, "bardensr_evidence"] == pytest.approx(0.9)
    assert result.loc[0, "x"] == 4
    assert result.loc[0, "y"] == 4


def test_imagestack_decoder_adds_physical_coordinates(monkeypatch):
    monkeypatch.setattr(
        bardensr_adapter,
        "decode_bardensr_array",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                Features.SPOT_ID: [0],
                Axes.X.value: [2.0],
                Axes.Y.value: [3.0],
                Axes.ZPLANE.value: [0],
                Features.TARGET: ["gene_a"],
                "xc": [np.nan],
                "yc": [np.nan],
                "zc": [np.nan],
            }
        ),
    )
    stack = FakeImageStack(np.ones((2, 2, 1, 6, 7), dtype=np.float32))
    result = bardensr_adapter.decode_imagestack_with_bardensr(
        stack, make_codebook()
    )
    assert result.loc[0, "xc"] == 11
    assert result.loc[0, "yc"] == 21.5
    assert result.loc[0, "zc"] == 0

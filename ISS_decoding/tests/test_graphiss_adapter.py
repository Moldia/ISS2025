import numpy as np
import pandas as pd
import pytest
import xarray as xr
from starfish.types import Axes, Coordinates, Features

from ISS_decoding import graphiss_adapter


def make_codebook():
    return xr.DataArray(
        np.array(
            [
                [[1, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [0, 1]],
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


def test_spacetx_adapters_preserve_graphiss_axis_order():
    codebook, names = graphiss_adapter.format_spacetx_codebook_for_graphiss(
        make_codebook()
    )
    assert codebook.shape == (2, 3, 2)
    assert names.tolist() == ["gene_a", "gene_b"]

    image = np.arange(3 * 2 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 2, 4, 5)
    formatted = graphiss_adapter.format_spacetx_image_for_graphiss(
        FakeImageStack(image), z_projection="max"
    )
    assert formatted.shape == (3, 2, 4, 5)
    np.testing.assert_array_equal(formatted, image.max(axis=2))


def test_graphiss_defaults_match_published_workflow():
    settings = graphiss_adapter.effective_graphiss_kwargs()
    assert settings["h"] == 0.05
    assert settings["radius"] == 3
    assert settings["graph_radius"] == 3
    assert settings["transition_radius"] == 4
    assert settings["spatial_decay"] == 0.33
    assert settings["search_mode"] == "prior"
    assert settings["verbose"] is True
    assert settings["max_candidates"] is None
    assert settings["max_component_size"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unexpected": 1}, "Unknown Graph-ISS"),
        ({"h": 0}, "'h'"),
        ({"radius": 0}, "radius"),
        ({"candidate_probability_threshold": 1.1}, "candidate_probability"),
        ({"search_mode": "other"}, "search_mode"),
        ({"normalize_frames": "yes"}, "normalize_frames"),
        ({"z_projection": "sum"}, "z_projection"),
        ({"verbose": "yes"}, "verbose"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_component_size": 1.5}, "max_component_size"),
    ],
)
def test_invalid_graphiss_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        graphiss_adapter.effective_graphiss_kwargs(overrides)


def test_graphiss_output_is_standardized_and_thresholds_are_retained(monkeypatch):
    captured = {}

    def fake_decode(images, codebook, target_names, **kwargs):
        captured.update(
            images=images, codebook=codebook, target_names=target_names, kwargs=kwargs
        )
        result = pd.DataFrame(
            {
                "spot_id": [0, 1],
                "x": [2.0, 4.0],
                "y": [3.0, 5.0],
                "z": [0.0, 0.0],
                "sequence": ["0-1-0", "1-1-1"],
                "target_id": [0, -1],
                "target": ["gene_a", None],
                "ambiguous_target": [False, False],
                "signal_probability_sum": [2.7, 2.4],
                "signal_probability_mean": [0.9, 0.8],
                "signal_probability_min": [0.85, 0.75],
                "max_distance": [1.0, 2.0],
                "transition_probability_product": [0.8, 0.6],
                "spatial_quality": [0.8, 0.6],
                "quality": [2.16, 1.44],
                "path_candidate_indices": ["[0, 1, 2]", "[3, 4, 5]"],
            }
        )
        result.attrs["diagnostics"] = {"candidate_count": 6}
        return result

    monkeypatch.setattr(graphiss_adapter, "_load_graphiss_decoder", lambda: fake_decode)
    images = np.zeros((3, 2, 8, 8), dtype=np.float32)
    images[:, :, 3, 3] = 1
    result = graphiss_adapter.decode_graphiss_array(
        images,
        make_codebook().values,
        np.array(["gene_a", "gene_b"]),
        settings={"search_mode": "blind", "min_signal_probability": 0.8},
    )

    assert captured["images"].shape == (3, 2, 8, 8)
    assert captured["kwargs"]["search_mode"] == "blind"
    assert captured["kwargs"]["verbose"] is True
    assert captured["kwargs"]["max_candidates"] is None
    assert captured["kwargs"]["max_component_size"] is None
    assert result[Features.TARGET].tolist()[0] == "gene_a"
    assert pd.isna(result[Features.TARGET].tolist()[1])
    assert result["candidate_target"].tolist() == ["gene_a", "sequence:1-1-1"]
    assert result["assignment_class"].tolist() == ["gene", "unexpected_sequence"]
    assert result[Features.PASSES_THRESHOLDS].tolist() == [True, False]
    assert result["decoder"].unique().tolist() == ["graphiss"]
    assert result.attrs["graphiss_diagnostics"] == {"candidate_count": 6}


def test_imagestack_decoder_adds_physical_coordinates(monkeypatch):
    monkeypatch.setattr(
        graphiss_adapter,
        "decode_graphiss_array",
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
    stack = FakeImageStack(np.ones((3, 2, 1, 6, 7), dtype=np.float32))
    result = graphiss_adapter.decode_imagestack_with_graphiss(stack, make_codebook())
    assert result.loc[0, "xc"] == 11
    assert result.loc[0, "yc"] == 21.5
    assert result.loc[0, "zc"] == 0

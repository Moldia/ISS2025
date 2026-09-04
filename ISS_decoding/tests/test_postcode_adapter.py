import json

import numpy as np
import pandas as pd
import pytest

from starfish import Codebook, ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import PerImageSliceSpotResults, SpotAttributes, SpotFindingResults
from starfish.core.util.logging import Log
from starfish.types import Axes, Features

from ISS_decoding import decoding
from ISS_decoding.postcode_adapter import (
    PostcodeInputs,
    format_intensity_table_for_postcode,
    format_spacetx_codebook_for_postcode,
    format_starfish_spots_for_postcode,
    postcode_output_to_decoded_table,
    summarize_postcode_output,
)


def make_codebook():
    code_array = [
        {
            Features.TARGET: "gene_a",
            Features.CODEWORD: [
                {Axes.ROUND: 0, Axes.CH: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND: 1, Axes.CH: 1, Features.CODE_VALUE: 1},
            ],
        },
        {
            Features.TARGET: "gene_b",
            Features.CODEWORD: [
                {Axes.ROUND: 0, Axes.CH: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND: 1, Axes.CH: 0, Features.CODE_VALUE: 1},
            ],
        },
    ]
    return Codebook.from_code_array(code_array)


def make_intensity_table(values):
    spot_count, rounds, channels = values.shape
    attributes = SpotAttributes(
        pd.DataFrame(
            {
                Axes.X.value: np.arange(spot_count) + 10,
                Axes.Y.value: np.arange(spot_count) + 20,
                Axes.ZPLANE.value: np.zeros(spot_count, dtype=int),
                Features.SPOT_RADIUS: np.full(spot_count, 2),
                Features.SPOT_ID: np.arange(spot_count),
            }
        )
    )
    table = IntensityTable.zeros(
        spot_attributes=attributes,
        round_labels=np.arange(rounds),
        ch_labels=np.arange(channels),
    )
    table.values[:] = values
    return table


def test_spacetx_codebook_is_transposed_for_postcode():
    barcodes, target_names = format_spacetx_codebook_for_postcode(make_codebook())

    assert barcodes.shape == (2, 2, 2)
    assert target_names.tolist() == ["gene_a", "gene_b"]
    np.testing.assert_array_equal(barcodes[0], [[1, 0], [0, 1]])


def test_intensity_table_is_transposed_for_postcode():
    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    table = make_intensity_table(values)

    spots = format_intensity_table_for_postcode(table)

    assert spots.shape == (3, 4, 2)
    np.testing.assert_array_equal(spots, np.swapaxes(values, 1, 2))


def test_starfish_spot_results_retain_pixel_and_physical_coordinates():
    stack = ImageStack.from_numpy(np.zeros((2, 2, 1, 5, 6), dtype=np.float32))
    result_slices = []
    for round_index in range(2):
        for channel_index in range(2):
            attributes = SpotAttributes(
                pd.DataFrame(
                    {
                        Axes.X.value: [1, 4],
                        Axes.Y.value: [2, 3],
                        Axes.ZPLANE.value: [0, 0],
                        Features.SPOT_RADIUS: [1.0, 1.0],
                        Features.SPOT_ID: [0, 1],
                        Features.INTENSITY: [
                            10 * round_index + 2 * channel_index + 1,
                            10 * round_index + 2 * channel_index + 2,
                        ],
                    }
                )
            )
            result_slices.append(
                (
                    PerImageSliceSpotResults(attributes, extras=None),
                    {Axes.ROUND: round_index, Axes.CH: channel_index},
                )
            )
    results = SpotFindingResults(stack.xarray.coords, Log(), result_slices)

    spots, table = format_starfish_spots_for_postcode(results)
    metadata = table.to_features_dataframe()

    np.testing.assert_array_equal(spots[0], [[1, 11], [3, 13]])
    assert metadata[Axes.X.value].tolist() == [1, 4]
    assert {"xc", "yc", "zc"} <= set(metadata)


def test_non_one_hot_codebook_is_rejected():
    codebook = make_codebook().copy()
    codebook.values[0, 0, :] = 1

    with pytest.raises(ValueError, match="exactly one active channel"):
        format_spacetx_codebook_for_postcode(codebook)


def test_postcode_output_retains_starfish_coordinates_and_threshold():
    table = make_intensity_table(np.ones((3, 2, 2), dtype=np.float32))
    output = {
        "class_probs": np.array(
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.1, 0.2, 0.65, 0.05],
                [0.2, 0.3, 0.1, 0.4],
            ]
        ),
        "class_ind": {
            "genes": np.array([0, 1]),
            "bkg": 2,
            "inf": 3,
            "nan": np.empty(0, dtype=int),
        },
    }

    decoded, probabilities, classes = postcode_output_to_decoded_table(
        output,
        table,
        np.array(["gene_a", "gene_b"]),
        probability_threshold=0.7,
    )
    dataframe = decoded.to_features_dataframe()
    summary = summarize_postcode_output(
        output,
        np.array(["gene_a", "gene_b"]),
        probability_threshold=0.7,
    )

    assert dataframe[Features.TARGET].tolist() == ["gene_a", "nan", "nan"]
    assert dataframe[Features.PASSES_THRESHOLDS].tolist() == [True, False, False]
    assert dataframe[Axes.X.value].tolist() == [10, 11, 12]
    np.testing.assert_allclose(probabilities, [0.8, 0.65, 0.4])
    assert classes.tolist() == ["gene", "background", "infeasible"]
    assert summary["target"].tolist() == ["gene_a", None, None]
    assert summary["candidate_target"].tolist() == ["gene_a", "gene_b", "gene_b"]
    assert summary["assignment_class"].tolist() == [
        "gene",
        "background",
        "infeasible",
    ]
    np.testing.assert_allclose(summary["best_gene_probability"], [0.8, 0.2, 0.3])
    np.testing.assert_allclose(summary["background_probability"], [0.05, 0.65, 0.1])
    np.testing.assert_allclose(summary["infeasible_probability"], [0.05, 0.05, 0.4])
    assert summary["second_gene"].tolist() == ["gene_b", "gene_a", "gene_a"]
    np.testing.assert_allclose(summary["gene_probability_margin"], [0.7, 0.1, 0.1])


def test_decode_spots_with_postcode_runs_the_pinned_decoder(monkeypatch):
    rng = np.random.default_rng(5)
    barcodes = np.array(
        [
            [[1, 1], [0, 0]],
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.float32,
    )
    spot_values = np.repeat(barcodes, 6, axis=0) * 80 + 10
    spot_values += rng.normal(0, 2, size=spot_values.shape).astype(np.float32)
    intensity_table = make_intensity_table(np.swapaxes(spot_values, 1, 2))
    prepared = PostcodeInputs(
        spot_intensities=spot_values,
        barcodes=barcodes,
        target_names=np.array(["a", "b", "c", "d"]),
        intensity_table=intensity_table,
    )
    monkeypatch.setattr(decoding, "prepare_postcode_inputs", lambda *_: prepared)

    dataframe, raw = decoding.decode_spots_with_postcode(
        object(),
        object(),
        prob_threshold=0.5,
        postcode_kwargs={
            "num_iter": 3,
            "up_prc_to_remove": 100,
            "print_training_progress": False,
            "device": "cpu",
            "set_seed": 3,
        },
        return_raw=True,
    )

    assert len(dataframe) == 24
    assert {
        "target",
        "candidate_target",
        "assignment_class",
        "assignment_probability",
        "best_gene_probability",
        "background_probability",
        "infeasible_probability",
        "second_gene",
        "second_gene_probability",
        "gene_probability_margin",
        "postcode_probability",
        "postcode_class",
        "decoder",
    } <= set(dataframe)
    assert np.isfinite(dataframe["postcode_probability"]).all()
    np.testing.assert_allclose(raw["class_probs"].sum(axis=1), 1, atol=1e-6)
    assert raw["target_names"].tolist() == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(raw["barcodes"], barcodes)


def test_decode_spots_with_postcode_marks_empty_tile_complete(monkeypatch):
    prepared = PostcodeInputs(
        spot_intensities=np.empty((0, 2, 2), dtype=np.float32),
        barcodes=np.ones((1, 2, 2), dtype=np.float32),
        target_names=np.array(["gene_a"]),
        intensity_table=make_intensity_table(np.empty((0, 2, 2), dtype=np.float32)),
    )
    monkeypatch.setattr(decoding, "prepare_postcode_inputs", lambda *_: prepared)

    dataframe, raw = decoding.decode_spots_with_postcode(
        object(),
        object(),
        return_raw=True,
    )

    assert dataframe.empty
    assert raw is None
    assert {
        "target",
        "candidate_target",
        "assignment_class",
        "passes_thresholds",
        "assignment_probability",
        "decoder",
    } <= set(dataframe)


def test_spot_identity_is_stable_and_disambiguates_duplicate_detector_ids():
    dataframe = pd.DataFrame(
        {
            "spot_id": [5, 5, 8],
            "target": ["a", None, "b"],
            "passes_thresholds": [True, False, True],
        }
    )

    identified = decoding.add_spot_identity(dataframe, "R2", "fov_007")

    assert identified.columns[:4].tolist() == ["spot_uid", "region", "tile", "spot_id"]
    assert identified["spot_uid"].tolist() == [
        "R2:fov_007:5:0",
        "R2:fov_007:5:1",
        "R2:fov_007:8",
    ]
    assert identified["spot_uid"].is_unique


def test_full_postcode_artifacts_can_be_saved_without_torch(tmp_path):
    output = {
        "class_probs": np.array([[0.8, 0.1, 0.05, 0.05]], dtype=np.float32),
        "class_ind": {
            "genes": np.array([0, 1]),
            "bkg": 2,
            "inf": 3,
            "nan": np.empty(0, dtype=int),
        },
        "target_names": np.array(["a", "b"]),
        "barcodes": np.array([[[1, 0]], [[0, 1]]], dtype=np.float32),
        "params": {
            "w_star": np.array([0.4, 0.6]),
            "losses": [10.0, 5.0],
        },
        "norm_const": {"log_add": np.array([1.0, 2.0])},
    }

    posterior_path, model_path = decoding.save_postcode_decoder_artifacts(
        output,
        tmp_path,
        "fov_001",
        ["R1:fov_001:0"],
        postcode_kwargs={"device": "cpu", "set_seed": 3},
    )

    with np.load(posterior_path, allow_pickle=False) as posterior:
        assert posterior["class_probs"].shape == (1, 4)
        assert posterior["spot_uid"].tolist() == ["R1:fov_001:0"]
        assert posterior["target_names"].tolist() == ["a", "b"]
    with np.load(model_path, allow_pickle=False) as model:
        assert model["barcodes"].shape == (2, 1, 2)
        assert model["params_losses"].tolist() == [10.0, 5.0]
        assert model["norm_const_log_add"].tolist() == [1.0, 2.0]
        assert '"device": "cpu"' in model["postcode_kwargs_json"].item()


def test_postcode_table_round_trips_through_parquet(tmp_path):
    dataframe = pd.DataFrame(
        {
            "spot_id": [0, 1],
            "target": ["gene_a", None],
            "candidate_target": ["gene_a", "gene_b"],
            "passes_thresholds": [True, False],
            "quality_all_bases": [[0.9, 0.8], [0.7, 0.6]],
        }
    )
    dataframe = decoding.add_spot_identity(dataframe, "R1", "fov_000")
    path = tmp_path / "fov_000.parquet"

    dataframe.to_parquet(path, index=False)
    restored = pd.read_parquet(path)

    assert restored["spot_uid"].tolist() == ["R1:fov_000:0", "R1:fov_000:1"]
    assert pd.isna(restored.loc[1, "target"])
    np.testing.assert_allclose(
        np.stack(restored["quality_all_bases"].to_numpy()),
        [[0.9, 0.8], [0.7, 0.6]],
    )


def test_process_experiment_writes_postcode_output_layout(monkeypatch, tmp_path):
    region_dir = tmp_path / "R1"
    spacetx_dir = region_dir / "decoding" / "1_SpaceTX_format"
    spacetx_dir.mkdir(parents=True)

    class FakeExperiment:
        codebook = object()

        def keys(self):
            return ["fov_000", "fov_001", "fov_002"]

        def __getitem__(self, tile_id):
            return tile_id

    monkeypatch.setattr(
        decoding,
        "Experiment",
        type("Experiment", (), {"from_json": staticmethod(lambda _: FakeExperiment())}),
    )
    monkeypatch.setattr(
        decoding,
        "read_spacetx_coordinate_metadata",
        lambda _: ("microns", 0.5),
    )

    def fake_pipeline(tile, codebook, **kwargs):
        if tile == "fov_002":
            empty = pd.DataFrame(
                {
                    "spot_id": pd.Series(dtype="int64"),
                    "target": pd.Series(dtype="object"),
                    "candidate_target": pd.Series(dtype="object"),
                    "assignment_class": pd.Series(dtype="object"),
                    "passes_thresholds": pd.Series(dtype="bool"),
                }
            )
            return (empty, None) if kwargs["return_postcode_raw"] else empty
        dataframe = pd.DataFrame(
            {
                "spot_id": [0, 1],
                "x": [10, 20],
                "y": [11, 21],
                "z": [0, 0],
                "target": ["gene_a", None],
                "candidate_target": ["gene_a", "gene_b"],
                "assignment_class": ["gene", "background"],
                "passes_thresholds": [True, False],
                "assignment_probability": [0.9, 0.8],
                "best_gene_probability": [0.9, 0.1],
                "background_probability": [0.05, 0.8],
                "infeasible_probability": [0.05, 0.1],
                "nan_probability": [0.0, 0.0],
                "second_gene": ["gene_b", "gene_a"],
                "second_gene_probability": [0.05, 0.05],
                "gene_probability_margin": [0.85, 0.05],
                "quality_all_bases": [[0.9, 0.8], [0.7, 0.6]],
                "decoder": ["postcode", "postcode"],
            }
        )
        raw = {
            "class_probs": np.array(
                [[0.9, 0.05, 0.03, 0.02], [0.1, 0.0, 0.8, 0.1]],
                dtype=np.float32,
            ),
            "class_ind": {
                "genes": np.array([0, 1]),
                "bkg": 2,
                "inf": 3,
                "nan": np.empty(0, dtype=int),
            },
            "target_names": np.array(["gene_a", "gene_b"]),
            "params": {"losses": [10.0, 5.0]},
            "norm_const": {"log_add": np.array([1.0])},
        }
        return (dataframe, raw) if kwargs["return_postcode_raw"] else dataframe

    monkeypatch.setattr(decoding, "ISS_pipeline", fake_pipeline)

    decoding.process_experiment(
        tmp_path,
        decode_mode="POSTCODE",
        prob_threshold=0.7,
        postcode_kwargs={"device": "cpu", "set_seed": 3},
        save_postcode_artifacts=True,
    )

    output_dir = region_dir / "decoding" / "2_decoded_postcode"
    final_parquet = output_dir / "R1_decoded_postcode.parquet"
    final_csv = output_dir / "R1_decoded_postcode.csv"
    assert final_parquet.exists()
    assert final_csv.exists()
    assert sorted(path.name for path in (output_dir / "tiles").glob("*.parquet")) == [
        "fov_000.parquet",
        "fov_001.parquet",
        "fov_002.parquet",
    ]
    assert len(list((output_dir / "posteriors").glob("*.npz"))) == 2
    assert len(list((output_dir / "models").glob("*.npz"))) == 2

    region_table = pd.read_parquet(final_parquet)
    assert len(region_table) == 4
    assert region_table["spot_uid"].is_unique
    assert region_table["region"].unique().tolist() == ["R1"]

    manifest_paths = list(output_dir.glob("decoding_run_*.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text())
    assert manifest["schema_version"] == "1.0"
    assert manifest["decoder"]["commit"] == decoding.POSTCODE_COMMIT
    assert manifest["tiles"] == {"done": 3, "remaining": 0, "total": 3}
    assert manifest["rows"] == 4

    final_csv.unlink()
    monkeypatch.setattr(
        decoding,
        "ISS_pipeline",
        lambda *_args, **_kwargs: pytest.fail("completed region should not be decoded again"),
    )
    decoding.process_experiment(tmp_path, decode_mode="POSTCODE")
    assert final_csv.exists()

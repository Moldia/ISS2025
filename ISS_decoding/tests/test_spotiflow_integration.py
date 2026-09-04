import json
import sys
import types
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from starfish.types import Features

from ISS_decoding import decoding


def test_spotiflow_settings_are_validated_and_normalized():
    settings = decoding.effective_spotiflow_kwargs(
        {
            "model": "general",
            "probability_threshold": 0.4,
            "min_distance": 3,
            "n_tiles": [2, 4],
            "measurement_type": "max",
        }
    )

    assert settings == {
        "model": "general",
        "probability_threshold": 0.4,
        "min_distance": 3,
        "n_tiles": (2, 4),
        "measurement_type": "max",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unexpected": 1}, "Unknown Spotiflow setting"),
        ({"probability_threshold": 1.1}, "between 0 and 1"),
        ({"min_distance": 0}, "greater than zero"),
        ({"n_tiles": (2,)}, "two positive integers"),
        ({"measurement_type": "median"}, "must be 'mean' or 'max'"),
    ],
)
def test_invalid_spotiflow_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        decoding.effective_spotiflow_kwargs(overrides)


def test_unknown_spot_detection_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown spot_detection_mode"):
        decoding.normalize_spot_detection_mode("not-a-detector")


def test_starfish_detector_uses_blob_parameters(monkeypatch):
    captured = {}
    detector = object()

    def fake_blob_detector(**kwargs):
        captured.update(kwargs)
        return detector

    monkeypatch.setattr(decoding.FindSpots, "BlobDetector", fake_blob_detector)

    result = decoding.create_spot_detector(
        "STARFISH",
        int_threshold=0.007,
        sigma_vals=(2, 8, 12),
    )

    assert result is detector
    assert captured == {
        "min_sigma": 2,
        "max_sigma": 8,
        "num_sigma": 12,
        "threshold": 0.007,
        "measurement_type": "mean",
    }


def test_spotiflow_detector_receives_settings_and_preserves_probability(monkeypatch):
    captured = {}

    class FakeSpotiflowDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def image_to_spots(self, _data_image):
            dataframe = pd.DataFrame({Features.INTENSITY: [0.9, 0.6]})
            return SimpleNamespace(spot_attrs=SimpleNamespace(data=dataframe))

    spotiflow_package = types.ModuleType("spotiflow")
    spotiflow_package.__path__ = []
    starfish_module = types.ModuleType("spotiflow.starfish")
    starfish_module.SpotiflowDetector = FakeSpotiflowDetector
    monkeypatch.setitem(sys.modules, "spotiflow", spotiflow_package)
    monkeypatch.setitem(sys.modules, "spotiflow.starfish", starfish_module)

    detector = decoding.create_spot_detector(
        "spotiflow",
        spotiflow_kwargs={
            "model": "hybiss",
            "probability_threshold": 0.35,
            "min_distance": 3,
            "n_tiles": (2, 2),
            "measurement_type": "mean",
        },
    )

    assert captured == {
        "model": "hybiss",
        "probability_threshold": 0.35,
        "min_distance": 3,
        "n_tiles": (2, 2),
        "measurement_type": "mean",
        "is_volume": False,
        "subpix": False,
    }
    results = detector.image_to_spots(np.zeros((5, 5), dtype=np.float32))
    assert results.spot_attrs.data["spotiflow_probability"].tolist() == [0.9, 0.6]


def test_iss_pipeline_uses_the_supplied_spotiflow_detector(monkeypatch):
    class FakeStack:
        def reduce(self, *_args, **_kwargs):
            return self

    class FakeTile:
        def get_image(self, _name):
            return FakeStack()

    class FakeFilter:
        def run(self, stack, **_kwargs):
            return stack

    monkeypatch.setattr(decoding.Filter, "WhiteTophat", lambda *_args, **_kwargs: FakeFilter())
    monkeypatch.setattr(
        decoding.Filter,
        "MatchHistograms",
        lambda *_args, **_kwargs: FakeFilter(),
    )
    monkeypatch.setattr(
        decoding,
        "create_spot_detector",
        lambda *_args, **_kwargs: pytest.fail("supplied detector should be reused"),
    )

    detected_spots = object()

    class FakeDetector:
        def __init__(self):
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return detected_spots

    detector = FakeDetector()

    decoded = object()

    class FakeDecoder:
        def run(self, *, spots):
            assert spots is detected_spots
            return decoded

    monkeypatch.setattr(
        decoding.DecodeSpots,
        "PerRoundMaxChannel",
        lambda **_kwargs: FakeDecoder(),
    )
    expected = pd.DataFrame({"target": ["gene_a"]})
    monkeypatch.setattr(
        decoding,
        "QC_score_calc",
        lambda value: expected if value is decoded else pytest.fail("wrong decoded table"),
    )

    result = decoding.ISS_pipeline(
        FakeTile(),
        object(),
        register=False,
        decode_mode="PRMC",
        spot_detection_mode="spotiflow",
        spot_detector=detector,
    )

    assert result is expected
    assert len(detector.calls) == 1
    assert isinstance(detector.calls[0]["reference_image"], FakeStack)
    assert isinstance(detector.calls[0]["image_stack"], FakeStack)


@pytest.mark.parametrize(
    ("decode_mode", "dense", "detector", "expected"),
    [
        ("PRMC", False, "starfish", "2_decoded"),
        ("PRMC", False, "spotiflow", "2_decoded_spotiflow"),
        ("POSTCODE", False, "starfish", "2_decoded_postcode"),
        ("POSTCODE", False, "spotiflow", "2_decoded_postcode_spotiflow"),
        ("PRMC", True, "starfish", "2_decoded_dense"),
        ("PRMC", True, "spotiflow", "2_decoded_dense_spotiflow"),
    ],
)
def test_detector_and_decoder_output_directories_are_separate(
    decode_mode, dense, detector, expected
):
    assert decoding.decoding_output_subdir(decode_mode, dense, detector) == expected


@pytest.mark.parametrize(
    ("decode_mode", "expected_subdir", "expected_stem"),
    [
        ("PRMC", "2_decoded_spotiflow", "R1_decoded"),
        (
            "POSTCODE",
            "2_decoded_postcode_spotiflow",
            "R1_decoded_postcode",
        ),
    ],
)
def test_process_experiment_reuses_spotiflow_detector_and_records_provenance(
    monkeypatch, tmp_path, decode_mode, expected_subdir, expected_stem
):
    region_dir = tmp_path / "R1"
    (region_dir / "decoding" / "1_SpaceTX_format").mkdir(parents=True)

    class FakeExperiment:
        codebook = object()

        def keys(self):
            return ["fov_000", "fov_001"]

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
    monkeypatch.setattr(decoding, "installed_spotiflow_version", lambda: "0.5.8")

    detector = object()
    detector_calls = []

    def fake_create_spot_detector(mode, **kwargs):
        detector_calls.append((mode, kwargs))
        return detector

    monkeypatch.setattr(decoding, "create_spot_detector", fake_create_spot_detector)

    pipeline_calls = []

    def fake_pipeline(tile, _codebook, **kwargs):
        pipeline_calls.append((tile, kwargs))
        return pd.DataFrame(
            {
                "spot_id": [0],
                "x": [10],
                "y": [20],
                "z": [0],
                "target": ["gene_a"],
                "passes_thresholds": [True],
                "spotiflow_probability": [0.9],
            }
        )

    monkeypatch.setattr(decoding, "ISS_pipeline", fake_pipeline)

    spotiflow_kwargs = {
        "model": "hybiss",
        "probability_threshold": 0.4,
        "min_distance": 3,
        "n_tiles": (2, 2),
        "measurement_type": "mean",
    }
    decoding.process_experiment(
        tmp_path,
        decode_mode=decode_mode,
        spot_detection_mode="Spotiflow",
        spotiflow_kwargs=spotiflow_kwargs,
        prob_threshold=0.75,
    )

    assert detector_calls == [
        ("spotiflow", {"spotiflow_kwargs": spotiflow_kwargs})
    ]
    assert len(pipeline_calls) == 2
    assert all(call[1]["spot_detector"] is detector for call in pipeline_calls)
    assert all(call[1]["spot_detection_mode"] == "spotiflow" for call in pipeline_calls)
    assert all(call[1]["prob_threshold"] == 0.75 for call in pipeline_calls)

    output_dir = region_dir / "decoding" / expected_subdir
    final_parquet = output_dir / f"{expected_stem}.parquet"
    assert final_parquet.exists()
    result = pd.read_parquet(final_parquet)
    assert result["spot_detector"].unique().tolist() == ["spotiflow"]
    assert result["spotiflow_probability"].tolist() == [0.9, 0.9]

    xml_path = next(output_dir.glob("decoding_run_*.xml"))
    parameters = ET.parse(xml_path).getroot().find("Parameters")
    assert parameters.findtext("spot_detection_mode") == "spotiflow"
    assert parameters.findtext("spotiflow_version") == "0.5.8"
    assert json.loads(parameters.findtext("spotiflow_kwargs")) == {
        **spotiflow_kwargs,
        "n_tiles": [2, 2],
    }

    default_output = region_dir / "decoding" / (
        "2_decoded_postcode" if decode_mode == "POSTCODE" else "2_decoded"
    )
    assert not default_output.exists()

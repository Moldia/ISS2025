import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from starfish import Codebook, ImageStack

from ISS_decoding import decoding


class FakeExperiment:
    codebook = object()

    def keys(self):
        return ["fov_000", "fov_001"]

    def __getitem__(self, tile_id):
        return tile_id


def test_effective_pixel_kwargs_validates_overrides():
    settings = decoding.effective_pixel_kwargs(
        {"distance_threshold": 0.4, "n_processes": 2}
    )
    assert settings["distance_threshold"] == 0.4
    assert settings["n_processes"] == 2

    with pytest.raises(ValueError, match="Unknown Starfish pixel-decoding"):
        decoding.effective_pixel_kwargs({"unknown": 1})
    with pytest.raises(ValueError, match="max_area"):
        decoding.effective_pixel_kwargs({"min_area": 10, "max_area": 10})
    with pytest.raises(ValueError, match="n_processes"):
        decoding.effective_pixel_kwargs({"n_processes": 0})


def test_pixel_pipeline_runs_starfish_pixel_decoder(monkeypatch):
    codebook = Codebook.synthetic_one_hot_codebook(
        n_round=2,
        n_channel=2,
        n_codes=1,
        target_names=["gene_a"],
    )
    code = np.asarray(codebook.values[0], dtype=np.float32)
    array = np.broadcast_to(code[:, :, None, None, None], (2, 2, 1, 8, 8)).copy()
    stack = ImageStack.from_numpy(array)
    monkeypatch.setattr(
        decoding,
        "preprocess_iss_tile",
        lambda *_args, **_kwargs: (stack, stack),
    )

    result = decoding.ISS_pixel_pipeline(
        object(),
        codebook,
        pixel_kwargs={"min_area": 1, "max_area": 1000, "n_processes": 1},
    )

    assert len(result) == 1
    assert result.loc[0, "target"] == "gene_a"
    assert result.loc[0, "pixel_area"] == 64
    assert result.loc[0, "passes_thresholds"]
    assert result.loc[0, "decoder"] == "starfish_pixel"


def test_pixel_pipeline_returns_typed_empty_table_when_no_pixels_pass(monkeypatch):
    codebook = Codebook.synthetic_one_hot_codebook(
        n_round=2,
        n_channel=2,
        n_codes=1,
        target_names=["gene_a"],
    )
    stack = ImageStack.from_numpy(np.zeros((2, 2, 1, 8, 8), dtype=np.float32))
    monkeypatch.setattr(
        decoding,
        "preprocess_iss_tile",
        lambda *_args, **_kwargs: (stack, stack),
    )

    result = decoding.ISS_pixel_pipeline(object(), codebook)

    assert result.empty
    assert result["spot_id"].dtype == "int64"
    assert result["passes_thresholds"].dtype == "bool"
    assert "pixel_distance" in result


def test_pixel_process_is_restartable_and_records_provenance(monkeypatch, tmp_path):
    region_dir = tmp_path / "R1"
    (region_dir / "decoding" / "1_SpaceTX_format").mkdir(parents=True)
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
    monkeypatch.setattr(
        decoding,
        "create_spot_detector",
        lambda *_args, **_kwargs: pytest.fail(
            "pixel decoding must bypass spot detectors"
        ),
    )

    calls = []

    def fake_pipeline(tile, _codebook, **kwargs):
        calls.append((tile, kwargs))
        tile_index = int(tile.rsplit("_", 1)[1])
        return pd.DataFrame(
            {
                "spot_id": [0],
                "x": [10 + tile_index],
                "y": [20 + tile_index],
                "z": [0],
                "xc": [5 + tile_index / 2],
                "yc": [10 + tile_index / 2],
                "zc": [0],
                "target": ["gene_a"],
                "candidate_target": ["gene_a"],
                "assignment_class": ["gene"],
                "passes_thresholds": [True],
                "distance": [0.1],
                "pixel_area": [8],
                "pixel_distance": [0.1],
                "decoder": ["starfish_pixel"],
            }
        )

    monkeypatch.setattr(decoding, "ISS_pixel_pipeline", fake_pipeline)
    settings = {
        "distance_threshold": 0.4,
        "magnitude_threshold": 0.2,
        "min_area": 3,
        "max_area": 40,
        "n_processes": 2,
    }

    decoding.process_experiment(
        tmp_path,
        decode_mode="PIXEL",
        pixel_kwargs=settings,
    )

    assert len(calls) == 2
    assert all(call[1]["pixel_kwargs"]["distance_threshold"] == 0.4 for call in calls)
    output_dir = region_dir / "decoding" / "2_decoded_pixel"
    final_parquet = output_dir / "R1_decoded_pixel.parquet"
    final_csv = output_dir / "R1_decoded_pixel.csv"
    assert final_parquet.exists()
    assert final_csv.exists()

    result = pd.read_parquet(final_parquet)
    assert result["spot_uid"].tolist() == ["R1:fov_000:0", "R1:fov_001:0"]
    assert result["spot_detector"].unique().tolist() == ["starfish_pixel"]
    assert result["decoder"].unique().tolist() == ["starfish_pixel"]

    xml_path = next(output_dir.glob("decoding_run_*.xml"))
    parameters = ET.parse(xml_path).getroot().find("Parameters")
    assert parameters.findtext("decode_mode") == "PIXEL"
    assert parameters.findtext("spot_detection_mode") == "starfish_pixel"
    assert parameters.findtext("starfish_version")
    assert json.loads(parameters.findtext("pixel_kwargs"))["max_area"] == 40

    manifest_path = next(output_dir.glob("decoding_run_*.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["decoder"]["name"] == "starfish_pixel"
    assert manifest["decoder"]["version"]
    assert manifest["decoder"]["commit"] is None
    assert manifest["parameters"]["pixel_kwargs"]["n_processes"] == 2
    assert manifest["rows"] == 2

    monkeypatch.setattr(
        decoding,
        "ISS_pixel_pipeline",
        lambda *_args, **_kwargs: pytest.fail("completed output must be skipped"),
    )
    decoding.process_experiment(tmp_path, decode_mode="PIXEL")
    assert len(list(output_dir.glob("decoding_run_*.xml"))) == 1


def test_pixel_decoder_rejects_incompatible_pipeline_modes(tmp_path):
    with pytest.raises(ValueError, match="joint spot detection"):
        decoding.process_experiment(
            tmp_path,
            decode_mode="PIXEL",
            spot_detection_mode="spotiflow",
        )
    with pytest.raises(ValueError, match="dense=True"):
        decoding.process_experiment(tmp_path, decode_mode="PIXEL", dense=True)

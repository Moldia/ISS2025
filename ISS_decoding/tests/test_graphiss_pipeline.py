import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from ISS_decoding import decoding


class FakeExperiment:
    codebook = object()

    def keys(self):
        return ["fov_000", "fov_001"]

    def __getitem__(self, tile_id):
        return tile_id


def test_graphiss_pipeline_leaves_tophat_filtering_to_graphiss(monkeypatch):
    captured = {}

    def fake_preprocess(tile, **kwargs):
        captured["preprocess"] = (tile, kwargs)
        return object(), "scaled"

    def fake_decode(image, codebook, graphiss_kwargs):
        captured["decode"] = (image, codebook, graphiss_kwargs)
        return "result"

    monkeypatch.setattr(decoding, "preprocess_iss_tile", fake_preprocess)
    monkeypatch.setattr(decoding, "decode_imagestack_with_graphiss", fake_decode)

    result = decoding.ISS_graphiss_pipeline(
        "tile",
        "codebook",
        graphiss_kwargs={"h": 0.04},
    )

    assert result == "result"
    assert captured["preprocess"][1]["filter_images"] is False
    assert captured["decode"] == ("scaled", "codebook", {"h": 0.04})


def test_graphiss_process_is_restartable_and_records_provenance(monkeypatch, tmp_path):
    region_dir = tmp_path / "R1"
    (region_dir / "decoding" / "1_SpaceTX_format").mkdir(parents=True)
    monkeypatch.setattr(
        decoding,
        "Experiment",
        type("Experiment", (), {"from_json": staticmethod(lambda _: FakeExperiment())}),
    )
    monkeypatch.setattr(
        decoding, "read_spacetx_coordinate_metadata", lambda _: ("microns", 0.5)
    )
    monkeypatch.setattr(decoding, "installed_graphiss_version", lambda: "0.1.0")
    monkeypatch.setattr(
        decoding,
        "create_spot_detector",
        lambda *_args, **_kwargs: pytest.fail("Graph-ISS must bypass spot detectors"),
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
                "target_id": [0],
                "assignment_class": ["gene"],
                "passes_thresholds": [True],
                "graphiss_sequence": ["0-1-0"],
                "graphiss_signal_probability_sum": [2.5],
                "graphiss_signal_probability_min": [0.8],
                "graphiss_max_distance": [1.5],
                "graphiss_quality": [2.0],
                "decoder": ["graphiss"],
                "spot_detector": ["graphiss_joint"],
            }
        )

    monkeypatch.setattr(decoding, "ISS_graphiss_pipeline", fake_pipeline)
    settings = {
        "search_mode": "prior",
        "h": 0.04,
        "candidate_probability_threshold": 0.5,
        "quality_threshold": 1.0,
    }

    decoding.process_experiment(
        tmp_path,
        decode_mode="GRAPHISS",
        graphiss_kwargs=settings,
    )

    assert len(calls) == 2
    assert all(call[1]["graphiss_kwargs"]["h"] == 0.04 for call in calls)
    output_dir = region_dir / "decoding" / "2_decoded_graphiss"
    final_parquet = output_dir / "R1_decoded_graphiss.parquet"
    final_csv = output_dir / "R1_decoded_graphiss.csv"
    assert final_parquet.exists()
    assert final_csv.exists()
    assert sorted(path.name for path in (output_dir / "tiles").glob("*.parquet")) == [
        "fov_000.parquet",
        "fov_001.parquet",
    ]

    result = pd.read_parquet(final_parquet)
    assert result["spot_uid"].tolist() == ["R1:fov_000:0", "R1:fov_001:0"]
    assert result["spot_detector"].unique().tolist() == ["graphiss_joint"]
    assert result["decoder"].unique().tolist() == ["graphiss"]

    xml_path = next(output_dir.glob("decoding_run_*.xml"))
    parameters = ET.parse(xml_path).getroot().find("Parameters")
    assert parameters.findtext("decode_mode") == "GRAPHISS"
    assert parameters.findtext("spot_detection_mode") == "graphiss_joint"
    assert parameters.findtext("graphiss_version") == "0.1.0"
    assert parameters.findtext("graphiss_commit") == decoding.GRAPHISS_COMMIT
    assert json.loads(parameters.findtext("graphiss_kwargs"))["h"] == 0.04

    manifest_path = next(output_dir.glob("decoding_run_*.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["decoder"] == {
        "name": "graphiss",
        "commit": decoding.GRAPHISS_COMMIT,
        "version": "0.1.0",
    }
    assert manifest["tiles"] == {"done": 2, "remaining": 0, "total": 2}
    assert manifest["rows"] == 2

    monkeypatch.setattr(
        decoding,
        "ISS_graphiss_pipeline",
        lambda *_args, **_kwargs: pytest.fail("completed output must be skipped"),
    )
    decoding.process_experiment(tmp_path, decode_mode="GRAPHISS")
    assert len(list(output_dir.glob("decoding_run_*.xml"))) == 1


def test_graphiss_rejects_incompatible_pipeline_modes(tmp_path):
    with pytest.raises(ValueError, match="joint spot detection"):
        decoding.process_experiment(
            tmp_path,
            decode_mode="GRAPHISS",
            spot_detection_mode="spotiflow",
        )
    with pytest.raises(ValueError, match="dense=True"):
        decoding.process_experiment(tmp_path, decode_mode="GRAPHISS", dense=True)

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


def test_bardensr_process_is_restartable_and_records_provenance(monkeypatch, tmp_path):
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
    monkeypatch.setattr(decoding, "installed_bardensr_version", lambda: "0.3.2")
    monkeypatch.setattr(
        decoding,
        "create_spot_detector",
        lambda *_args, **_kwargs: pytest.fail("Bardensr must bypass spot detectors"),
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
                "bardensr_evidence": [0.8],
                "bardensr_peak_threshold": [0.72],
                "bardensr_tile_max": [0.9],
                "bardensr_tile": ["0_0"],
                "bardensr_method": ["singleshot"],
                "decoder": ["bardensr"],
                "spot_detector": ["bardensr_joint"],
            }
        )

    monkeypatch.setattr(decoding, "ISS_bardensr_pipeline", fake_pipeline)
    settings = {
        "method": "singleshot",
        "tile_size": (256, 256),
        "peak_threshold": 0.7,
        "device": "cpu",
    }

    decoding.process_experiment(
        tmp_path,
        decode_mode="BARDENSR",
        bardensr_kwargs=settings,
    )

    assert len(calls) == 2
    assert all(call[1]["bardensr_kwargs"]["tile_size"] == (256, 256) for call in calls)
    output_dir = region_dir / "decoding" / "2_decoded_bardensr"
    final_parquet = output_dir / "R1_decoded_bardensr.parquet"
    final_csv = output_dir / "R1_decoded_bardensr.csv"
    assert final_parquet.exists()
    assert final_csv.exists()
    assert sorted(path.name for path in (output_dir / "tiles").glob("*.parquet")) == [
        "fov_000.parquet",
        "fov_001.parquet",
    ]

    result = pd.read_parquet(final_parquet)
    assert result["spot_uid"].tolist() == ["R1:fov_000:0", "R1:fov_001:0"]
    assert result["spot_detector"].unique().tolist() == ["bardensr_joint"]
    assert result["decoder"].unique().tolist() == ["bardensr"]

    xml_path = next(output_dir.glob("decoding_run_*.xml"))
    parameters = ET.parse(xml_path).getroot().find("Parameters")
    assert parameters.findtext("decode_mode") == "BARDENSR"
    assert parameters.findtext("spot_detection_mode") == "bardensr_joint"
    assert parameters.findtext("bardensr_version") == "0.3.2"
    assert parameters.findtext("bardensr_commit") == decoding.BARDENSR_COMMIT
    assert json.loads(parameters.findtext("bardensr_kwargs"))["tile_size"] == [256, 256]

    manifest_path = next(output_dir.glob("decoding_run_*.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["decoder"] == {
        "name": "bardensr",
        "commit": decoding.BARDENSR_COMMIT,
        "version": "0.3.2",
    }
    assert manifest["tiles"] == {"done": 2, "remaining": 0, "total": 2}
    assert manifest["rows"] == 2

    monkeypatch.setattr(
        decoding,
        "ISS_bardensr_pipeline",
        lambda *_args, **_kwargs: pytest.fail("completed output must be skipped"),
    )
    decoding.process_experiment(tmp_path, decode_mode="BARDENSR")
    assert len(list(output_dir.glob("decoding_run_*.xml"))) == 1


def test_bardensr_rejects_incompatible_pipeline_modes(tmp_path):
    with pytest.raises(ValueError, match="joint spot detection"):
        decoding.process_experiment(
            tmp_path,
            decode_mode="BARDENSR",
            spot_detection_mode="spotiflow",
        )
    with pytest.raises(ValueError, match="dense=True"):
        decoding.process_experiment(tmp_path, decode_mode="BARDENSR", dense=True)

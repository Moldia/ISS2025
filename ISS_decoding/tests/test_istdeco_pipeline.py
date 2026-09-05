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


def test_istdeco_process_is_restartable_and_records_provenance(monkeypatch, tmp_path):
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
    monkeypatch.setattr(decoding, "installed_istdeco_version", lambda: "0.1.0")
    monkeypatch.setattr(
        decoding,
        "create_spot_detector",
        lambda *_args, **_kwargs: pytest.fail("ISTDECO must bypass spot detectors"),
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
                "istdeco_intensity": [0.8],
                "istdeco_quality": [0.7],
                "istdeco_intensity_threshold": [0.2],
                "istdeco_tile": ["0_0"],
                "decoder": ["istdeco"],
                "spot_detector": ["istdeco_joint"],
            }
        )

    monkeypatch.setattr(decoding, "ISS_istdeco_pipeline", fake_pipeline)
    settings = {
        "sigma": 1.5,
        "tile_size": (256, 256),
        "niter": 2,
        "device": "cpu",
    }

    decoding.process_experiment(
        tmp_path,
        decode_mode="ISTDECO",
        istdeco_kwargs=settings,
    )

    assert len(calls) == 2
    assert all(call[1]["istdeco_kwargs"]["sigma"] == (1.5, 1.5) for call in calls)
    output_dir = region_dir / "decoding" / "2_decoded_istdeco"
    final_parquet = output_dir / "R1_decoded_istdeco.parquet"
    final_csv = output_dir / "R1_decoded_istdeco.csv"
    assert final_parquet.exists()
    assert final_csv.exists()
    assert sorted(path.name for path in (output_dir / "tiles").glob("*.parquet")) == [
        "fov_000.parquet",
        "fov_001.parquet",
    ]

    result = pd.read_parquet(final_parquet)
    assert result["spot_uid"].tolist() == ["R1:fov_000:0", "R1:fov_001:0"]
    assert result["spot_detector"].unique().tolist() == ["istdeco_joint"]
    assert result["decoder"].unique().tolist() == ["istdeco"]

    xml_path = next(output_dir.glob("decoding_run_*.xml"))
    parameters = ET.parse(xml_path).getroot().find("Parameters")
    assert parameters.findtext("decode_mode") == "ISTDECO"
    assert parameters.findtext("spot_detection_mode") == "istdeco_joint"
    assert parameters.findtext("istdeco_version") == "0.1.0"
    assert parameters.findtext("istdeco_commit") == decoding.ISTDECO_COMMIT
    assert json.loads(parameters.findtext("istdeco_kwargs"))["tile_size"] == [256, 256]

    manifest_path = next(output_dir.glob("decoding_run_*.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["decoder"] == {
        "name": "istdeco",
        "commit": decoding.ISTDECO_COMMIT,
        "version": "0.1.0",
    }
    assert manifest["tiles"] == {"done": 2, "remaining": 0, "total": 2}
    assert manifest["rows"] == 2

    monkeypatch.setattr(
        decoding,
        "ISS_istdeco_pipeline",
        lambda *_args, **_kwargs: pytest.fail("completed output must be skipped"),
    )
    decoding.process_experiment(tmp_path, decode_mode="ISTDECO")
    assert len(list(output_dir.glob("decoding_run_*.xml"))) == 1


def test_istdeco_rejects_incompatible_pipeline_modes(tmp_path):
    with pytest.raises(ValueError, match="joint spot detection"):
        decoding.process_experiment(
            tmp_path,
            decode_mode="ISTDECO",
            spot_detection_mode="spotiflow",
        )
    with pytest.raises(ValueError, match="dense=True"):
        decoding.process_experiment(tmp_path, decode_mode="ISTDECO", dense=True)

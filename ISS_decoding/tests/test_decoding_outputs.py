import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from ISS_decoding import decoding


class FakeExperiment:
    codebook = object()

    def keys(self):
        return ["fov_000", "fov_001", "fov_002"]

    def __getitem__(self, tile_id):
        return tile_id


def configure_fake_experiment(monkeypatch):
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

    def fake_pipeline(tile, _codebook, **_kwargs):
        if tile == "fov_002":
            return pd.DataFrame(
                {
                    "spot_id": pd.Series(dtype="int64"),
                    "target": pd.Series(dtype="object"),
                    "passes_thresholds": pd.Series(dtype="bool"),
                    "quality_all_bases": pd.Series(dtype="object"),
                }
            )
        tile_index = int(tile.rsplit("_", 1)[1])
        return pd.DataFrame(
            {
                "spot_id": [tile_index],
                "x": [10 + tile_index],
                "y": [20 + tile_index],
                "z": [0],
                "target": ["gene_a"],
                "passes_thresholds": [True],
                "quality_all_bases": [[0.9, 0.8]],
            }
        )

    monkeypatch.setattr(decoding, "ISS_pipeline", fake_pipeline)


@pytest.mark.parametrize(
    ("dense", "decoded_subdir"),
    [(False, "2_decoded"), (True, "2_decoded_dense")],
)
def test_starfish_outputs_use_parquet_tiles_and_region_csv(
    monkeypatch, tmp_path, dense, decoded_subdir
):
    region_dir = tmp_path / "R1"
    (region_dir / "decoding" / "1_SpaceTX_format").mkdir(parents=True)
    configure_fake_experiment(monkeypatch)

    decoding.process_experiment(tmp_path, decode_mode="PRMC", dense=dense)

    output_dir = region_dir / "decoding" / decoded_subdir
    final_parquet = output_dir / "R1_decoded.parquet"
    final_csv = output_dir / "R1_decoded.csv"
    assert final_parquet.exists()
    assert final_csv.exists()
    assert sorted(path.name for path in (output_dir / "tiles").glob("*.parquet")) == [
        "fov_000.parquet",
        "fov_001.parquet",
        "fov_002.parquet",
    ]
    assert not list(output_dir.glob("fov_*.csv"))

    region_table = pd.read_parquet(final_parquet)
    assert region_table["cont. spot ids"].tolist() == [0, 1]
    assert region_table["tile"].tolist() == ["fov_000", "fov_001"]
    np.testing.assert_allclose(
        np.stack(region_table["quality_all_bases"].to_numpy()),
        [[0.9, 0.8], [0.9, 0.8]],
    )

    xml_paths = list(output_dir.glob("decoding_run_*.xml"))
    assert len(xml_paths) == 1
    paths = ET.parse(xml_paths[0]).getroot().find("Paths")
    assert paths.findtext("FinalParquet") == str(final_parquet)
    assert paths.findtext("FinalCSV") == str(final_csv)
    tiles = ET.parse(xml_paths[0]).getroot().find("Tiles")
    assert tiles.findtext("tiles_done") == "3"
    assert tiles.findtext("tiles_remaining") == "0"

    final_csv.unlink()
    monkeypatch.setattr(
        decoding,
        "ISS_pipeline",
        lambda *_args, **_kwargs: pytest.fail("completed region should not run again"),
    )
    decoding.process_experiment(tmp_path, decode_mode="PRMC", dense=dense)
    assert final_csv.exists()
    assert len(list(output_dir.glob("decoding_run_*.xml"))) == 1


def test_legacy_starfish_region_csv_is_still_treated_as_complete(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "R1" / "decoding" / "2_decoded"
    output_dir.mkdir(parents=True)
    legacy_csv = output_dir / "R1_decoded.csv"
    legacy_csv.write_text("target,x,y\ngene_a,1,2\n", encoding="utf-8")
    monkeypatch.setattr(
        decoding,
        "Experiment",
        type(
            "Experiment",
            (),
            {
                "from_json": staticmethod(
                    lambda _: pytest.fail("legacy completed region should not be loaded")
                )
            },
        ),
    )

    decoding.process_experiment(tmp_path, decode_mode="PRMC")

    assert legacy_csv.exists()
    assert not (output_dir / "R1_decoded.parquet").exists()
    assert not list(output_dir.glob("decoding_run_*.xml"))

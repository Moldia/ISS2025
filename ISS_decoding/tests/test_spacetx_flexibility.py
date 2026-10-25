import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from ISS_decoding import SpaceTx_format


class _FakeCodebook:
    def to_json(self, path):
        Path(path).write_text('{"codebook": true}', encoding="utf-8")


class SpaceTxPathOverrideTests(unittest.TestCase):
    def _inputs(self, root: Path):
        region_dir = root / "region-folder-with-no-convention"
        tile_dir = root / "tiles-anywhere"
        region_dir.mkdir()
        tile_dir.mkdir()
        tifffile.imwrite(tile_dir / "Cycle9_s0_ch0.tif", np.ones((3, 4), dtype=np.uint16))
        (tile_dir / "positions.csv").write_text("0,0\n", encoding="utf-8")
        codebook = root / "custom-codebook.csv"
        codebook.write_text("GeneA,1\n", encoding="utf-8")
        return region_dir, tile_dir, codebook

    def test_explicit_region_tiles_and_output_are_used_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            region_dir, tile_dir, codebook = self._inputs(root)
            output_dir = root / "published-spacetx"
            captured = {}

            def fake_write_experiment_json(**kwargs):
                captured.update(kwargs)
                Path(kwargs["path"]).mkdir(parents=True, exist_ok=True)
                (Path(kwargs["path"]) / "experiment.json").write_text(
                    "{}", encoding="utf-8"
                )

            with (
                patch.object(
                    SpaceTx_format,
                    "write_experiment_json",
                    side_effect=fake_write_experiment_json,
                ),
                patch.object(
                    SpaceTx_format.Codebook,
                    "from_code_array",
                    return_value=_FakeCodebook(),
                ),
            ):
                SpaceTx_format.make_spacetx_format(
                    input_dir=None,
                    codebook_csv=codebook,
                    region_directories={"R5": region_dir},
                    tile_directories={"R5": {"Cycle9": tile_dir}},
                    cycle_names_by_region={"R5": ["Cycle9"]},
                    output_dirs={"R5": output_dir},
                    nuclei_channel="Cy3",
                )

            self.assertTrue((output_dir / "experiment.json").is_file())
            self.assertTrue((output_dir / "codebook.json").is_file())
            self.assertTrue(list(output_dir.glob("spacetx_run_*.xml")))
            primary = captured["primary_tile_fetcher"]
            auxiliary = captured["aux_tile_fetcher"]["nuclei"]
            self.assertEqual(primary.tile_directories["Cycle9"], tile_dir)
            self.assertEqual(auxiliary.tile_directories["Cycle9"], tile_dir)
            self.assertEqual(auxiliary.nuclei_channel, 2)

    def test_missing_explicit_tiles_are_checked_before_cached_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            region_dir = root / "region"
            region_dir.mkdir()
            codebook = root / "codebook.csv"
            codebook.write_text("GeneA,1\n", encoding="utf-8")
            output = root / "cached"
            output.mkdir()
            (output / "experiment.json").write_text("{}", encoding="utf-8")
            (output / "codebook.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "Explicit tile directory"):
                SpaceTx_format.make_spacetx_format(
                    input_dir=None,
                    codebook_csv=codebook,
                    region_directories={"R1": region_dir},
                    tile_directories={"R1": {"Cycle1": root / "missing"}},
                    output_dirs={"R1": output},
                )

    def test_legacy_tile_fetcher_path_is_unchanged(self):
        fetcher = SpaceTx_format.ISS2DPrimaryTileFetcher(
            Path("/dataset/R1"),
            ["Cycle1"],
            [2],
            {"Cycle1": np.array([[0, 0]])},
            (3, 4),
            1.0,
            "4_retiled",
        )
        tile = fetcher.get_tile(tile=0, c=0, ch=0, z=0)
        self.assertEqual(
            Path(tile.file_path),
            Path("/dataset/R1/preprocessing/Cycle1/4_retiled/Cycle1_s0_ch2.tif"),
        )


if __name__ == "__main__":
    unittest.main()

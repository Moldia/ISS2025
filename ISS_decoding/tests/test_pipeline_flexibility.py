import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ISS_decoding import decoding


class _FakeExperiment(dict):
    def __init__(self):
        super().__init__({"fov_000": object()})
        self.codebook = object()


class DecodingPathOverrideTests(unittest.TestCase):
    def _experiment_dir(self, root: Path) -> Path:
        directory = root / "arbitrary-spacetx"
        directory.mkdir(parents=True)
        (directory / "experiment.json").write_text("{}", encoding="utf-8")
        return directory

    @staticmethod
    def _decoded_frame():
        return pd.DataFrame({"target": ["GeneA"], "xc": [1.0], "yc": [2.0]})

    def test_exact_input_output_paths_and_parameter_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spacetx = self._experiment_dir(root)
            decoded_dir = root / "per-fov-cache"
            final_csv = root / "published" / "custom-name.csv"

            with (
                patch.object(decoding.Experiment, "from_json", return_value=_FakeExperiment()),
                patch.object(
                    decoding,
                    "read_spacetx_coordinate_metadata",
                    return_value=("microns", 0.2),
                ),
                patch.object(
                    decoding,
                    "ISS_pipeline",
                    return_value=self._decoded_frame(),
                ) as pipeline,
            ):
                decoding.process_experiment(
                    input_dir=None,
                    region_names=["sample-a"],
                    spacetx_dirs={"sample-a": spacetx},
                    decoded_dirs={"sample-a": decoded_dir},
                    final_csv_files={"sample-a": final_csv},
                    spot_detection_mode="starfish",
                    prob_threshold=0.7,
                )

            self.assertTrue((decoded_dir / "fov_000.csv").is_file())
            self.assertTrue(final_csv.is_file())
            self.assertEqual(pd.read_csv(final_csv)["target"].tolist(), ["GeneA"])
            self.assertEqual(pipeline.call_args.kwargs["spot_detection_mode"], "starfish")
            self.assertEqual(pipeline.call_args.kwargs["prob_threshold"], 0.7)

    def test_missing_explicit_input_is_not_hidden_by_cached_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "cached.csv"
            cached.write_text("target\nGeneA\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "experiment.json"):
                decoding.process_experiment(
                    input_dir=None,
                    region_names=["sample-a"],
                    spacetx_dirs={"sample-a": root / "missing"},
                    final_csv_files={"sample-a": cached},
                )

    def test_overwrite_recomputes_existing_fov_and_final_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spacetx = self._experiment_dir(root)
            decoded_dir = root / "decoded"
            decoded_dir.mkdir()
            pd.DataFrame({"target": ["old"]}).to_csv(
                decoded_dir / "fov_000.csv", index=False
            )
            final_csv = root / "result.csv"
            final_csv.write_text("old\nvalue\n", encoding="utf-8")

            with (
                patch.object(decoding.Experiment, "from_json", return_value=_FakeExperiment()),
                patch.object(
                    decoding,
                    "read_spacetx_coordinate_metadata",
                    return_value=("pixels", 1.0),
                ),
                patch.object(
                    decoding,
                    "ISS_pipeline",
                    return_value=self._decoded_frame(),
                ) as pipeline,
            ):
                decoding.process_experiment(
                    input_dir=None,
                    region_names=["sample-a"],
                    spacetx_dirs={"sample-a": spacetx},
                    decoded_dirs={"sample-a": decoded_dir},
                    final_csv_files={"sample-a": final_csv},
                    overwrite=True,
                )

            pipeline.assert_called_once()
            self.assertEqual(pd.read_csv(final_csv)["target"].tolist(), ["GeneA"])

    def test_legacy_paths_remain_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spacetx = root / "R1" / "decoding" / "1_SpaceTX_format"
            spacetx.mkdir(parents=True)
            (spacetx / "experiment.json").write_text("{}", encoding="utf-8")

            with (
                patch.object(decoding.Experiment, "from_json", return_value=_FakeExperiment()),
                patch.object(
                    decoding,
                    "read_spacetx_coordinate_metadata",
                    return_value=("unknown", None),
                ),
                patch.object(
                    decoding,
                    "ISS_pipeline",
                    return_value=self._decoded_frame(),
                ),
            ):
                decoding.process_experiment(root, regions_to_process=[1])

            expected = root / "R1" / "decoding" / "2_decoded" / "R1_decoded.csv"
            self.assertTrue(expected.is_file())


if __name__ == "__main__":
    unittest.main()

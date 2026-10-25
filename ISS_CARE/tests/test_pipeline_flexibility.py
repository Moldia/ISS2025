import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

# CARE is only needed for model execution, which these path/orchestration tests
# mock. Keeping the stub local avoids adding a test-only dependency.
if "csbdeep" not in sys.modules:
    csbdeep = types.ModuleType("csbdeep")
    models = types.ModuleType("csbdeep.models")
    models.CARE = object
    csbdeep.models = models
    sys.modules["csbdeep"] = csbdeep
    sys.modules["csbdeep.models"] = models
if "tensorflow" not in sys.modules:
    tensorflow = types.ModuleType("tensorflow")
    tensorflow.config = types.SimpleNamespace(
        list_physical_devices=lambda kind: [],
        experimental=types.SimpleNamespace(set_memory_growth=lambda gpu, enabled: None),
    )
    sys.modules["tensorflow"] = tensorflow

from ISS_CARE import ISS_CARE_prediction as prediction


class CarePathOverrideTests(unittest.TestCase):
    def test_process_cycle_uses_exact_input_and_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_tiles = root / "unrelated-input"
            output_tiles = root / "published-output"
            input_tiles.mkdir()
            tifffile.imwrite(
                input_tiles / "arbitrary_s0_ch4.tif",
                np.ones((4, 4), dtype=np.uint16),
            )
            (input_tiles / "coordinates.csv").write_text("0,0\n", encoding="utf-8")
            model = types.SimpleNamespace(name="fake", config=None)

            prediction.process_one_cycle(
                region_name="R9",
                cycle_name="Cycle12",
                cycle_dir=root / "unused-cycle",
                out_cycle_dir=root / "unused-output-cycle",
                model=model,
                model_dir=root / "model",
                model_name="fake",
                training_metadata_file=None,
                dapi_ch=4,
                normalize_input=False,
                normalization_pmin=1.0,
                normalization_pmax=99.8,
                normalization_eps=1e-8,
                output_rescale_mode="none",
                output_rescale_pmin=1.0,
                output_rescale_pmax=99.8,
                output_rescale_eps=1e-8,
                debug_prints=False,
                debug_print_limit=0,
                overwrite=False,
                run_id="test",
                preprocessing_relpath="preprocessing",
                tiles_subpath="4_retiled",
                axes="YX",
                out_dtype="uint16",
                input_tile_dir=input_tiles,
                output_tile_dir=output_tiles,
            )

            self.assertTrue((output_tiles / "arbitrary_s0_ch4.tif").is_file())
            self.assertTrue((output_tiles / "coordinates.csv").is_file())
            self.assertTrue((output_tiles / "CARE_run_test.xml").is_file())

    def test_missing_explicit_input_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "Explicit CARE input"):
                prediction.process_one_cycle(
                    region_name="R1",
                    cycle_name="Cycle1",
                    cycle_dir=root,
                    out_cycle_dir=root,
                    model=object(),
                    model_dir=root,
                    model_name="fake",
                    training_metadata_file=None,
                    dapi_ch=4,
                    normalize_input=False,
                    normalization_pmin=1.0,
                    normalization_pmax=99.8,
                    normalization_eps=1e-8,
                    output_rescale_mode="none",
                    output_rescale_pmin=1.0,
                    output_rescale_pmax=99.8,
                    output_rescale_eps=1e-8,
                    debug_prints=False,
                    debug_print_limit=0,
                    overwrite=False,
                    run_id="test",
                    preprocessing_relpath="preprocessing",
                    tiles_subpath="4_retiled",
                    axes="YX",
                    out_dtype="uint16",
                    input_tile_dir=root / "missing",
                )

    def test_high_level_mappings_reach_cycle_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cycle_dir = root / "cycle-anywhere"
            input_tiles = root / "tiles-anywhere"
            output_tiles = root / "output-anywhere"
            cycle_dir.mkdir()
            input_tiles.mkdir()

            with (
                patch.object(
                    prediction,
                    "resolve_inference_normalization_settings",
                    return_value=(False, 1.0, 99.8, 1e-8, None, None),
                ),
                patch.object(
                    prediction,
                    "CARE",
                    return_value=types.SimpleNamespace(name="fake"),
                ),
                patch.object(prediction, "process_one_cycle") as process,
            ):
                prediction.ISS_CARE_predict(
                    input_dir=None,
                    model_dir=root / "model",
                    model_name="fake",
                    dapi_ch=4,
                    cycle_directories={("R3", "Cycle8"): cycle_dir},
                    input_tile_dirs={("R3", "Cycle8"): input_tiles},
                    output_tile_dirs={("R3", "Cycle8"): output_tiles},
                )

            self.assertEqual(process.call_args.kwargs["cycle_dir"], cycle_dir)
            self.assertEqual(process.call_args.kwargs["input_tile_dir"], input_tiles)
            self.assertEqual(process.call_args.kwargs["output_tile_dir"], output_tiles)


if __name__ == "__main__":
    unittest.main()

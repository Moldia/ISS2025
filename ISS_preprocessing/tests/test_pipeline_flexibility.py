import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

# The decoding test environment used in CI may not include this small sorting
# dependency. The production environment declares it; a standard sort is enough
# for these path/orchestration unit tests.
try:
    import natsort  # noqa: F401
except ImportError:
    module = types.ModuleType("natsort")
    module.natsorted = sorted
    sys.modules["natsort"] = module

from ISS_preprocessing import preprocessing


class PreprocessingFlexibilityTests(unittest.TestCase):
    def test_later_stages_can_be_selected_without_raw_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            region = Path(tmp) / "R1"
            region.mkdir()
            with (
                patch.object(preprocessing, "deconvolve_and_mip") as extraction,
                patch.object(preprocessing, "mipped_to_OME_tiffs") as ome,
                patch.object(preprocessing, "align_and_stitch") as align,
                patch.object(preprocessing, "retile_stitched_images") as retile,
            ):
                preprocessing.preprocessing_main(
                    input_dirs=None,
                    cycles=[1],
                    output_dir_prefix=None,
                    mode="not-needed",
                    n_total_cycles=1,
                    run_extraction=False,
                    run_ome_conversion=False,
                    run_alignment=True,
                    run_retiling=False,
                    region_directories=[region],
                    overwrite=True,
                    ome_input_files={("R1", 1): Path(tmp) / "input.ome.tiff"},
                    stitched_output_dirs={("R1", 1): Path(tmp) / "stitched"},
                )

            extraction.assert_not_called()
            ome.assert_not_called()
            retile.assert_not_called()
            self.assertTrue(align.call_args.kwargs["overwrite"])
            self.assertEqual(
                align.call_args.kwargs["ome_input_files"][("R1", 1)],
                Path(tmp) / "input.ome.tiff",
            )

    def test_explicit_missing_mip_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            region = Path(tmp) / "R1"
            region.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "Explicit MIP directory"):
                preprocessing.mipped_to_OME_tiffs(
                    [region],
                    [1],
                    mipped_dirs={("R1", 1): Path(tmp) / "missing"},
                )

    def test_ashlar_paths_and_tuning_are_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            region = root / "external-region"
            region.mkdir()
            ome_path = root / "inputs" / "anything.ome.tiff"
            ome_path.parent.mkdir()
            ome_path.write_bytes(b"synthetic OME input")
            stitched_dir = root / "custom-stitched"
            work_dir = root / "custom-work"
            captured = {}

            fake_ashlar = types.ModuleType("ashlar.scripts.ashlar")
            fake_ashlar.configure_terminal = lambda: None
            fake_ashlar.print_error = lambda message: self.fail(message)
            fake_ashlar.ProcessingError = RuntimeError

            def process_single(
                inputs,
                pattern,
                flip_x,
                flip_y,
                ffp,
                dfp,
                barrel,
                aligner_args,
                mosaic_args,
                pyramid,
                quiet,
            ):
                captured.update(aligner_args)
                for channel in range(2):
                    output = Path(pattern.format(cycle=0, channel=channel))
                    output.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(output, np.zeros((4, 4), dtype=np.uint16))
                return 0

            fake_ashlar.process_single = process_single
            fake_ashlar.process_plates = lambda *args, **kwargs: self.fail("unexpected plates")
            ashlar_pkg = types.ModuleType("ashlar")
            scripts_pkg = types.ModuleType("ashlar.scripts")
            scripts_pkg.ashlar = fake_ashlar
            ashlar_pkg.scripts = scripts_pkg

            fake_series = types.SimpleNamespace(axes="CYX", shape=(2, 4, 4))
            fake_tiff = unittest.mock.MagicMock()
            fake_tiff.__enter__.return_value.series = [fake_series]

            with (
                patch.dict(
                    sys.modules,
                    {
                        "ashlar": ashlar_pkg,
                        "ashlar.scripts": scripts_pkg,
                        "ashlar.scripts.ashlar": fake_ashlar,
                    },
                ),
                patch.object(preprocessing.tifffile, "TiffFile", return_value=fake_tiff),
            ):
                preprocessing.align_and_stitch(
                    [region],
                    [7],
                    n_total_cycles=1,
                    align_channel=1,
                    maximum_shift=777,
                    filter_sigma=9,
                    ome_input_files={("external-region", 7): ome_path},
                    stitched_output_dirs={("external-region", 7): stitched_dir},
                    work_dirs={"external-region": work_dir},
                    overwrite=True,
                )

            self.assertEqual(captured["max_shift"], 777)
            self.assertEqual(captured["filter_sigma"], 9)
            self.assertTrue((stitched_dir / "Cycle7_ch0.tif").is_file())
            self.assertTrue((stitched_dir / "Cycle7_ch1.tif").is_file())

    def test_legacy_stage_defaults_are_unchanged(self):
        source = Path(preprocessing.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preprocessing_main"
        )
        defaults = {
            arg.arg: ast.literal_eval(default)
            for arg, default in zip(
                function.args.args[-len(function.args.defaults):],
                function.args.defaults,
            )
        }
        self.assertTrue(defaults["run_extraction"])
        self.assertTrue(defaults["run_ome_conversion"])
        self.assertTrue(defaults["run_alignment"])
        self.assertTrue(defaults["run_retiling"])
        self.assertFalse(defaults["overwrite"])


if __name__ == "__main__":
    unittest.main()

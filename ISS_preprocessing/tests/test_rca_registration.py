import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

try:
    import natsort  # noqa: F401
except ImportError:
    module = types.ModuleType("natsort")
    module.natsorted = sorted
    sys.modules["natsort"] = module

from ISS_preprocessing import preprocessing
from ISS_preprocessing.registration import (
    create_rca_augmented_ome,
    estimate_residual_translation,
    generate_rca_composite,
    resolve_rca_channels,
)


def write_test_ome(path, series_arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as writer:
        for tile_index, array in enumerate(series_arrays):
            n_channels = array.shape[0]
            writer.write(
                array,
                photometric="minisblack",
                metadata={
                    "axes": "CYX",
                    "Channel": {
                        "Name": [f"signal-{i}" for i in range(n_channels - 1)]
                        + ["DAPI"]
                    },
                    "Pixels": {
                        "PhysicalSizeX": 0.5,
                        "PhysicalSizeXUnit": "µm",
                        "PhysicalSizeY": 0.5,
                        "PhysicalSizeYUnit": "µm",
                    },
                    "Plane": {
                        "PositionX": [10.0 + tile_index] * n_channels,
                        "PositionY": [20.0 + tile_index] * n_channels,
                    },
                },
            )
    return path


class RCACompositeTests(unittest.TestCase):
    def test_max_sum_and_named_channel_selection(self):
        images = {
            "DAPI": np.full((2, 3), 100, dtype=np.uint16),
            "Cy5": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
            "Cy3": np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint16),
        }
        maximum = generate_rca_composite(
            images, method="max", background_subtraction=False
        )
        summed = generate_rca_composite(
            images,
            channels=["Cy3", "Cy5"],
            method="sum",
            background_subtraction=False,
        )
        np.testing.assert_array_equal(
            maximum,
            np.array([[6, 5, 4], [4, 5, 6]], dtype=np.float32),
        )
        np.testing.assert_array_equal(summed, np.full((2, 3), 7))

    def test_normalized_max_handles_empty_channels_and_integer_output(self):
        images = np.stack(
            [
                np.zeros((8, 8), dtype=np.uint16),
                np.full((8, 8), 5, dtype=np.uint16),
                np.arange(64, dtype=np.uint16).reshape(8, 8),
            ]
        )
        result = generate_rca_composite(
            images,
            channels=[0, 1, 2],
            output_dtype=np.uint16,
        )
        self.assertEqual(result.shape, (8, 8))
        self.assertEqual(result.dtype, np.uint16)
        self.assertGreater(int(result.max()), 0)

    def test_channel_resolution_excludes_dapi_unless_explicit(self):
        with self.assertRaisesRegex(ValueError, "must be specified"):
            generate_rca_composite(np.zeros((3, 4, 4), dtype=np.uint16))
        self.assertEqual(
            resolve_rca_channels(None, n_channels=5, nuclear_channel=4),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            resolve_rca_channels([4, 1], n_channels=5, nuclear_channel=4),
            [4, 1],
        )
        self.assertEqual(
            resolve_rca_channels(
                ["green", "red"],
                n_channels=3,
                nuclear_channel=2,
                channel_names=["red", "green", "DAPI"],
            ),
            [1, 0],
        )

    def test_augmented_ome_preserves_originals_and_spatial_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = [
                np.stack(
                    [
                        np.arange(20, dtype=np.uint16).reshape(4, 5),
                        np.full((4, 5), 7, dtype=np.uint16),
                        np.full((4, 5), 50, dtype=np.uint16),
                    ]
                ),
                np.stack(
                    [
                        np.arange(20, dtype=np.uint16).reshape(4, 5) + 2,
                        np.full((4, 5), 9, dtype=np.uint16),
                        np.full((4, 5), 60, dtype=np.uint16),
                    ]
                ),
            ]
            source = write_test_ome(root / "source.ome.tiff", arrays)
            output = root / "augmented.ome.tiff"
            create_rca_augmented_ome(
                source,
                output,
                rca_channels=[0, 1],
                method="max",
                background_subtraction=False,
            )

            with tifffile.TiffFile(output) as tiff:
                self.assertEqual(len(tiff.series), 2)
                for index, series in enumerate(tiff.series):
                    augmented = series.asarray()
                    self.assertEqual(augmented.shape, (4, 4, 5))
                    np.testing.assert_array_equal(augmented[:3], arrays[index])
                    np.testing.assert_array_equal(
                        augmented[3], np.maximum(arrays[index][0], arrays[index][1])
                    )
                metadata = tifffile.xml2dict(tiff.ome_metadata)["OME"]["Image"]
                self.assertEqual(metadata[0]["Pixels"]["PhysicalSizeX"], 0.5)
                self.assertEqual(metadata[1]["Pixels"]["Plane"][0]["PositionX"], 11.0)
                self.assertEqual(
                    metadata[0]["Pixels"]["Channel"][-1]["Name"],
                    "RCA composite",
                )

    def test_synthetic_multicycle_rca_translation_is_recovered(self):
        rng = np.random.default_rng(123)
        shape = (96, 96)
        yy, xx = np.indices(shape)
        locations = rng.integers(12, 84, size=(30, 2))

        def cycle_images(cycle):
            images = np.zeros((4,) + shape, dtype=np.float32)
            for spot_index, (y, x) in enumerate(locations):
                channel = (spot_index + cycle) % 4
                intensity = rng.uniform(0.5, 2.0)
                images[channel] += intensity * np.exp(
                    -((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.2**2)
                )
            return images

        reference = generate_rca_composite(cycle_images(0), channels=range(4))
        moving_channels = np.roll(
            cycle_images(1), shift=(3, -4), axis=(1, 2)
        )
        moving = generate_rca_composite(moving_channels, channels=range(4))
        shift, _, _ = estimate_residual_translation(reference, moving)
        np.testing.assert_allclose(shift, [-3, 4], atol=0.35)


class RCAAshlarIntegrationTests(unittest.TestCase):
    def test_augmented_inputs_use_standard_ashlar_for_all_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            region = root / "external-region"
            region.mkdir()
            arrays = [
                np.stack(
                    [
                        np.arange(64, dtype=np.uint16).reshape(8, 8),
                        np.full((8, 8), 100, dtype=np.uint16),
                    ]
                )
            ]
            inputs = {
                ("external-region", cycle): write_test_ome(
                    root / f"cycle-{cycle}.ome.tiff", arrays
                )
                for cycle in (1, 2)
            }
            outputs = {
                ("external-region", cycle): root / f"stitched-{cycle}"
                for cycle in (1, 2)
            }
            work = root / "work"
            diagnostic_dir = root / "custom-registration-qc"
            captured = {}

            fake_ashlar = types.ModuleType("ashlar.scripts.ashlar")
            fake_ashlar.configure_terminal = lambda: None
            fake_ashlar.print_error = lambda message: self.fail(message)
            fake_ashlar.ProcessingError = RuntimeError

            def process_single(
                paths,
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
                captured["paths"] = list(paths)
                captured["aligner_args"] = dict(aligner_args)
                captured["mosaic_args"] = dict(mosaic_args)
                captured["cycle_markers"] = []
                for cycle_index, path in enumerate(paths):
                    with tifffile.TiffFile(path) as tiff:
                        self.assertEqual(tiff.series[0].shape[0], 3)
                        captured["cycle_markers"].append(
                            Path(path).name
                        )
                    for channel in mosaic_args["channels"]:
                        output = Path(
                            pattern.format(cycle=cycle_index, channel=channel)
                        )
                        output.parent.mkdir(parents=True, exist_ok=True)
                        tifffile.imwrite(
                            output,
                            np.full((32, 32), channel + 1, dtype=np.uint16),
                        )
                return 0

            fake_ashlar.process_single = process_single
            fake_ashlar.process_plates = lambda *args, **kwargs: self.fail(
                "unexpected plates"
            )
            ashlar_pkg = types.ModuleType("ashlar")
            scripts_pkg = types.ModuleType("ashlar.scripts")
            scripts_pkg.ashlar = fake_ashlar
            ashlar_pkg.scripts = scripts_pkg

            with patch.dict(
                sys.modules,
                {
                    "ashlar": ashlar_pkg,
                    "ashlar.scripts": scripts_pkg,
                    "ashlar.scripts.ashlar": fake_ashlar,
                },
            ):
                preprocessing.align_and_stitch(
                    [region],
                    [1, 2],
                    n_total_cycles=2,
                    align_channel=1,
                    registration_reference="rca_composite",
                    rca_channels=[0],
                    reference_cycle=2,
                    ome_input_files=inputs,
                    stitched_output_dirs=outputs,
                    work_dirs={"external-region": work},
                    save_registration_composites=True,
                    registration_diagnostic_dirs={
                        "external-region": diagnostic_dir
                    },
                    overwrite=True,
                )

            # The appended channel is used once by the standard ASHLAR call for
            # both edge and layer alignment; all original channels share it.
            self.assertEqual(captured["aligner_args"]["channel"], 2)
            self.assertEqual(captured["mosaic_args"]["channels"], [0, 1, 2])
            self.assertEqual(captured["cycle_markers"], ["Cycle2.ome.tiff", "Cycle1.ome.tiff"])
            self.assertTrue((outputs[("external-region", 1)] / "Cycle1_ch0.tif").is_file())
            self.assertTrue((outputs[("external-region", 2)] / "Cycle2_ch1.tif").is_file())
            self.assertFalse((outputs[("external-region", 1)] / "Cycle1_ch2.tif").exists())
            self.assertTrue((diagnostic_dir / "Cycle1_rca_composite.tif").is_file())
            self.assertTrue((diagnostic_dir / "Cycle2_rca_composite.tif").is_file())
            self.assertTrue(all(not Path(path).exists() for path in captured["paths"]))

            metadata_path = diagnostic_dir / "registration_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["registration_reference"], "rca_composite")
            self.assertEqual(metadata["stitching_reference"], "rca_composite")
            self.assertEqual(metadata["reference_cycle"], 2)
            self.assertEqual(metadata["rca_channels"], [0])


if __name__ == "__main__":
    unittest.main()

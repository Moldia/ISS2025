import tempfile
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz

# PCIseq is an optional runtime dependency and is not needed for the output
# preflight exercised here.
if "pciSeq" not in sys.modules:
    pci_module = types.ModuleType("pciSeq")
    pci_module.fit = None
    pci_module.config = types.SimpleNamespace(DEFAULT={})
    sys.modules["pciSeq"] = pci_module

from ISS_postprocessing import annotated_objects, pciseq, segmentation


class PostprocessingPathOverrideTests(unittest.TestCase):
    @staticmethod
    def _write_inputs(root: Path):
        spots_path = root / "external" / "spots.csv"
        spots_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "target": ["GeneA", "GeneB"],
                "xc": [0.0, 1.0],
                "yc": [0.0, 1.0],
                "quality_minimum": [1.0, 1.0],
            }
        ).to_csv(spots_path, index=False)
        labels = np.zeros((3, 3), dtype=np.uint32)
        labels[0, 0] = 1
        labels[1, 1] = 2
        mask_path = root / "external" / "labels.npz"
        save_npz(mask_path, coo_matrix(labels))
        return spots_path, mask_path

    def test_create_anndata_accepts_exact_paths_without_pipeline_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spots, mask = self._write_inputs(root)
            output = root / "published" / "cells.custom.h5ad"

            adata = annotated_objects.create_anndata_obj(
                input_dir=None,
                region=None,
                spots_path=spots,
                segmentation_path=mask,
                output_file=output,
                filter_data=False,
                convert_coords=False,
            )

            self.assertEqual(adata.shape, (2, 2))
            self.assertTrue(output.is_file())

    def test_partial_override_uses_legacy_mask_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spots, source_mask = self._write_inputs(root)
            legacy_mask = (
                root
                / "R2"
                / "postprocessing"
                / "segmentation"
                / "R2_cellpose_expanded.npz"
            )
            legacy_mask.parent.mkdir(parents=True)
            legacy_mask.write_bytes(source_mask.read_bytes())

            adata = annotated_objects.create_anndata_obj(
                root,
                "R2",
                "cellpose",
                spots_path=spots,
                output_file=root / "custom.h5ad",
                filter_data=False,
                convert_coords=False,
            )
            self.assertEqual(adata.n_obs, 2)

    def test_invalid_explicit_input_is_checked_before_cached_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mask = self._write_inputs(root)
            cached = root / "cached.h5ad"
            cached.write_bytes(b"not read because validation comes first")

            with self.assertRaisesRegex(FileNotFoundError, "Decoded transcript"):
                annotated_objects.create_anndata_obj(
                    spots_path=root / "missing.csv",
                    segmentation_path=mask,
                    output_file=cached,
                )

    def test_segmentation_exact_output_can_be_reused_without_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing.npz"
            labels = np.array([[0, 1], [0, 2]], dtype=np.uint32)
            save_npz(output, coo_matrix(labels))

            dense, sparse = segmentation.cell_pose_segmentation_to_coo(
                input_dir=None,
                region=None,
                output_file=output,
                overwrite=False,
                auto_select_gpu=False,
            )
            np.testing.assert_array_equal(dense, labels)
            self.assertEqual(sparse.shape, labels.shape)

    def test_segmentation_invalid_explicit_input_precedes_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "existing.npz"
            save_npz(output, coo_matrix(np.array([[0, 1]], dtype=np.uint32)))
            with self.assertRaisesRegex(FileNotFoundError, "Explicit stitched image"):
                segmentation.cell_pose_segmentation_to_coo(
                    input_dir=None,
                    region=None,
                    input_image_type="stitched",
                    stitched_image_path=root / "missing.tif",
                    output_file=output,
                    overwrite=False,
                    auto_select_gpu=False,
                )

    def test_explicit_retiled_image_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "z.tif"
            second = root / "a.tif"
            first.touch()
            second.touch()
            resolved = segmentation.get_retiled_images(
                None,
                None,
                image_paths=[first, second],
            )
            self.assertEqual(resolved, [first, second])

    def test_legacy_segmentation_paths_are_unchanged(self):
        root = Path("/dataset")
        self.assertEqual(
            segmentation.get_retiled_dir(root, "R1"),
            root / "R1" / "preprocessing" / "Cycle1" / "4_retiled",
        )
        self.assertEqual(
            segmentation.get_segmentation_dir(root, "R1", create=False),
            root / "R1" / "postprocessing" / "segmentation",
        )

    def test_pciseq_exact_output_refuses_replacement_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "arbitrary-pciseq"
            output.mkdir()
            (output / "cellData.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "overwrite=True"):
                pciseq.run_pciseq(
                    "sample",
                    spots=None,
                    coo_mask=None,
                    sc_expression_matrix=None,
                    output_dir=output,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()

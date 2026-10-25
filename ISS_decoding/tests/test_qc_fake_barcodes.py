import numpy as np
import pandas as pd
import pytest

from ISS_decoding.qc_metrics import estimate_fake_barcode_fdr


def test_fake_barcode_fdr_scales_for_control_panel_size():
    reads = pd.DataFrame(
        {
            "is_fake_barcode": [False] * 20 + [True, True] + [True],
            "passes_thresholds": [True] * 22 + [False],
            "barcode_control_true_count": [100] * 23,
            "barcode_control_fake_count": [10] * 23,
        }
    )

    estimate = estimate_fake_barcode_fdr(reads)

    assert estimate["accepted_real_calls"] == 20
    assert estimate["accepted_fake_calls"] == 2
    assert estimate["estimated_false_calls"] == 20
    assert estimate["estimated_fdr"] == 1


def test_fake_barcode_fdr_requires_configured_controls():
    reads = pd.DataFrame(
        {
            "is_fake_barcode": [False],
            "passes_thresholds": [True],
            "barcode_control_true_count": [10],
            "barcode_control_fake_count": [0],
        }
    )
    with pytest.raises(ValueError, match="No fake barcodes"):
        estimate_fake_barcode_fdr(reads)

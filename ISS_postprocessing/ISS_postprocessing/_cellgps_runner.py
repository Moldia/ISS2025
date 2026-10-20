"""Execute a prepared CellGPS analysis inside its isolated environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def run(config_path: str | Path) -> None:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    try:
        import cellgps
    except ImportError as exc:
        raise RuntimeError(
            "CellGPS is not installed. Create ISS_postprocessing/environments/"
            "cellgps.yml and run this adapter through that environment."
        ) from exc

    points = pd.read_csv(config["input_csv"])
    columns = config["columns"]
    required = {columns["x"], columns["y"], columns["entity"]}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"Prepared CellGPS point table is missing columns: {missing}")

    analysis = config["analysis"]
    row_coph, col_coph = cellgps.compute_cophenetic_distances_from_df_memory_opt(
        df=points,
        x_col=columns["x"],
        y_col=columns["y"],
        celltype_col=columns["entity"],
        method=analysis["linkage_method"],
        show_corr=analysis["show_corr"],
        batch_size=analysis["batch_size"],
    )

    outputs = {name: Path(value) for name, value in config["outputs"].items()}
    outputs["searcher_matrix"].parent.mkdir(parents=True, exist_ok=True)
    row_coph.to_csv(outputs["searcher_matrix"])
    col_coph.to_csv(outputs["findee_matrix"])
    (
        points[columns["entity"]]
        .value_counts()
        .rename_axis("entity")
        .rename("n_points")
        .to_csv(outputs["entity_counts"])
    )

    sample = str(config["region"])
    cellgps.plot_cophenetic_heatmap(
        row_coph,
        matrix_name="row_coph",
        output_dir=str(outputs["searcher_plot"].parent),
        output_filename=outputs["searcher_plot"].name,
        sample=sample,
    )
    cellgps.plot_cophenetic_heatmap(
        col_coph,
        matrix_name="col_coph",
        output_dir=str(outputs["findee_plot"].parent),
        output_filename=outputs["findee_plot"].name,
        sample=sample,
    )

    metadata = {
        "method": "cellgps",
        "cellgps_version": getattr(cellgps, "__version__", "unknown"),
        "tested_version": config["tested_version"],
        "region": config["region"],
        "input_kind": config["input_kind"],
        "entity_kind": config["entity_kind"],
        "coordinate_unit": config["coordinate_unit"],
        "n_points": int(len(points)),
        "n_entities": int(points[columns["entity"]].nunique()),
        "matrix_semantics": {
            "searcher": "normalized row cophenetic distance",
            "findee": "normalized column cophenetic distance",
        },
        "analysis": analysis,
        "input_summary": config["input_summary"],
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: _cellgps_runner.py CONFIG.json")
    run(sys.argv[1])

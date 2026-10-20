"""Adapters for CellGPS/COSTE spatial-topology analysis.

CellGPS compares labelled spatial point sets.  In this integration the points
can be either cell centroids from a spatial AnnData object or decoded
transcripts labelled by gene.  Neither workflow creates a cell-instance mask.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .segmentation_wrappers import (
    Command,
    PathLike,
    TableLike,
    _as_command,
    _run_command,
    normalize_transcript_table,
)


CELLGPS_TESTED_VERSION = "0.0.6"


def _cellgps_output_dir(
    *,
    region: str,
    input_kind: str,
    input_dir: PathLike | None,
    output_dir: PathLike | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if input_dir is None:
        raise ValueError("Pass input_dir or an explicit output_dir")
    return (
        Path(input_dir)
        / region
        / "postprocessing"
        / "spatial_topology"
        / "cellgps"
        / input_kind
    )


def _cellgps_paths(output_dir: Path, region: str, input_kind: str) -> dict[str, Path]:
    stem = f"{region}_cellgps_{input_kind}"
    return {
        "output_dir": output_dir,
        "points": output_dir / f"{stem}_points.csv",
        "config": output_dir / f"{stem}_config.json",
        "entity_counts": output_dir / f"{stem}_entity_counts.csv",
        "searcher_matrix": output_dir / f"{stem}_searcher_structuremap.csv",
        "findee_matrix": output_dir / f"{stem}_findee_structuremap.csv",
        "searcher_plot": output_dir / f"{stem}_searcher_structuremap.pdf",
        "findee_plot": output_dir / f"{stem}_findee_structuremap.pdf",
        "metadata": output_dir / f"{stem}_metadata.json",
    }


def _validate_analysis_settings(
    *,
    coordinate_unit: str,
    linkage_method: str,
    batch_size: int | None,
) -> None:
    if str(coordinate_unit).lower() not in {"pixel", "um"}:
        raise ValueError("coordinate_unit must be 'pixel' or 'um'")
    valid_methods = {
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward",
    }
    if linkage_method not in valid_methods:
        raise ValueError(
            f"Unsupported linkage_method {linkage_method!r}; choose one of "
            f"{sorted(valid_methods)}"
        )
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0 when provided")


def _write_cellgps_config(
    *,
    paths: dict[str, Path],
    region: str,
    input_kind: str,
    entity_kind: str,
    coordinate_unit: str,
    linkage_method: str,
    batch_size: int | None,
    show_corr: bool,
    input_summary: dict[str, Any],
) -> None:
    config = {
        "method": "cellgps",
        "tested_version": CELLGPS_TESTED_VERSION,
        "region": str(region),
        "input_kind": input_kind,
        "entity_kind": entity_kind,
        "coordinate_unit": str(coordinate_unit).lower(),
        "input_csv": str(paths["points"].resolve()),
        "columns": {"x": "x", "y": "y", "entity": "celltype"},
        "analysis": {
            "linkage_method": linkage_method,
            "batch_size": None if batch_size is None else int(batch_size),
            "show_corr": bool(show_corr),
        },
        "outputs": {
            name: str(paths[name].resolve())
            for name in (
                "entity_counts",
                "searcher_matrix",
                "findee_matrix",
                "searcher_plot",
                "findee_plot",
                "metadata",
            )
        },
        "input_summary": input_summary,
    }
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")


def _filter_entities(
    points: pd.DataFrame,
    *,
    minimum: int,
    minimum_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if int(minimum) <= 0:
        raise ValueError(f"{minimum_name} must be > 0")
    counts = points["celltype"].value_counts(sort=False)
    keep = counts[counts >= int(minimum)].index
    filtered = points.loc[points["celltype"].isin(keep)].copy()
    retained_counts = filtered["celltype"].value_counts().sort_index()
    if len(retained_counts) < 2:
        raise ValueError(
            "CellGPS requires at least two spatial entities after filtering; "
            f"only {len(retained_counts)} remain"
        )
    return filtered.reset_index(drop=True), retained_counts


def _load_anndata(adata_or_path: Any):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("AnnData input requires the anndata package") from exc

    if isinstance(adata_or_path, (str, os.PathLike)):
        path = Path(adata_or_path)
        if not path.is_file():
            raise FileNotFoundError(f"AnnData file not found: {path}")
        return ad.read_h5ad(path)
    if isinstance(adata_or_path, ad.AnnData):
        return adata_or_path
    raise TypeError("adata_or_path must be an AnnData object or an .h5ad path")


def prepare_cellgps_from_h5ad(
    adata_or_path: Any,
    *,
    region: str,
    cluster_col: str,
    coordinate_key: str = "spatial",
    coordinate_unit: str = "pixel",
    input_dir: PathLike | None = None,
    output_dir: PathLike | None = None,
    min_cells_per_group: int = 2,
    linkage_method: str = "average",
    batch_size: int | None = 100_000,
    show_corr: bool = True,
) -> dict[str, Path]:
    """Prepare cell centroids and annotations from a spatial AnnData object.

    The first two columns of ``adata.obsm[coordinate_key]`` are used as x/y.
    Cells without a cluster label are omitted; the source object is not changed.
    """
    _validate_analysis_settings(
        coordinate_unit=coordinate_unit,
        linkage_method=linkage_method,
        batch_size=batch_size,
    )
    adata = _load_anndata(adata_or_path)
    if cluster_col not in adata.obs.columns:
        raise KeyError(f"AnnData obs does not contain cluster column {cluster_col!r}")
    if coordinate_key not in adata.obsm:
        raise KeyError(f"AnnData obsm does not contain coordinates {coordinate_key!r}")

    raw_coords = adata.obsm[coordinate_key]
    if hasattr(raw_coords, "toarray"):
        raw_coords = raw_coords.toarray()
    elif hasattr(raw_coords, "to_numpy"):
        raw_coords = raw_coords.to_numpy()
    coords = np.asarray(raw_coords)
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{coordinate_key!r}] must have shape (n_obs, >=2); "
            f"found {coords.shape}"
        )
    xy = coords[:, :2].astype(float, copy=False)
    if not np.isfinite(xy).all():
        raise ValueError("AnnData spatial coordinates must be finite")

    labels = adata.obs[cluster_col]
    valid = labels.notna().to_numpy()
    points = pd.DataFrame(
        {
            "x": xy[valid, 0],
            "y": xy[valid, 1],
            "celltype": labels.loc[valid].astype(str).to_numpy(),
        }
    )
    points = points.loc[points["celltype"].str.strip().ne("")].copy()
    points, counts = _filter_entities(
        points,
        minimum=min_cells_per_group,
        minimum_name="min_cells_per_group",
    )

    destination = _cellgps_output_dir(
        region=region,
        input_kind="cells",
        input_dir=input_dir,
        output_dir=output_dir,
    )
    destination.mkdir(parents=True, exist_ok=True)
    paths = _cellgps_paths(destination, region, "cells")
    points.to_csv(paths["points"], index=False)
    _write_cellgps_config(
        paths=paths,
        region=region,
        input_kind="h5ad",
        entity_kind="cell_group",
        coordinate_unit=coordinate_unit,
        linkage_method=linkage_method,
        batch_size=batch_size,
        show_corr=show_corr,
        input_summary={
            "n_input_cells": int(adata.n_obs),
            "n_retained_cells": int(len(points)),
            "n_entities": int(len(counts)),
            "cluster_col": cluster_col,
            "coordinate_key": coordinate_key,
            "min_cells_per_group": int(min_cells_per_group),
        },
    )
    return paths


def prepare_cellgps_from_transcripts(
    transcripts: TableLike,
    *,
    region: str,
    input_dir: PathLike | None = None,
    output_dir: PathLike | None = None,
    gene_col: str = "target",
    x_col: str = "xc",
    y_col: str = "yc",
    quality_col: str | None = None,
    min_quality: float | None = None,
    genes: Sequence[str] | None = None,
    min_transcripts_per_gene: int = 2,
    coordinate_unit: str = "pixel",
    linkage_method: str = "average",
    batch_size: int | None = 100_000,
    show_corr: bool = True,
) -> dict[str, Path]:
    """Prepare decoded transcripts for segmentation-free gene topology.

    Gene names become the CellGPS point labels, so the resulting StructureMap
    compares transcript distributions rather than cell types.
    """
    _validate_analysis_settings(
        coordinate_unit=coordinate_unit,
        linkage_method=linkage_method,
        batch_size=batch_size,
    )
    table = normalize_transcript_table(
        transcripts,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        quality_col=quality_col,
        min_quality=min_quality,
    )
    n_input = len(table)
    if genes is not None:
        selected = {str(gene) for gene in genes}
        if not selected:
            raise ValueError("genes cannot be empty when provided")
        table = table.loc[table["gene"].isin(selected)].copy()
        if table.empty:
            raise ValueError("None of the selected genes occur in the transcript table")

    points = table[["x", "y", "gene"]].rename(columns={"gene": "celltype"})
    points, counts = _filter_entities(
        points,
        minimum=min_transcripts_per_gene,
        minimum_name="min_transcripts_per_gene",
    )

    destination = _cellgps_output_dir(
        region=region,
        input_kind="transcripts",
        input_dir=input_dir,
        output_dir=output_dir,
    )
    destination.mkdir(parents=True, exist_ok=True)
    paths = _cellgps_paths(destination, region, "transcripts")
    points.to_csv(paths["points"], index=False)
    _write_cellgps_config(
        paths=paths,
        region=region,
        input_kind="transcript_table",
        entity_kind="gene",
        coordinate_unit=coordinate_unit,
        linkage_method=linkage_method,
        batch_size=batch_size,
        show_corr=show_corr,
        input_summary={
            "n_input_transcripts": int(n_input),
            "n_retained_transcripts": int(len(points)),
            "n_entities": int(len(counts)),
            "min_transcripts_per_gene": int(min_transcripts_per_gene),
            "quality_col": quality_col,
            "min_quality": min_quality,
            "selected_genes": (
                None if genes is None else sorted({str(gene) for gene in genes})
            ),
        },
    )
    return paths


def _run_prepared_cellgps(
    paths: dict[str, Path],
    *,
    cellgps_python: Command,
) -> dict[str, Path]:
    runner = Path(__file__).with_name("_cellgps_runner.py")
    command = _as_command(cellgps_python) + [str(runner), str(paths["config"])]
    _run_command(command)
    required = [
        paths["entity_counts"],
        paths["searcher_matrix"],
        paths["findee_matrix"],
        paths["searcher_plot"],
        paths["findee_plot"],
        paths["metadata"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "CellGPS completed without producing required output(s): "
            + ", ".join(missing)
        )
    return paths


def run_cellgps_from_h5ad(
    adata_or_path: Any,
    *,
    region: str,
    cluster_col: str,
    input_dir: PathLike | None = None,
    output_dir: PathLike | None = None,
    cellgps_python: Command = sys.executable,
    overwrite: bool = False,
    **kwargs,
) -> dict[str, Path]:
    """Run CellGPS on spatially annotated cells from AnnData or an h5ad file."""
    destination = _cellgps_output_dir(
        region=region,
        input_kind="cells",
        input_dir=input_dir,
        output_dir=output_dir,
    )
    paths = _cellgps_paths(destination, region, "cells")
    if paths["metadata"].is_file() and not overwrite:
        print(f"[INFO] Reusing existing CellGPS output: {paths['metadata']}")
        return paths
    paths = prepare_cellgps_from_h5ad(
        adata_or_path,
        region=region,
        cluster_col=cluster_col,
        input_dir=input_dir,
        output_dir=output_dir,
        **kwargs,
    )
    return _run_prepared_cellgps(paths, cellgps_python=cellgps_python)


def run_cellgps_from_transcripts(
    transcripts: TableLike,
    *,
    region: str,
    input_dir: PathLike | None = None,
    output_dir: PathLike | None = None,
    cellgps_python: Command = sys.executable,
    overwrite: bool = False,
    **kwargs,
) -> dict[str, Path]:
    """Run segmentation-free CellGPS topology on decoded transcript points."""
    destination = _cellgps_output_dir(
        region=region,
        input_kind="transcripts",
        input_dir=input_dir,
        output_dir=output_dir,
    )
    paths = _cellgps_paths(destination, region, "transcripts")
    if paths["metadata"].is_file() and not overwrite:
        print(f"[INFO] Reusing existing CellGPS output: {paths['metadata']}")
        return paths
    paths = prepare_cellgps_from_transcripts(
        transcripts,
        region=region,
        input_dir=input_dir,
        output_dir=output_dir,
        **kwargs,
    )
    return _run_prepared_cellgps(paths, cellgps_python=cellgps_python)


__all__ = [
    "CELLGPS_TESTED_VERSION",
    "prepare_cellgps_from_h5ad",
    "prepare_cellgps_from_transcripts",
    "run_cellgps_from_h5ad",
    "run_cellgps_from_transcripts",
]

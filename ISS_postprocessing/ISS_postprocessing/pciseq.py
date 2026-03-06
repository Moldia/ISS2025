import pciSeq
from pciSeq import utils, fit

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib as mpl

# Core
import os
from pathlib import Path

# Data
import numpy as np
import pandas as pd

# Sparse I/O
from scipy.sparse import load_npz

# Plotting
import matplotlib.pyplot as plt


def preprocess_spots(spots_file, conversion_factor=0.1625):
    """
    Preprocess ISS spots for PCIseq.

    Parameters
    ----------
    spots_file : str, Path, or pd.DataFrame
        Path to decoded spots CSV or a DataFrame with spot data.
    conversion_factor : float, default=0.1625
        Conversion factor from microns to pixels.

    Returns
    -------
    pd.DataFrame
        Preprocessed spots with columns: Gene, x, y
    """
    # Load if path is provided
    if isinstance(spots_file, (str, Path)):
        spots = pd.read_csv(spots_file)
    else:
        spots = spots_file.copy()

    # Drop missing rows
    spots = spots.dropna()

    # Keep only required columns
    spots = spots[['target', 'xc', 'yc']].copy()

    # Rename to standard columns
    spots = spots.rename(columns={'target': 'Gene', 'xc': 'x', 'yc': 'y'})

    # Convert microns → pixels
    spots['x'] = spots['x'] / conversion_factor
    spots['y'] = spots['y'] / conversion_factor

    return spots




def preprocess_inputs(
    input_dir,
    region,
    segmentation_method,
    segmentation_file=None,
    scRNAseq,
    dense=True,
    quality_threshold=0.5,
    conversion_factor=1.0,
    plot=True,
):
    """
    Preprocess ISS spots and segmentation mask for PCIseq analysis.

    This version keeps the standard dataset layout based on `input_dir` and `region`,
    but also allows the user to override the segmentation path by providing
    `segmentation_file`.

    Parameters
    ----------
    input_dir : str or Path
        Base directory of the dataset.
    region : str
        Region identifier, e.g. "R1".
    segmentation_method : str
        Name/label of the segmentation used for default path construction and logging.
        Example: "cellpose", "stardist"
    scRNAseq : pd.DataFrame
        Reference scRNAseq expression matrix.
    segmentation_file : str or Path or None, default=None
        Optional full path to a segmentation `.npz` file.
        If provided, this path is used directly.
        If None, the default path is constructed as:
        <input_dir>/<region>/postprocessing/segmentation/{region}_{segmentation_method}_expanded.npz
    dense : bool, default=True
        Whether to read decoded spots from `2_decoded_dense` or `2_decoded`.
    quality_threshold : float, default=0.5
        Minimum quality threshold for keeping decoded spots.
    conversion_factor : float, default=1.0
        Factor used to convert spot coordinates into pixel units.
    plot : bool, default=True
        If True, show segmentation/spot overlay plot.

    Returns
    -------
    coo : scipy sparse matrix
        Segmentation mask loaded from `.npz`.
    processed_spots_clean : pd.DataFrame
        Filtered ISS spots containing only overlapping genes.
    scrnaseq_clean : pd.DataFrame
        Filtered scRNAseq matrix containing only overlapping genes.
    """

    # ------------------------------------------------------------------
    # Build the standard decoded-spots path from the dataset layout.
    # We keep this behavior unchanged because `input_dir` and `region`
    # still define where the decoded ISS results live.
    # ------------------------------------------------------------------
    decoded_dir = Path(input_dir) / region / "decoding" / f"2_decoded{'_dense' if dense else ''}"
    spots_file = decoded_dir / f"{region}_decoded.csv"

    # ------------------------------------------------------------------
    # Resolve segmentation path.
    #
    # New behavior:
    # - If `segmentation_file` is provided, use it directly.
    # - Otherwise, fall back to the old/default path-building logic.
    #
    # This keeps the pipeline convenient for the normal case while also
    # allowing custom storage locations for segmentation masks.
    # ------------------------------------------------------------------
    if segmentation_file is not None:
        coo_file = Path(segmentation_file)
    else:
        segmentation_dir = Path(input_dir) / region / "postprocessing" / "segmentation"
        coo_file = segmentation_dir / f"{region}_{segmentation_method}_expanded.npz"

    # ------------------------------------------------------------------
    # Validate that input files actually exist before trying to load them.
    # This gives a much clearer error than failing later inside load/read.
    # ------------------------------------------------------------------
    if not coo_file.exists():
        raise FileNotFoundError(
            f"Segmentation file not found: {coo_file}\n"
            f"Provided segmentation_method: {segmentation_method}\n"
            f"Provided segmentation_file: {segmentation_file}"
        )

    if not spots_file.exists():
        raise FileNotFoundError(f"Spots file not found: {spots_file}")

    # ------------------------------------------------------------------
    # Load segmentation mask and decoded spots.
    # ------------------------------------------------------------------
    coo = load_npz(coo_file)
    iss_spots = pd.read_csv(spots_file)

    print(f"Loaded segmentation mask: {coo_file}")
    print(f" - Shape: {coo.shape}")
    print(f"Loaded spots file: {spots_file}")
    print(f" - Spots: {len(iss_spots)}")

    # ------------------------------------------------------------------
    # Filter decoded spots by quality threshold.
    # ------------------------------------------------------------------
    if "quality_minimum" not in iss_spots.columns:
        raise KeyError("Column 'quality_minimum' not found in spots file.")

    spots_filt = iss_spots.loc[iss_spots["quality_minimum"] > quality_threshold].copy()

    # ------------------------------------------------------------------
    # Convert decoded spots into PCIseq-compatible format.
    # Expected output columns: Gene, x, y
    # ------------------------------------------------------------------
    processed_spots = preprocess_spots(spots_filt, conversion_factor=conversion_factor)

    # ------------------------------------------------------------------
    # Ensure scRNAseq orientation is genes x clusters.
    #
    # Current heuristic:
    # If the first column name is not found among spot genes, transpose.
    #
    # This is your existing logic, kept here to avoid changing behavior
    # beyond the requested segmentation-path update.
    # ------------------------------------------------------------------
    if scRNAseq.columns[0] not in processed_spots["Gene"].values:
        print("Detected scRNAseq genes in columns, transposing...")
        scRNAseq = scRNAseq.T

    # ------------------------------------------------------------------
    # Keep only genes shared between ISS spots and scRNAseq reference.
    # PCIseq should only be run on overlapping genes.
    # ------------------------------------------------------------------
    ISS_genes = list(processed_spots["Gene"].unique())
    scseq_genes = list(scRNAseq.index)

    overlap = sorted(set(scseq_genes).intersection(ISS_genes))
    print(f"Found {len(overlap)} overlapping genes.")
    if len(overlap) > 0:
        print("Example overlap genes:", overlap[:10])

    scrnaseq_clean = scRNAseq.loc[overlap, :]
    processed_spots_clean = processed_spots[processed_spots["Gene"].isin(overlap)]

    print(f"scrnaseq_clean shape: {scrnaseq_clean.shape}")
    print(f"processed_spots_clean shape: {processed_spots_clean.shape}")

    # ------------------------------------------------------------------
    # Optional sanity plot:
    # show segmentation mask with filtered/overlapping ISS spots.
    #
    # Note:
    # For plotting we convert the sparse mask to dense.
    # That is fine for visualization, but can be memory-heavy for very
    # large masks.
    # ------------------------------------------------------------------
    if plot:
        labels = coo.toarray().astype(np.int32)
        plt.figure(figsize=(6, 6))
        plt.imshow(labels, cmap="gray", alpha=0.6)
        plt.scatter(processed_spots_clean["x"], processed_spots_clean["y"], s=1, c="red")
        plt.title(f"{region} segmentation + ISS spots ({segmentation_method})")
        plt.axis("off")
        plt.show()

    return coo, processed_spots_clean, scrnaseq_clean


def get_most_probable_call_pciseq(cellData: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the most probable cell type assignment per cell from PCIseq output.

    Parameters
    ----------
    cellData : pd.DataFrame
        PCIseq cellData table containing columns:
        ['Cell_Num', 'X', 'Y', 'ClassName', 'Prob'].
        - ClassName: list of candidate cell types per cell
        - Prob: list of probabilities per cell

    Returns
    -------
    pd.DataFrame
        Table with one row per cell:
        ['Cell_Num', 'X', 'Y', 'ClassName', 'Prob'] (most probable only).
    """
    records = []

    for i, row in cellData.iterrows():
        cell = row["Cell_Num"]
        X, Y = row["X"], row["Y"]
        names, probs = row["ClassName"], row["Prob"]

        # Handle single vs list probabilities
        if not isinstance(probs, (list, tuple, np.ndarray)):
            probs = [probs]
            names = [names]

        max_idx = int(np.argmax(probs))
        records.append([cell, X, Y, names[max_idx], probs[max_idx]])

    return pd.DataFrame(records, columns=["Cell_Num", "X", "Y", "ClassName", "Prob"])



def run_pciseq(input_dir, region, spots, coo_mask, sc_expression_matrix, save_output=True):
    """
    Run PCIseq on a given region and save results to disk.

    Parameters
    ----------
    input_dir : str or Path
        Base directory of your dataset.
    region : str
        Region identifier (e.g., "R1").
    spots : pd.DataFrame
        ISS spots DataFrame with columns ['Gene','x','y'] (floats or ints).
    coo_mask : scipy.sparse.coo_matrix
        Sparse segmentation mask (labels).
    sc_expression_matrix : pd.DataFrame
        Reference scRNAseq expression matrix (genes x clusters).
    save_output : bool, default=True
        If True, writes results to <input_dir>/<region>/postprocessing/PCIseq/.

    Returns
    -------
    cellData : pd.DataFrame
    geneData : pd.DataFrame
    most_probable : pd.DataFrame
    """

    # --- Ensure PCIseq directory exists ---
    PCIseq_dir = Path(input_dir) / region / "postprocessing" / "PCIseq"
    PCIseq_dir.mkdir(parents=True, exist_ok=True)

    # --- Fix inputs ---
    # 1. segmentation mask: coo → csr, uint32
    coo_csr = coo_mask.tocsr().astype(np.uint32)

    # 2. spots: enforce int pixel coords
    spots = spots.copy()
    spots["x"] = np.rint(spots["x"]).astype(np.int32)
    spots["y"] = np.rint(spots["y"]).astype(np.int32)

    # --- Run PCIseq ---
    print(f"Running PCIseq on region {region} with {len(spots)} spots…")
    cellData, geneData = fit(
        spots,
        coo_csr,
        scRNAseq=sc_expression_matrix
    )
    cellData = cellData.reset_index()

    # Get most probable calls
    most_probable = get_most_probable_call_pciseq(cellData)

    # --- Save output ---
    if save_output:
        cellData.to_json(PCIseq_dir / "cellData.json")
        geneData.to_json(PCIseq_dir / "geneData.json")
        most_probable.to_csv(PCIseq_dir / "most_probable.csv", index=False)
        print(f"PCIseq results saved in: {PCIseq_dir}")

    return cellData, geneData, most_probable

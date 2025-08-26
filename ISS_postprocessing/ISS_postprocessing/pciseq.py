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


def preprocess_inputs(input_dir, region, segmentation_method, scRNAseq, 
                      dense=True, quality_threshold=0.5, conversion_factor=1.0, plot=True):
    """
    Preprocess ISS spots and segmentation mask for PCIseq analysis.
    Ensures scRNAseq matrix is in the correct orientation (genes x clusters).
    """

    # --- Build input paths ---
    segmentation_dir = Path(input_dir) / region / "postprocessing" / "segmentation"
    decoded_dir = Path(input_dir) / region / "decoding" / f"2_decoded{'_dense' if dense else ''}"

    coo_file   = segmentation_dir / f"{region}_{segmentation_method}_expanded.npz"
    spots_file = decoded_dir / f"{region}_decoded.csv"

    # --- Load inputs ---
    coo = load_npz(coo_file)               # sparse segmentation
    labels = coo.toarray().astype(np.int32)  # dense labels
    iss_spots = pd.read_csv(spots_file)    # decoded ISS spots

    print(f"Loaded segmentation mask: {coo_file}")
    print(f" - Shape: {labels.shape}, {labels.max()} cells")
    print(f"Loaded spots file: {spots_file}")
    print(f" - Spots: {len(iss_spots)}")

    # --- 1. Filter decoded ISS spots by quality threshold ---
    spots_filt = iss_spots.loc[iss_spots['quality_minimum'] > quality_threshold]

    # --- 2. Convert coordinates to pixel units ---
    processed_spots = preprocess_spots(spots_filt, conversion_factor=conversion_factor)

    # --- 3. Ensure scRNAseq orientation (genes x clusters) ---
    if scRNAseq.columns[0] not in processed_spots['Gene'].values:
        print("Detected scRNAseq genes in columns, transposing...")
        scRNAseq = scRNAseq.T

    # --- 4. Select overlapping gene set ---
    ISS_genes = list(processed_spots['Gene'].unique())
    scseq_genes = list(scRNAseq.index)

    overlap = sorted(set(scseq_genes).intersection(ISS_genes))
    print(f"Found {len(overlap)} overlapping genes.")
    if len(overlap) > 0:
        print("Example overlap genes:", overlap[:10])

    # --- 5. Filter both datasets to only shared genes ---
    scrnaseq_clean = scRNAseq.loc[overlap, :]
    processed_spots_clean = processed_spots[processed_spots['Gene'].isin(overlap)]

    print(f"scrnaseq_clean shape: {scrnaseq_clean.shape}")
    print(f"processed_spots_clean shape: {processed_spots_clean.shape}")

    # --- 6. Optional sanity plot ---
    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(labels, cmap="gray", alpha=0.6)
        plt.scatter(processed_spots['x'], processed_spots['y'], s=1, c='red')
        plt.title(f"{region} segmentation + ISS spots")
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
    spots["x"] = spots["x"].astype(int)
    spots["y"] = spots["y"].astype(int)

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

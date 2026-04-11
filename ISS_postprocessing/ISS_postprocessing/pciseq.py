from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

from pciSeq import fit, config


def _compute_bounds_metrics(spots_df, mask_shape):
    """
    Compute coordinate-vs-mask fit metrics.

    Parameters
    ----------
    spots_df : pd.DataFrame
        DataFrame with columns 'x' and 'y'.
    mask_shape : tuple[int, int]
        Segmentation mask shape as (height, width).

    Returns
    -------
    dict
        Dictionary with coordinate ranges and in-bounds fractions.
    """
    height, width = mask_shape

    x_min = float(spots_df["x"].min())
    x_max = float(spots_df["x"].max())
    y_min = float(spots_df["y"].min())
    y_max = float(spots_df["y"].max())

    frac_x_in_bounds = ((spots_df["x"] >= 0) & (spots_df["x"] < width)).mean()
    frac_y_in_bounds = ((spots_df["y"] >= 0) & (spots_df["y"] < height)).mean()
    frac_in_bounds = (
        (spots_df["x"] >= 0)
        & (spots_df["x"] < width)
        & (spots_df["y"] >= 0)
        & (spots_df["y"] < height)
    ).mean()

    return {
        "height": int(height),
        "width": int(width),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "frac_x_in_bounds": float(frac_x_in_bounds),
        "frac_y_in_bounds": float(frac_y_in_bounds),
        "frac_in_bounds": float(frac_in_bounds),
    }


def _print_bounds_metrics(metrics, title="[INFO] Coordinate vs mask summary"):
    """
    Pretty-print coordinate-vs-mask metrics.
    """
    print(title)
    print(f" - Mask shape: height={metrics['height']}, width={metrics['width']}")
    print(f" - Spot x range: {metrics['x_min']:.3f} to {metrics['x_max']:.3f}")
    print(f" - Spot y range: {metrics['y_min']:.3f} to {metrics['y_max']:.3f}")
    print(f" - Fraction of spots in x bounds: {metrics['frac_x_in_bounds']:.2%}")
    print(f" - Fraction of spots in y bounds: {metrics['frac_y_in_bounds']:.2%}")
    print(f" - Fraction of spots fully inside mask bounds: {metrics['frac_in_bounds']:.2%}")


def preprocess_spots(spots_file, pixel_to_um=1.0, mask_shape=None):
    """
    Preprocess ISS spots for PCIseq.

    Parameters
    ----------
    spots_file : str, Path, or pd.DataFrame
        Path to a decoded spots CSV file or a DataFrame with spot data.
    pixel_to_um : float, default=1.0
        Physical size of one image pixel in micrometers per pixel (µm/pixel).

        Interpretation:
        - If spot coordinates are already in pixels, use `pixel_to_um=1.0`
          so that no conversion is applied.
        - If spot coordinates are in microns, they are converted to pixels as:

              pixels = microns / pixel_to_um

    mask_shape : tuple[int, int] or None, default=None
        Optional segmentation mask shape as (height, width).
        If provided, additional sanity prints compare spot coordinates to the
        mask before and after conversion.

    Returns
    -------
    pd.DataFrame
        Preprocessed spot table with columns: `Gene`, `x`, `y`.

    Notes
    -----
    The input is expected to contain the columns:
    `target`, `xc`, `yc`

    This function preserves the behavior of the old working code:
    - keep only the required columns
    - rename them to PCIseq-compatible names
    - convert coordinates into pixel units
    """
    if pixel_to_um <= 0:
        raise ValueError(f"pixel_to_um must be > 0, got {pixel_to_um}")

    # ------------------------------------------------------------------
    # Load spot table if a file path is provided.
    # Otherwise, work on a copy of the provided DataFrame.
    # ------------------------------------------------------------------
    if isinstance(spots_file, (str, Path)):
        spots = pd.read_csv(spots_file)
    else:
        spots = spots_file.copy()

    # ------------------------------------------------------------------
    # Validate required columns.
    # ------------------------------------------------------------------
    required_cols = {"target", "xc", "yc"}
    missing_cols = required_cols - set(spots.columns)
    if missing_cols:
        raise KeyError(f"Missing required spot columns: {sorted(missing_cols)}")

    # ------------------------------------------------------------------
    # Drop rows missing any of the required columns.
    # ------------------------------------------------------------------
    before_dropna = len(spots)
    spots = spots.dropna(subset=["target", "xc", "yc"]).copy()
    after_dropna = len(spots)

    if before_dropna != after_dropna:
        print(f"[INFO] Dropped {before_dropna - after_dropna} spots with missing target/xc/yc")

    # ------------------------------------------------------------------
    # Keep only required columns and rename to PCIseq format.
    # ------------------------------------------------------------------
    spots = spots[["target", "xc", "yc"]].copy()
    spots = spots.rename(columns={"target": "Gene", "xc": "x", "yc": "y"})

    # ------------------------------------------------------------------
    # Force numeric coordinates.
    # ------------------------------------------------------------------
    spots["x"] = pd.to_numeric(spots["x"], errors="coerce")
    spots["y"] = pd.to_numeric(spots["y"], errors="coerce")

    before_numeric_drop = len(spots)
    spots = spots.dropna(subset=["x", "y"]).copy()
    after_numeric_drop = len(spots)

    if before_numeric_drop != after_numeric_drop:
        print(f"[INFO] Dropped {before_numeric_drop - after_numeric_drop} spots with non-numeric x/y")

    # ------------------------------------------------------------------
    # Print coordinate ranges before conversion.
    # ------------------------------------------------------------------
    print("[INFO] Spot coordinate summary BEFORE conversion:")
    print(f" - x range: {spots['x'].min():.3f} to {spots['x'].max():.3f}")
    print(f" - y range: {spots['y'].min():.3f} to {spots['y'].max():.3f}")
    print(f" - pixel_to_um: {pixel_to_um}")

    if (spots["x"] < 0).any() or (spots["y"] < 0).any():
        print("[WARN] Some input spot coordinates are negative")

    if mask_shape is not None:
        raw_metrics = _compute_bounds_metrics(spots, mask_shape)
        _print_bounds_metrics(raw_metrics, title="[INFO] Raw coordinates compared to mask")

        if raw_metrics["frac_in_bounds"] > 0.999 and pixel_to_um != 1.0:
            print(
                "[WARN] Raw coordinates already fit the mask extremely well. "
                "This suggests the coordinates may already be in pixels, "
                "so applying pixel_to_um may be unnecessary."
            )

    # ------------------------------------------------------------------
    # Convert coordinates into pixel units.
    # ------------------------------------------------------------------
    if pixel_to_um == 1.0:
        print("[INFO] pixel_to_um=1.0 -> assuming spot coordinates are already in pixels")
    else:
        print(
            "[INFO] Applying conversion: assuming spot coordinates are in microns "
            "and converting to pixels via x / pixel_to_um"
        )

    spots["x"] = spots["x"] / pixel_to_um
    spots["y"] = spots["y"] / pixel_to_um

    # ------------------------------------------------------------------
    # Print coordinate ranges after conversion.
    # ------------------------------------------------------------------
    print("[INFO] Spot coordinate summary AFTER conversion:")
    print(f" - x range: {spots['x'].min():.3f} to {spots['x'].max():.3f}")
    print(f" - y range: {spots['y'].min():.3f} to {spots['y'].max():.3f}")

    if (spots["x"] < 0).any() or (spots["y"] < 0).any():
        print("[WARN] Some converted spot coordinates are negative")

    if mask_shape is not None:
        converted_metrics = _compute_bounds_metrics(spots, mask_shape)
        _print_bounds_metrics(converted_metrics, title="[INFO] Converted coordinates compared to mask")

        if converted_metrics["frac_in_bounds"] < 0.95:
            print(
                "[WARN] Fewer than 95% of spots fall inside the segmentation bounds. "
                "This strongly suggests a coordinate mismatch, wrong pixel_to_um, "
                "or shifted/cropped data."
            )

        if converted_metrics["x_max"] < 0.25 * converted_metrics["width"] and converted_metrics["y_max"] < 0.25 * converted_metrics["height"]:
            print(
                "[WARN] Spot coordinates occupy only a small corner of the mask. "
                "Possible over-shrinking due to wrong pixel_to_um."
            )

        if converted_metrics["x_max"] > 5 * converted_metrics["width"] or converted_metrics["y_max"] > 5 * converted_metrics["height"]:
            print(
                "[WARN] Spot coordinates are far larger than the mask dimensions. "
                "Possible missing conversion or wrong pixel_to_um."
            )

        if pixel_to_um != 1.0:
            if converted_metrics["frac_in_bounds"] < raw_metrics["frac_in_bounds"]:
                print(
                    "[WARN] After applying pixel_to_um, the coordinates fit the mask worse than before. "
                    "That suggests the spots may already be in pixels, or pixel_to_um is not correct."
                )
            elif converted_metrics["frac_in_bounds"] > raw_metrics["frac_in_bounds"]:
                print(
                    "[INFO] After applying pixel_to_um, the coordinates fit the mask better than before."
                )
            else:
                print(
                    "[INFO] Applying pixel_to_um did not change the overall in-bounds fraction."
                )

    return spots


def prepare_pciseq_inputs(
    spots_file,
    coo_file,
    scRNAseq,
    pixel_to_um=1.0,
    quality_threshold=None,
    shuffle_spots=True,
    random_state=42,
    plot=True,
    region=None,
):
    """
    Prepare decoded ISS spots, segmentation mask, and scRNAseq reference
    for PCIseq analysis.

    This function is designed to preserve the behavior of the old working
    PCIseq script as closely as possible, while making the workflow more
    structured and reusable.

    Parameters
    ----------
    spots_file : str or Path
        Full path to a decoded spots CSV file.
    coo_file : str or Path
        Full path to a segmentation `.npz` file.
    scRNAseq : pd.DataFrame
        Reference scRNAseq expression matrix.

        In the old working code, this matrix was provided in the format:

            cell types/clusters x genes

        and then explicitly transposed before PCIseq. The same behavior is
        preserved here.
    pixel_to_um : float, default=1.0
        Physical size of one image pixel in micrometers per pixel (µm/pixel).

        - If spot coordinates are already in pixels, use `1.0`.
        - If spot coordinates are in microns, use the microscope pixel size
          (for example `0.95`) so that:

              pixels = microns / pixel_to_um
    quality_threshold : float or None, default=None
        If provided, filter spots by:

            quality_minimum > quality_threshold

        If None, no quality filtering is applied.

        This preserves compatibility with both:
        - already-filtered decoded spot files (e.g. reads_filt_min05.csv)
        - raw decoded spot files with QC columns
    shuffle_spots : bool, default=True
        If True, shuffle the final filtered spot table before PCIseq.

        This matches the old working code:
            sample(frac=1, random_state=42).reset_index(drop=True)
    random_state : int, default=42
        Random seed used when shuffling spots.
    plot : bool, default=True
        If True, show a segmentation/spot overlay plot as a sanity check.
    region : str or None, default=None
        Optional region label used only for plotting and logging.

    Returns
    -------
    coo : scipy.sparse.coo_matrix
        Segmentation mask in COO format with int32 dtype.
    processed_spots_clean : pd.DataFrame
        Final spot table restricted to genes overlapping with the scRNAseq
        reference. Columns are `Gene`, `x`, `y`.
    scrnaseq_clean : pd.DataFrame
        scRNAseq matrix restricted to overlapping genes, in PCIseq-compatible
        orientation:

            genes x cell types/clusters
    """
    # ------------------------------------------------------------------
    # Resolve explicit input file paths.
    # ------------------------------------------------------------------
    spots_file = Path(spots_file)
    segmentation_file = Path(coo_file)

    # ------------------------------------------------------------------
    # Validate that required input files exist before loading them.
    # ------------------------------------------------------------------
    if not spots_file.exists():
        raise FileNotFoundError(f"Spots file not found: {spots_file}")

    if not segmentation_file.exists():
        raise FileNotFoundError(f"Segmentation file not found: {segmentation_file}")

    # ------------------------------------------------------------------
    # Load segmentation mask and decoded spots.
    # ------------------------------------------------------------------
    iss_spots = pd.read_csv(spots_file)
    mask_sparse = load_npz(segmentation_file)
    coo = mask_sparse.tocoo().astype(np.int32)
    coo.sum_duplicates()

    print(f"Loaded segmentation mask: {segmentation_file}")
    print(f" - Shape: {coo.shape}")
    print(f"Loaded spots file: {spots_file}")
    print(f" - Spots: {len(iss_spots)}")

    # ------------------------------------------------------------------
    # Optional quality filtering.
    # ------------------------------------------------------------------
    if quality_threshold is not None:
        if "quality_minimum" not in iss_spots.columns:
            raise KeyError("Column 'quality_minimum' not found in spots file.")

        before = len(iss_spots)
        iss_spots = iss_spots.loc[iss_spots["quality_minimum"] > quality_threshold].copy()
        after = len(iss_spots)

        print(f"Applied quality filter: quality_minimum > {quality_threshold}")
        print(f" - Kept {after} / {before} spots ({after / before:.2%})")

    # ------------------------------------------------------------------
    # Convert decoded spots into PCIseq-compatible format.
    # Expected output columns: Gene, x, y
    # ------------------------------------------------------------------
    processed_spots = preprocess_spots(
        iss_spots,
        pixel_to_um=pixel_to_um,
        mask_shape=coo.shape,
    )

    # ------------------------------------------------------------------
    # Prepare scRNAseq reference.
    # ------------------------------------------------------------------
    scRNAseq = scRNAseq.copy()
    scRNAseq = scRNAseq.T
    scRNAseq = scRNAseq.apply(pd.to_numeric, errors="coerce").fillna(0)
    scRNAseq.index = scRNAseq.index.astype(str)
    scRNAseq.columns = scRNAseq.columns.astype(str)

    # ------------------------------------------------------------------
    # Keep only genes shared between ISS spots and scRNAseq reference.
    # ------------------------------------------------------------------
    iss_genes = list(processed_spots["Gene"].unique())
    scseq_genes = list(scRNAseq.index)

    overlap = sorted(set(scseq_genes).intersection(iss_genes))
    print(f"Found {len(overlap)} overlapping genes.")
    if len(overlap) > 0:
        print("Example overlap genes:", overlap[:10])
    else:
        raise ValueError("No overlapping genes found between spots and scRNAseq reference.")

    scrnaseq_clean = scRNAseq.loc[overlap, :].copy()
    processed_spots_clean = processed_spots[processed_spots["Gene"].isin(overlap)].copy()

    # ------------------------------------------------------------------
    # Remove exact duplicate spots (same Gene, x, y)
    # ------------------------------------------------------------------
    before = len(processed_spots_clean)
    processed_spots_clean = processed_spots_clean.drop_duplicates()
    after = len(processed_spots_clean)

    print(f"Removed {before - after} duplicate spots -> {after} remaining")

    # ------------------------------------------------------------------
    # Match old working behavior:
    # shuffle spots and reset index before running PCIseq.
    # ------------------------------------------------------------------
    if shuffle_spots:
        processed_spots_clean = (
            processed_spots_clean
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
    else:
        processed_spots_clean = processed_spots_clean.reset_index(drop=True)

    print(f"scrnaseq_clean shape: {scrnaseq_clean.shape}")
    print(f"processed_spots_clean shape: {processed_spots_clean.shape}")

    # ------------------------------------------------------------------
    # Sanity checks.
    # ------------------------------------------------------------------
    if not scrnaseq_clean.index.is_unique:
        raise ValueError("scRNAseq gene index is not unique after preprocessing.")

    if not scrnaseq_clean.columns.is_unique:
        raise ValueError("scRNAseq cell-type columns are not unique after preprocessing.")

    if scrnaseq_clean.isna().any().any():
        raise ValueError("scRNAseq contains NaN values after preprocessing.")

    if processed_spots_clean[["x", "y"]].isna().any().any():
        raise ValueError("Processed spots contain NaN coordinates after preprocessing.")

    # ------------------------------------------------------------------
    # Optional sanity plot:
    # show segmentation mask with overlapping ISS spots.
    # ------------------------------------------------------------------
    if plot:
        labels = coo.toarray().astype(np.int32)
        plt.figure(figsize=(6, 6))
        plt.imshow(labels, cmap="gray", alpha=0.6)
        plt.scatter(processed_spots_clean["x"], processed_spots_clean["y"], s=1, c="red")

        title = f"{region} segmentation + ISS spots" if region is not None else "Segmentation + ISS spots"
        plt.title(title)

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

        - `ClassName`: list of candidate cell types per cell
        - `Prob`: list of probabilities per cell

    Returns
    -------
    pd.DataFrame
        Table with one row per cell:
        ['Cell_Num', 'X', 'Y', 'ClassName', 'Prob']
        containing only the most probable assignment.
    """
    records = []

    for _, row in cellData.iterrows():
        cell = row["Cell_Num"]
        x = row["X"]
        y = row["Y"]
        names = row["ClassName"]
        probs = row["Prob"]

        if not isinstance(probs, (list, tuple, np.ndarray)):
            probs = [probs]
            names = [names]

        max_idx = int(np.argmax(probs))
        records.append([cell, x, y, names[max_idx], probs[max_idx]])

    return pd.DataFrame(records, columns=["Cell_Num", "X", "Y", "ClassName", "Prob"])


def run_pciseq(
    region,
    spots,
    coo_mask,
    sc_expression_matrix,
    output_dir_prefix,
    save_output=True,
    prob_threshold=None,
    retry_with_thinning=False,
    retry_spot_fraction=0.9999,
    retry_random_state=42,
):
    """
    Run PCIseq on a given region and optionally save results to disk.

    This function keeps the old PCIseq behavior while providing structured,
    reproducible output handling.

    Results are saved under:

        <output_dir_prefix>/<region>/PCIseq/
    """
    # ------------------------------------------------------------------
    # Output directory handling.
    # ------------------------------------------------------------------
    output_dir_prefix = Path(output_dir_prefix)
    output_dir_prefix.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using output_dir_prefix: {output_dir_prefix.resolve()}")

    PCIseq_dir = output_dir_prefix / region / "PCIseq"
    PCIseq_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Prepare inputs for PCIseq.
    # ------------------------------------------------------------------
    coo = coo_mask.tocoo().astype(np.int32)
    coo.sum_duplicates()

    spots_base = spots.copy()
    spots_base["x"] = spots_base["x"].astype(float)
    spots_base["y"] = spots_base["y"].astype(float)
    spots_base = spots_base.reset_index(drop=True)

    print("[INFO] Final PCIseq input summary:")
    print(f" - Segmentation shape: {coo.shape}")
    print(f" - Spots: {len(spots_base)}")
    print(f" - scRNAseq shape: {sc_expression_matrix.shape}")
    print(f" - Spot x range: {spots_base['x'].min():.3f} to {spots_base['x'].max():.3f}")
    print(f" - Spot y range: {spots_base['y'].min():.3f} to {spots_base['y'].max():.3f}")
    print(f" - Unique genes in spots: {spots_base['Gene'].nunique()}")

    if (spots_base["x"] < 0).any() or (spots_base["y"] < 0).any():
        print("[WARN] Negative spot coordinates detected before PCIseq")

    if spots_base[["x", "y"]].isna().any().any():
        raise ValueError("NaN spot coordinates detected before PCIseq")

    # ------------------------------------------------------------------
    # Configure pciSeq output path before calling fit().
    # ------------------------------------------------------------------
    config.DEFAULT["output_path"] = [str(output_dir_prefix / region)]

    # ------------------------------------------------------------------
    # Internal helper to run pciSeq on a given spot table.
    # ------------------------------------------------------------------
    def _run_fit(spots_to_use, label):
        print(f"Running PCIseq on region {region} with {len(spots_to_use)} spots ({label})...")
        return fit(
            spots=spots_to_use,
            coo=coo,
            scRNAseq=sc_expression_matrix,
        )

    # ------------------------------------------------------------------
    # First attempt: full dataset.
    # ------------------------------------------------------------------
    try:
        cellData, geneData = _run_fit(spots_base, label="full dataset")

    except AssertionError as e:
        known_msg = "The sum of the background spots and the total gene counts should be equal to the number of spots"

        if (not retry_with_thinning) or (known_msg not in str(e)):
            raise

        print("[WARN] pciSeq failed on full dataset with known assertion error.")
        print(
            f"[WARN] Retrying with spot thinning: "
            f"keep frac={retry_spot_fraction}, random_state={retry_random_state}"
        )

        spots_retry = (
            spots_base
            .sample(frac=retry_spot_fraction, random_state=retry_random_state)
            .reset_index(drop=True)
        )

        print(f"[INFO] Full spots: {len(spots_base)}")
        print(f"[INFO] Retry spots: {len(spots_retry)}")

        cellData, geneData = _run_fit(spots_retry, label="retry with thinning")

    # ------------------------------------------------------------------
    # Reset index for easier downstream handling.
    # ------------------------------------------------------------------
    cellData = cellData.reset_index()

    # ------------------------------------------------------------------
    # Extract most probable cell type per cell.
    # ------------------------------------------------------------------
    most_probable = get_most_probable_call_pciseq(cellData)

    # ------------------------------------------------------------------
    # Optional probability filtering.
    # ------------------------------------------------------------------
    if prob_threshold is not None:
        most_probable_filtered = most_probable.loc[
            most_probable["Prob"] > prob_threshold
        ].copy()

        print(f"Applied probability filter: Prob > {prob_threshold}")
        print(f" - Kept {len(most_probable_filtered)} / {len(most_probable)} cells")
    else:
        most_probable_filtered = None

    # ------------------------------------------------------------------
    # Save outputs (optional).
    # ------------------------------------------------------------------
    if save_output:
        cellData.to_json(PCIseq_dir / "cellData.json")
        geneData.to_json(PCIseq_dir / "geneData.json")
        most_probable.to_csv(PCIseq_dir / "most_probable.csv", index=False)

        if most_probable_filtered is not None:
            threshold_str = str(prob_threshold).replace(".", "p")
            most_probable_filtered.to_csv(
                PCIseq_dir / f"most_probable_p{threshold_str}.csv",
                index=False,
            )

        print(f"PCIseq results saved in: {PCIseq_dir}")

    return cellData, geneData, most_probable, most_probable_filtered
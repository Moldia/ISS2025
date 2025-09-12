# === Quiet setup (must be at very top; before any TF/StarDist import) ===
import os, contextlib, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # hide TF INFO+WARN
os.environ["TQDM_DISABLE"] = "1"            # hide tqdm bars (if supported)

@contextlib.contextmanager
def mute_fds():
    """Silence stdout/stderr at both Python and C level inside the block."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = os.dup(1), os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)  # C-level stdout
            os.dup2(devnull.fileno(), 2)  # C-level stderr
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        finally:
            os.dup2(old_out, 1); os.dup2(old_err, 2)
            os.close(old_out); os.close(old_err)

# ---- helpers (no TensorFlow dependencies) ----
def normalize_percentile(x, pmin=1, pmax=99.8, clip=True):
    lo, hi = np.percentile(x, (pmin, pmax))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x.astype(np.float32) - lo) / (hi - lo)
    return np.clip(y, 0, 1) if clip else y


# === Standard imports (safe; no TF pulled in) ===
from pathlib import Path
import sys, time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, save_npz, load_npz

from skimage import io as skio
from skimage import feature, measure, segmentation
from skimage.segmentation import expand_labels

from tifffile import imread, imwrite



# ========= StarDist segmentation (quiet, GPU-enabled) =========
def stardist_segmentation(
    input_dir,
    region,
    DAPI_ch: int = 4,
    model_name: str = "2D_versatile_fluo",
    expand_cells: bool = True,
    n_tiles: tuple[int, int] = (4, 4),
    expanded_distance: int = 20,
):
    print(f"\n\033[1mProcessing region: {region}\033[0m")

    # --- paths ---
    image_path = Path(input_dir) / region / "preprocessing" / "Cycle1" / "3_stitched" / f"Cycle1_ch{int(DAPI_ch)}.tif"
    print(f"DAPI image path: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"[ERROR] DAPI image not found: {image_path}")

    output_dir = Path(input_dir) / region / "postprocessing" / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load & normalize (no TF) ---
    image = imread(str(image_path))
    print("Image shape:", image.shape)
    print("Normalizing image (1–99.8 percentile)…")
    image_norm = normalize_percentile(image, 1, 99.8)

    # --- init StarDist quietly & warm up once ---
    print(f"Initializing StarDist model: {model_name}")
    with mute_fds():
        import tensorflow as tf
        from stardist.models import StarDist2D
        for g in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        model = StarDist2D.from_pretrained(model_name)
        _ = model.predict_instances(np.zeros((8, 8), np.float32), n_tiles=(1, 1))

    # --- predict ---
    print(f"Predicting instances (n_tiles={n_tiles})…")
    try:
        labels, details = model.predict_instances(image_norm, n_tiles=n_tiles, show_tile_progress=False)
    except TypeError:
        labels, details = model.predict_instances(image_norm, n_tiles=n_tiles)

    # --- relabel & save ---
    print("Relabeling…")
    labels = measure.label(labels)

    labels_npz = output_dir / f"{region}_stardist_labels.npz"
    print(f"Saving labels: {labels_npz}")
    save_npz(labels_npz, coo_matrix(labels), compressed=True)

    if expand_cells:
        print(f"Expanding labels by {expanded_distance} px…")
        expanded = expand_labels(labels, distance=expanded_distance)
        expanded_npz = output_dir / f"{region}_stardist_expanded.npz"
        print(f"Saving expanded labels (npz): {expanded_npz}")
        save_npz(expanded_npz, coo_matrix(expanded), compressed=True)

    print("Processing complete.\n")



# === Cellpose (on an image array) -> expanded labels + COO ===
def cell_pose_segmentation_to_coo(
        input_dir: str,
        region: str,
        DAPI_ch: int = 4,
        diameter: float | int | None = None,
        expanded_distance: int = 20):
    """
    Segment nuclei from a DAPI-stained image using Cellpose, refine with watershed,
    expand labels, and save outputs in {input_dir}/{region}/postprocessing/segmentation.
    """
    from cellpose import models

    print(f"\n\033[1mProcessing region: {region}\033[0m")

    # --- build input path ---
    image_path = Path(input_dir) / region / "preprocessing" / "Cycle1" / "3_stitched" / f"Cycle1_ch{int(DAPI_ch)}.tif"
    print(f"DAPI image path: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"[ERROR] DAPI image not found: {image_path}")

    # --- build output path ---
    output_dir = Path(input_dir) / region / "postprocessing" / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load image ---
    image = imread(str(image_path))
    print("Image shape:", image.shape)

    # --- init Cellpose model (v4+ uses pretrained_model; older uses model_type) ---
    print("Initializing Cellpose model (nuclei)…")
    with open(os.devnull, "w") as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        try:
            model = models.CellposeModel(gpu=True, pretrained_model="nuclei") # v4.0.1+
        except TypeError:
            model = models.CellposeModel(gpu=True, model_type="nuclei")       # pre-v4
  

    # --- run segmentation ---
    print(f"Running Cellpose segmentation (diameter={diameter})…")
    masks_nuclei, flows, styles = model.eval(image, diameter=diameter)

    # --- watershed refinement ---
    print("Refining segmentation with watershed…")
    distance = ndi.distance_transform_edt(masks_nuclei)
    local_max_coords = feature.peak_local_max(distance, min_distance=7)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    if local_max_coords.size:
        local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segmented_cells = segmentation.watershed(-distance, markers, mask=masks_nuclei)
    seg1 = measure.label(segmented_cells)

    # --- expand ---
    print(f"Expanding labels by {expanded_distance} pixels…")
    expanded = expand_labels(seg1, distance=expanded_distance)
    expanded_new = expanded.astype("uint32")

    # --- sparse & save ---
    coo = coo_matrix(expanded_new)

    # tif_out = output_dir / f"{region}_cellpose_expanded.tif"
    npz_out = output_dir / f"{region}_cellpose_expanded.npz"
    
    # print(f"Saving expanded labels to: {tif_out}")
    # imwrite(str(tif_out), expanded_new)
    
    print(f"Saving sparse matrix to: {npz_out}")
    save_npz(npz_out, coo, compressed=True)

    print("Processing complete.\n")
    
    return expanded_new, coo
    
# === Tile-wise segmentation + stitching ===
def segment_tile(
    sample_folder: str | Path,
    segment: bool = True,
    dapi_channel: int = 5,
    diam: int = 40,
    expanded_distance: int = 30,
    big_section: bool = False,
    output_file_name: str = "cellpose_segmentation.npz",
    expand_tile: bool = False,
):
    """
    Run Cellpose on resliced DAPI tiles and stitch into a single segmentation map.
    Saves per-tile .npz, per-column .npz, and final stitched .npz.
    """
    sample_folder = Path(sample_folder)
    output_path = sample_folder / "cell_segmentation"
    output_path.mkdir(parents=True, exist_ok=True)

    # Tiles directory: .../preprocessing/ReslicedTiles/Base_5_stitched-{dapi_channel}/
    tiles_dir = sample_folder / "preprocessing" / "ReslicedTiles" / f"Base_5_stitched-{dapi_channel}"
    if not tiles_dir.is_dir():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    tiles_segmented = {p.name for p in output_path.glob("*.npz")}
    tiles_to_segment = sorted([p for p in tiles_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    print("n tiles:", len(tiles_to_segment))

    if segment:
        print("segmenting tiles…")
        for tile_path in tiles_to_segment:
            out_name = tile_path.stem + ".npz"
            if out_name in tiles_segmented:
                continue
            print(tile_path.name)
            dapi = skio.imread(str(tile_path))
            expanded_labels, coo = cell_pose_segmentation_to_coo(dapi, diam=diam, expanded_distance=expanded_distance)
            save_npz(output_path / out_name, coo, compressed=True)
    else:
        print("not segmenting")

    # Stitch columns using tilepos.csv (assumes 1-indexed tile order by rows/cols)
    tiles_csv = sample_folder / "preprocessing" / "ReslicedTiles" / "tilepos.csv"
    tiles_df = pd.read_csv(tiles_csv, header=None)

    # Per unique column (tiles_df[1]), concatenate horizontally
    for col_id in tiles_df[1].unique():
        col_df = tiles_df[tiles_df[1] == col_id]
        tile_indices = list((col_df.index + 1).astype(str))  # "1", "2", ...
        mats = []
        for idx in tile_indices:
            mask = load_npz(output_path / f"tile{idx}.npz").toarray()
            if expand_tile:
                mask = expand_labels(mask, expanded_distance)
            mats.append(mask)
        concatenated = np.concatenate(tuple(mats), axis=1)
        save_npz(output_path / f"tiles_{col_id}.npz", coo_matrix(concatenated))

    # Build final by stacking columns vertically
    col_ids = list(tiles_df[1].unique())
    if big_section:
        print("splitting top/bottom (saved separately in variables, final still stitched together)")
        top_ids = col_ids[: round(len(col_ids) / 2)]
        bottom_ids = col_ids[round(len(col_ids) / 2) :]
        top = [load_npz(output_path / f"tiles_{cid}.npz").toarray() for cid in top_ids]
        bottom = [load_npz(output_path / f"tiles_{cid}.npz").toarray() for cid in bottom_ids]
        concatenated_top = np.concatenate(tuple(top), axis=0)
        concatenated_bottom = np.concatenate(tuple(bottom), axis=0)
        concatenated = np.concatenate((concatenated_top, concatenated_bottom), axis=0)
    else:
        print("not splitting")
        cols = [load_npz(output_path / f"tiles_{cid}.npz").toarray() for cid in col_ids]
        concatenated = np.concatenate(tuple(cols), axis=0)

    concatenated_relabeled = label(concatenated)
    save_npz(output_path / output_file_name, coo_matrix(concatenated_relabeled))

# === Utilities / plotting ===
def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))

def plot_segmentation_mask_colored(
    ad_sp,
    coo_file: str | Path,
    color_column: str,
    positions: tuple[int, int, int, int],
    output_file: str | Path,
):
    """
    Color a segmentation crop (positions=(r0, r1, c0, c1)) by per-cell RGB in ad_sp.obs[color_column].
    """
    from skimage.color import label2rgb

    coo = load_npz(str(coo_file))
    array = coo.toarray()

    r0, r1, c0, c1 = positions
    image_subset = array[r0:r1, c0:c1]

    # Prepare colors
    ad_sp.obs["CellID"] = list(ad_sp.obs["CellID"])
    ad_sp.obs["col_rgb"] = [hex_to_rgb(h) for h in ad_sp.obs[color_column]]

    filtered = ad_sp[ad_sp.obs["CellID"].astype(int).isin(image_subset.flatten())]
    lut = dict(zip(filtered.obs["CellID"].astype(int), filtered.obs["col_rgb"]))

    # Build color map for present labels
    max_label = int(image_subset.max()) if image_subset.size else 0
    colors = np.zeros((max_label + 1, 3), dtype=int)
    for k, v in lut.items():
        if k <= max_label:
            colors[k] = np.array(v)

    colored_image = colors[image_subset]

    with plt.rc_context({"figure.figsize": (20, 20)}):
        plt.imshow(colored_image)
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(str(output_file), dpi=600)
        plt.show()


"""
Segmentation overlay and contour mask extraction for ISS_postprocessing.

Loads sparse Stardist/Cellpose masks, overlays on raw image,
and extracts contour masks for QC/visualization.
"""
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from skimage.io import imread
from tifffile import imwrite
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.morphology import binary_dilation, disk



def load_sparse_mask(mask_file: Path) -> np.ndarray:
    """Load sparse mask stored as .npz into dense label image."""
    data = np.load(mask_file)
    row, col, vals, shape = data["row"], data["col"], data["data"], data["shape"]
    mask_sparse = coo_matrix((vals, (row, col)), shape=shape)
    labels = mask_sparse.toarray().astype(np.int32)
    print(f"Loaded segmentation mask from: {mask_file}")
    print(f" - Shape: {labels.shape}")
    print(f" - Unique labels: {len(np.unique(labels))}")
    return labels


def load_dapi_image(input_dir: Path, region: str, DAPI_ch: int) -> np.ndarray:
    """Load DAPI / fluorescence image for a region."""
    DAPI_file = Path(input_dir) / region / "preprocessing" / "Cycle1" / "3_stitched" / f"Cycle1_ch{int(DAPI_ch)}.tif"
    image_wh = imread(DAPI_file)
    print("Loaded image:", DAPI_file)
    print("Shape:", image_wh.shape, "dtype:", image_wh.dtype)
    return image_wh


def plot_segmentation_overlay(
    image, labels,
    crop_coords=None,
    brightness_factor=4,
    figsize=(8, 8), dpi=300,
    title="Segmentation Overlay (Brightened)"
):
    """Overlay segmentation labels on an image."""
    # Crop if requested
    if crop_coords:
        y_start, y_end, x_start, x_end = crop_coords
        if y_end > image.shape[0] or x_end > image.shape[1]:
            print(f"Crop larger than image. Showing full image instead.")
        else:
            image = image[y_start:y_end, x_start:x_end]
            labels = labels[y_start:y_end, x_start:x_end]
            print(f"Cropped image shape: {image.shape}")
    else:
        print(f"Using full image, shape: {image.shape}")

    # Normalize and brighten
    image_min, image_max = image.min(), image.max()
    image_range = image_max - image_min if image_max > image_min else 1
    image_norm = (image - image_min) / image_range
    image_bright = np.clip(image_norm * brightness_factor, 0, 1)

    # Overlay boundaries
    overlay = mark_boundaries(
        image_bright, labels,
        color=(1, 1, 0),
        mode="outer",
        background_label=0
    )

    # Plot
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.show()


def extract_contour_mask(labels: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Extract a contour mask from label image."""
    contour = find_boundaries(labels, mode="outer")
    thick_contour = binary_dilation(contour, footprint=disk(thickness))
    contour_image = (thick_contour.astype(np.uint8)) * 255
    return contour_image


def save_contour_mask(segmentation_dir: Path, region: str, segmentation_method: str, contour_image: np.ndarray):
    """Save contour mask as TIFF."""
    out_path = segmentation_dir / f"{region}_{segmentation_method}_contour_mask.tif"
    imwrite(out_path, contour_image.astype(np.uint8))
    print(f"Contour mask saved to: {out_path}")
    return out_path


# --- Example main function for running end-to-end ---
def inspect_and_work_with_segmentation(input_dir: str, region: str, segmentation_method: str, DAPI_ch: int = 4, crop_coords=None):
    segmentation_dir = Path(input_dir) / region / "postprocessing" / "segmentation"
    mask_file = segmentation_dir / f"{region}_{segmentation_method}_expanded.npz"

    labels = load_sparse_mask(mask_file)
    image_wh = load_dapi_image(input_dir, region, DAPI_ch)

    plot_segmentation_overlay(image_wh, labels, crop_coords=crop_coords)

    contour_image = extract_contour_mask(labels, thickness=3)
    save_contour_mask(segmentation_dir, region, segmentation_method, contour_image)




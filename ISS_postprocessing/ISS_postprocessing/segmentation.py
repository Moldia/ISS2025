# =============================================================================
# Simple Cellpose + StarDist segmentation utilities
#
# Main intended usage
# -------------------
# for region in regions:
#     SEG.cell_pose_segmentation_to_coo(
#         input_dir,
#         region,
#         DAPI_ch=4,
#         diameter=None,
#         expanded_distance=20
#     )
#
# What this does
# --------------
# Cellpose:
# - finds all retiled images in:
#       {input_dir}/{region}/preprocessing/Cycle1/4_retiled/
# - expected names:
#       Cycle1_s0_ch4.tif
#       Cycle1_s1_ch4.tif
#       ...
# - segments every tile for the selected DAPI channel
# - expands labels
# - offsets tile labels so IDs stay unique across tiles
# - stitches tiles into one final full-size segmentation using:
#       Cycle1_retiled_coords.csv
# - saves per-tile sparse masks and one final stitched sparse mask
#
# StarDist:
# - kept simple
# - runs on one image at a time, like the old code
#
# Notes
# -----
# - This keeps the old outcome, but modernizes path handling and stitching.
# - Final stitched labels are NOT re-labeled as binary connected components,
#   because that would merge objects and lose per-tile label identities.
# =============================================================================

# -----------------------------------------------------------------------------
# Top-level setup
# -----------------------------------------------------------------------------
import os
import re
import subprocess
import contextlib
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, save_npz, load_npz

from skimage import feature, measure, segmentation
from skimage.segmentation import expand_labels, mark_boundaries, find_boundaries
from skimage.morphology import binary_dilation, disk

from tifffile import imread, imwrite


# -----------------------------------------------------------------------------
# GPU selection
# -----------------------------------------------------------------------------
def choose_gpu(preferred_max_mem_mb=2000, preferred_max_util=20):
    """
    Choose a relatively free NVIDIA GPU on a shared server.

    This must run before importing GPU-heavy libraries that initialize CUDA.
    If it fails, the environment is left unchanged.
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,nounits,noheader",
            ],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()

        if not result:
            print("No GPUs reported by nvidia-smi.")
            return None

        rows = []
        for line in result.split("\n"):
            idx, mem, util = [x.strip() for x in line.split(",")]
            rows.append((int(idx), int(mem), int(util)))

        preferred = [
            row for row in rows
            if row[1] <= preferred_max_mem_mb and row[2] <= preferred_max_util
        ]

        if preferred:
            preferred.sort(key=lambda x: (x[1], x[2], x[0]))
            gpu_id = preferred[0][0]
        else:
            rows.sort(key=lambda x: (x[1], x[2], x[0]))
            gpu_id = rows[0][0]

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        RED_BOLD = "\033[1;31m"
        RED = "\033[31m"
        RESET = "\033[0m"

        print(f"{RED_BOLD}Selected physical GPU {gpu_id}{RESET}")
        print(f"{RED}CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}{RESET}")

        return gpu_id

    except FileNotFoundError:
        print("nvidia-smi not found. Leaving GPU selection unchanged.")
        return None
    except Exception as e:
        print(f"GPU auto-selection failed: {e}")
        return None


# Comment out if your scheduler already sets the GPU
choose_gpu(preferred_max_mem_mb=2000, preferred_max_util=20)


# -----------------------------------------------------------------------------
# Quiet helper
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def mute_fds():
    """
    Silence stdout/stderr at both Python and C-extension level.
    """
    with open(os.devnull, "w") as devnull:
        old_out, old_err = os.dup(1), os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        finally:
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(old_out)
            os.close(old_err)


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def get_retiled_dir(input_dir, region):
    """
    Return:
    {input_dir}/{region}/preprocessing/Cycle1/4_retiled/
    """
    return Path(input_dir) / region / "preprocessing" / "Cycle1" / "4_retiled"


def get_segmentation_dir(input_dir, region, output_dir_prefix=None):
    """
    Determine where segmentation outputs should be written.

    Default behavior (same as old pipeline):
        {input_dir}/{region}/postprocessing/segmentation/

    If output_dir_prefix is provided:
        {output_dir_prefix}/{region}/postprocessing/segmentation/
    """

    if output_dir_prefix is not None:
        output_dir = Path(output_dir_prefix) / region / "postprocessing" / "segmentation"
    else:
        output_dir = Path(input_dir) / region / "postprocessing" / "segmentation"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_coords_csv(input_dir, region):
    """
    Return:
    {input_dir}/{region}/preprocessing/Cycle1/4_retiled/Cycle1_retiled_coords.csv
    """
    csv_path = get_retiled_dir(input_dir, region) / "Cycle1_retiled_coords.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Retiled coords CSV not found: {csv_path}")
    return csv_path


def get_retiled_images(input_dir, region, DAPI_ch=4):
    """
    Find all retiled images for one channel.

    Expected names:
        Cycle1_s0_ch4.tif
        Cycle1_s1_ch4.tif
        Cycle1_s2_ch4.tif
        ...

    Returns
    -------
    list[Path]
        Sorted by section number s0, s1, s2, ...
    """
    retiled_dir = get_retiled_dir(input_dir, region)
    if not retiled_dir.is_dir():
        raise FileNotFoundError(f"Retiled directory not found: {retiled_dir}")

    pattern = re.compile(rf"^Cycle1_s(\d+)_ch{int(DAPI_ch)}\.tif$", re.IGNORECASE)

    images = []
    for p in retiled_dir.iterdir():
        if p.is_file() and pattern.match(p.name):
            images.append(p)

    images.sort(key=lambda p: int(pattern.match(p.name).group(1)))

    if not images:
        raise FileNotFoundError(
            f"No retiled images found for channel ch{int(DAPI_ch)} in {retiled_dir}"
        )

    return images


def read_retiled_coords(input_dir, region):
    """
    Read Cycle1_retiled_coords.csv.

    Expected format:
    - two numeric columns
    - one row per tile
    - row 0 corresponds to s0
    - row 1 corresponds to s1
    - ...

    Returns
    -------
    coords : np.ndarray of shape (N, 2)
        coords[:, 0] = x
        coords[:, 1] = y
    """
    coords_df = pd.read_csv(get_coords_csv(input_dir, region), header=None)

    if coords_df.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns in Cycle1_retiled_coords.csv, got {coords_df.shape[1]}"
        )

    coords = coords_df.iloc[:, :2].to_numpy()

    if not np.issubdtype(coords.dtype, np.number):
        raise ValueError("Coordinate CSV must contain numeric x,y values")

    return coords.astype(int)


# -----------------------------------------------------------------------------
# Normalization helper for StarDist
# -----------------------------------------------------------------------------
def normalize_percentile(x, pmin=1, pmax=99.8, clip=True):
    """
    Simple percentile normalization similar to csbdeep normalize(...).
    """
    lo, hi = np.percentile(x, (pmin, pmax))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x.astype(np.float32) - lo) / (hi - lo)
    return np.clip(y, 0, 1) if clip else y


# -----------------------------------------------------------------------------
# StarDist
# -----------------------------------------------------------------------------
def stardist_segmentation(
    input_dir,
    region,
    image_name,
    model_name="2D_versatile_fluo",
    expand_cells=True,
    n_tiles=(4, 4),
    expanded_distance=20,
):
    """
    Run StarDist on ONE image and save sparse labels / expanded labels.

    This keeps the old StarDist behavior: one image per call.
    """
    image_path = get_retiled_dir(input_dir, region) / image_name
    output_dir = get_segmentation_dir(input_dir, region, output_dir_prefix)

    print(f"\nProcessing region: {region}")
    print(f"Image: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = imread(str(image_path))
    print("Image shape:", image.shape)

    print("Normalizing image")
    image_norm = normalize_percentile(image, 1, 99.8)

    print(f"Initializing StarDist model: {model_name}")
    with mute_fds():
        import tensorflow as tf
        from stardist.models import StarDist2D

        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

        model = StarDist2D.from_pretrained(model_name)

    print("Running StarDist")
    try:
        labels, details = model.predict_instances(
            image_norm,
            n_tiles=n_tiles,
            show_tile_progress=False,
        )
    except TypeError:
        labels, details = model.predict_instances(image_norm, n_tiles=n_tiles)

    labels = measure.label(labels)

    stem = image_path.stem

    labels_out = output_dir / f"{region}_{stem}_stardist_labels.npz"
    save_npz(labels_out, coo_matrix(labels), compressed=True)

    if expand_cells:
        expanded = expand_labels(labels, distance=expanded_distance)
        expanded_out = output_dir / f"{region}_{stem}_stardist_expanded.npz"
        save_npz(expanded_out, coo_matrix(expanded), compressed=True)

    print("Done\n")


# -----------------------------------------------------------------------------
# Cellpose helpers
# -----------------------------------------------------------------------------
def build_cellpose_model():
    """
    Build Cellpose nuclei model once and reuse it across all tiles.

    This is simpler and faster than rebuilding it for every tile.
    """
    with mute_fds():
        from cellpose import models
        try:
            model = models.CellposeModel(gpu=True, pretrained_model="nuclei")
        except TypeError:
            model = models.CellposeModel(gpu=True, model_type="nuclei")
    return model


def segment_one_tile_with_cellpose(image, model, diameter=None, expanded_distance=20):
    """
    Segment one tile with Cellpose, then refine with watershed and expand labels.

    This stays very close to the old code behavior.

    Returns
    -------
    expanded_labels : np.ndarray
    coo : scipy.sparse.coo_matrix
    """
    masks_nuclei, flows, styles = model.eval(image, diameter=diameter)

    distance = ndi.distance_transform_edt(masks_nuclei > 0)

    local_max_coords = feature.peak_local_max(distance, min_distance=7)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    if local_max_coords.size:
        local_max_mask[tuple(local_max_coords.T)] = True

    markers = measure.label(local_max_mask)
    segmented_cells = segmentation.watershed(-distance, markers, mask=(masks_nuclei > 0))
    seg1 = measure.label(segmented_cells)

    expanded = expand_labels(seg1, distance=expanded_distance).astype(np.uint32)
    coo = coo_matrix(expanded)

    return expanded, coo


def offset_labels(mask, offset):
    """
    Offset nonzero labels so label IDs remain unique across tiles.

    Example:
    - tile 1 labels 1..300
    - tile 2 becomes 301..650
    """
    mask = mask.astype(np.uint32, copy=True)
    nonzero = mask > 0
    if nonzero.any():
        mask[nonzero] += np.uint32(offset)
        offset = int(mask.max())
    return mask, offset


def stitch_tiles_from_coords(tile_arrays, coords):
    """
    Stitch label tiles into one canvas using direct (x, y) placement.

    Parameters
    ----------
    tile_arrays : list[np.ndarray]
        Tile label images, already offset so IDs are unique.
    coords : np.ndarray of shape (N, 2)
        x,y coordinates for each tile.

    Returns
    -------
    stitched : np.ndarray
        Final stitched label image.

    Notes
    -----
    - Assumes coords[:,0] = x and coords[:,1] = y
    - Assumes all tiles have same shape
    - Nonzero labels are written into the stitched canvas
    - Existing nonzero labels are preserved if there is accidental overlap
    """
    if len(tile_arrays) != len(coords):
        raise ValueError(
            f"Number of tiles ({len(tile_arrays)}) does not match number of coords ({len(coords)})"
        )

    if len(tile_arrays) == 0:
        raise ValueError("No tiles to stitch")

    tile_h, tile_w = tile_arrays[0].shape

    xs = coords[:, 0]
    ys = coords[:, 1]

    canvas_w = int(xs.max() + tile_w)
    canvas_h = int(ys.max() + tile_h)

    stitched = np.zeros((canvas_h, canvas_w), dtype=np.uint32)

    for tile, (x, y) in zip(tile_arrays, coords):
        y0, y1 = int(y), int(y) + tile_h
        x0, x1 = int(x), int(x) + tile_w

        view = stitched[y0:y1, x0:x1]

        # Only write new labels into currently empty pixels.
        # This avoids zero-valued background wiping out existing labels.
        mask_new = (tile > 0) & (view == 0)
        view[mask_new] = tile[mask_new]

    return stitched


# -----------------------------------------------------------------------------
# Main Cellpose batch function
# -----------------------------------------------------------------------------
def cell_pose_segmentation_to_coo(
    input_dir,
    region,
    output_dir_prefix=None,
    DAPI_ch=4,
    diameter=None,
    expanded_distance=20,
):
    """
    Segment all retiled DAPI images in one region and stitch them into one final
    segmentation mask.

    This function is the region-level batch equivalent of the old workflow.

    Parameters
    ----------
    input_dir : str | Path
        Root folder containing region folders.
    region : str
        Region name, for example "R1".
    output_dir_prefix : str | Path | None, optional
        Optional base directory for segmentation outputs.

        If None, outputs are written to the default location under the region:
            {input_dir}/{region}/postprocessing/segmentation/

        If provided, outputs are written instead to:
            {output_dir_prefix}/{region}/segmentation/

        This is useful when you want to write results to a scratch disk,
        temporary workspace, or another output location instead of the
        original region directory.
    DAPI_ch : int, optional
        DAPI channel number, for example 4 for files named like
        `Cycle1_s0_ch4.tif`.
    diameter : int | float | None, optional
        Cellpose diameter parameter. Use None to let Cellpose estimate it.
    expanded_distance : int, optional
        Label expansion distance in pixels after watershed refinement.

    Outputs
    -------
    Saves:
    - one sparse `.npz` per tile:
        `{image_stem}_cellpose_tile.npz`
    - one final stitched sparse `.npz`:
        `{region}_cellpose_expanded.npz`

    Returns
    -------
    stitched : np.ndarray
        Final stitched label image.
    stitched_coo : scipy.sparse.coo_matrix
        Sparse version of the final stitched label image.
    """
    
    print(f"\n\033[1mProcessing region: {region}\033[0m")

    output_dir = get_segmentation_dir(input_dir, region)
    image_paths = get_retiled_images(input_dir, region, DAPI_ch=DAPI_ch)
    coords = read_retiled_coords(input_dir, region)

    print(f"Found {len(image_paths)} retiled images for channel ch{int(DAPI_ch)}")
    print(f"Found {len(coords)} coordinate rows")

    if len(image_paths) != len(coords):
        raise ValueError(
            f"Number of images ({len(image_paths)}) does not match coordinate rows ({len(coords)})"
        )

    print("Initializing Cellpose model (nuclei)")
    model = build_cellpose_model()

    # -------------------------------------------------------------------------
    # Step 1: segment each tile and save sparse output
    # -------------------------------------------------------------------------
    tile_arrays = []
    running_offset = 0

    for image_path in image_paths:
        print(f"Tile: {image_path.name}")

        image = imread(str(image_path))
        expanded_labels, tile_coo = segment_one_tile_with_cellpose(
            image=image,
            model=model,
            diameter=diameter,
            expanded_distance=expanded_distance,
        )

        # Make labels globally unique across tiles before stitching
        expanded_labels, running_offset = offset_labels(expanded_labels, running_offset)
        tile_arrays.append(expanded_labels)

        tile_out = output_dir / f"{image_path.stem}_cellpose_tile.npz"
        save_npz(tile_out, coo_matrix(expanded_labels), compressed=True)

    # -------------------------------------------------------------------------
    # Step 2: stitch tiles into one final full-size label image
    # -------------------------------------------------------------------------
    print("Stitching tiles from coordinate CSV")
    stitched = stitch_tiles_from_coords(tile_arrays, coords)

    # -------------------------------------------------------------------------
    # Step 3: save final stitched segmentation
    # -------------------------------------------------------------------------
    # Important:
    # We save the stitched labels directly, preserving unique IDs.
    # We do NOT run measure.label(stitched > 0), because that would merge
    # touching cells and lose original label identity.
    stitched_coo = coo_matrix(stitched)

    final_out = output_dir / f"{region}_cellpose_expanded.npz"
    save_npz(final_out, stitched_coo, compressed=True)

    print(f"Saved final stitched segmentation to: {final_out}\n")
    return stitched, stitched_coo


# -----------------------------------------------------------------------------
# Sparse mask utilities
# -----------------------------------------------------------------------------
def load_sparse_mask(mask_file):
    """
    Load a sparse segmentation mask saved with scipy.sparse.save_npz(...).
    """
    labels = load_npz(str(mask_file)).toarray().astype(np.int32)
    print(f"Loaded segmentation mask from: {mask_file}")
    print(f" - Shape: {labels.shape}")
    print(f" - Unique labels: {len(np.unique(labels))}")
    return labels


# -----------------------------------------------------------------------------
# Overlay / contour utilities
# -----------------------------------------------------------------------------
def plot_segmentation_overlay(
    image,
    labels,
    crop_coords=None,
    brightness_factor=4,
    figsize=(8, 8),
    dpi=300,
    title="Segmentation Overlay",
):
    """
    Overlay segmentation boundaries on an image.

    crop_coords format:
    (y_start, y_end, x_start, x_end)
    """
    if crop_coords:
        y_start, y_end, x_start, x_end = crop_coords
        if y_end > image.shape[0] or x_end > image.shape[1]:
            print("Crop larger than image. Showing full image instead.")
        else:
            image = image[y_start:y_end, x_start:x_end]
            labels = labels[y_start:y_end, x_start:x_end]
            print(f"Cropped image shape: {image.shape}")
    else:
        print(f"Using full image, shape: {image.shape}")

    image_min, image_max = image.min(), image.max()
    image_range = image_max - image_min if image_max > image_min else 1
    image_norm = (image - image_min) / image_range
    image_bright = np.clip(image_norm * brightness_factor, 0, 1)

    overlay = mark_boundaries(
        image_bright,
        labels,
        color=(1, 1, 0),
        mode="outer",
        background_label=0,
    )

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.show()


def extract_contour_mask(labels, thickness=2):
    """
    Extract a thick contour mask from a label image.
    """
    contour = find_boundaries(labels, mode="outer")
    thick_contour = binary_dilation(contour, footprint=disk(thickness))
    contour_image = (thick_contour.astype(np.uint8)) * 255
    return contour_image


def save_contour_mask(segmentation_dir, region, segmentation_method, contour_image):
    """
    Save contour mask as TIFF.
    """
    out_path = Path(segmentation_dir) / f"{region}_{segmentation_method}_contour_mask.tif"
    imwrite(out_path, contour_image.astype(np.uint8))
    print(f"Contour mask saved to: {out_path}")
    return out_path


def inspect_stitched_segmentation(
    input_dir,
    region,
    stitched_mask_name=None,
):
    """
    Load the final stitched segmentation and export a contour mask.

    By default, uses:
        {region}_cellpose_expanded.npz
    """
    segmentation_dir = get_segmentation_dir(input_dir, region, output_dir_prefix)

    if stitched_mask_name is None:
        stitched_mask_name = f"{region}_cellpose_expanded.npz"

    mask_file = segmentation_dir / stitched_mask_name
    labels = load_sparse_mask(mask_file)

    contour_image = extract_contour_mask(labels, thickness=3)
    save_contour_mask(segmentation_dir, region, "cellpose", contour_image)


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
# regions = ["R1"]
# input_dir = "/home/sagah/moldia_archive/Agustin/MicroRNA_Result_E1_R3_Deconvolved_V1_USER_Standard_Done/"
#
# for region in regions:
#     cell_pose_segmentation_to_coo(
#         input_dir,
#         region,
#         DAPI_ch=4,
#         diameter=None,
#         expanded_distance=20
#     )
#
# Example StarDist on one retiled image:
# stardist_segmentation(
#     input_dir=input_dir,
#     region="R1",
#     image_name="Cycle1_s0_ch4.tif",
#     model_name="2D_versatile_fluo",
#     expand_cells=True,
#     n_tiles=(4, 4),
#     expanded_distance=20,
# )
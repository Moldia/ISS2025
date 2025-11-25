

# --- Standard Library ---
import os
import re
import shutil
import subprocess
import time
import math
import warnings
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional, List

# --- Third-Party ---
import numpy as np
import pandas as pd
import tifffile
import cv2
import dask.array as da
from tqdm import tqdm
from natsort import natsorted
from aicspylibczi import CziFile
from readlif.reader import LifFile
import nd2

# --- Local Modules ---
import RedLionfishDeconv as rl
import ISS_preprocessing.psf as fd_psf
import ashlar.scripts.ashlar as ashlar

from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity

def convert_16bit_to_8bit_auto(image_16):
    # Stretch intensities based on actual data range
    image_rescaled = rescale_intensity(image_16, in_range='image', out_range='dtype')
    image_8 = img_as_ubyte(image_rescaled)
    return image_8


def custom_copy(src, dest):
    """Custom function to copy a file to a destination."""
    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(src))
    shutil.copyfile(src, dest)

def generate_psf(psf_output, resxy, resz, wavelength, NA, ni):
    """dw_bw command to generate PSF."""
    command = [
        "dw_bw",  # Make sure dw_bw is in your PATH or specify the full path
        "--resxy", str(resxy),  # Lateral pixel size (nm)
        "--resz", str(resz),    # Axial pixel size (nm)
        "--lambda", str(wavelength),  # Wavelength (nm)
        "--NA", str(NA),  # Numerical aperture
        "--ni", str(ni),  # Refractive index
        psf_output  # Output PSF file (e.g., PSF_dapi.tif)
    ]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        #print(f"PSF generated and saved as {psf_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating PSF: {e}")


def deconvolve_image(input_image, psf_image, output_image, iterations, tilesize=None):
    """DeconWolf command to deconvolve the image"""

    command = [
    "deconwolf",
    "--iter", str(iterations),
    input_image,
    psf_image,
    "--out", output_image
    ]

    if tilesize is not None:
        command += ['--tilesize', str(tilesize)]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        print(f"Deconvolution finished. Output saved to {output_image}")
        
    except subprocess.CalledProcessError as e:
        print(f"\033[91mError during deconvolution: {e}\033[0m")
    except FileNotFoundError as e:
        # e.filename is the missing executable or file
        print(f"\033[91mError: executable not found: {e.filename}\033[0m")
    except Exception as e:
        print(f"\033[91mUnexpected error: {e}\033[0m")

def file_exists_and_valid(path: Path, min_size: int = 1024) -> bool:
    """
    Check if a file exists and is larger than a minimum size (default 1 KB).
    This helps detect corrupted or empty files from failed previous runs.

    Parameters
    ----------
    path : Path
        Path to the file being checked.
    min_size : int, optional
        Minimum file size in bytes. Default is 1024 (1 KB).

    Returns
    -------
    bool
        True if the file exists and is valid, False otherwise.
    """
    return path.exists() and path.stat().st_size > min_size

def normalize_czi_array(arr, dims):
    """
    Normalize CZI numpy array into shape (M, Z, C, Y, X).
    Missing dimensions are inserted as singleton axes.

    Parameters
    ----------
    arr : np.ndarray
        Array from CziFile.asarray().
    dims : dict
        Dimension sizes from CziFile.dims.

    Returns
    -------
    np.ndarray
        Array reshaped to (M, Z, C, Y, X).
    """

    # Extract sizes (default to 1 if missing)
    s = dims.get("S", 1)   # scenes
    m = dims.get("M", 1)   # mosaic tiles
    z = dims.get("Z", 1)   # z-slices
    c = dims.get("C", 1)   # channels
    y = dims.get("Y")
    x = dims.get("X")

    # Collapse S and M into one "M"
    msize = s * m

    expected_size = msize * z * c * y * x
    if arr.size != expected_size:
        raise ValueError(
            f"Array size mismatch: got {arr.shape}, "
            f"expected total {expected_size} "
            f"from sizes M={msize}, Z={z}, C={c}, Y={y}, X={x}"
        )

    arr = arr.reshape((msize, z, c, y, x))
    return arr

def normalize_dims_shape(czi):
    """
    Normalize dims_shape from aicspylibczi.CziFile.get_dims_shape() to always be a dict {axis: size}.
    Flexible for different versions of aicspylibczi:
      - dict already → return as-is
      - list of dicts → unpack axis: (start, size) → keep only size
      - list of tuples → take first two items (axis, size)
    """
    dims_shape = czi.get_dims_shape()

    if isinstance(dims_shape, dict):
        return dims_shape

    if isinstance(dims_shape, list):
        out = {}
        for elem in dims_shape:
            if isinstance(elem, dict):
                # Example: {'X': (0, 2048), 'Y': (0, 2048), ...}
                for axis, rng in elem.items():
                    if isinstance(rng, tuple) and len(rng) == 2:
                        out[axis] = rng[1]  # take size only
                    else:
                        raise ValueError(f"Unexpected value for axis {axis}: {rng}")

            elif isinstance(elem, tuple) and len(elem) >= 2:
                # Example: ("X", 2048)
                axis, size = elem[0], elem[1]
                out[axis] = size

            else:
                raise ValueError(f"Unexpected element in dims_shape: {elem}")

        return out

    raise TypeError(f"Unexpected dims_shape type: {type(dims_shape)}")


def normalize_nd2_array(arr, sizes):
    """
    Normalize ND2 numpy array into shape (M, Z, C, Y, X).
    Missing dimensions are inserted as singleton axes (size=1).

    Parameters:
        arr (np.ndarray): array from nd2.imread() or f.to_dask().compute()
        sizes (dict): dimension sizes from ND2File.sizes

    Returns:
        np.ndarray: array reshaped to (M, Z, C, Y, X)
    """
    m = sizes.get("M", 1)
    z = sizes.get("Z", 1)
    c = sizes.get("C", 1)
    y = sizes.get("Y")
    x = sizes.get("X")

    # Make sure array has the right number of elements
    expected_size = m * z * c * y * x
    if arr.size != expected_size:
        raise ValueError(
            f"Array size mismatch: got {arr.shape}, expected total {expected_size} "
            f"from sizes M={m}, Z={z}, C={c}, Y={y}, X={x}"
        )

    # Reshape into consistent 5D layout
    arr = arr.reshape((m, z, c, y, x))
    return arr


def decide_and_write_tilescan(
    *,
    # inputs common to all modes
    x_raw, y_raw,                         # np.array of raw stage coords (PosX, PosY)
    image_dimensions,                     # (X, Y) in pixels
    pixel_to_um_manual=None,              # float|None: manual pixel size (µm/px)
    pixel_to_um_calc=None,                # float|None: metadata-derived pixel size (µm/px)
    unit_hint_raw="",                     # e.g. "m", "µm", "", "pixels"
    off_tol=0.35,                         # tolerance for accepting metadata unit (slightly looser)
    # writing params
    tiles_iter=None,                      # iterable of (FieldX, FieldY, PosX_raw, PosY_raw)
    app_name="LAS X",                     # Attachment@Application
    out_xml_path=None,                     # Path for output XML
    deconvolution_method=None,       
    deconvolution_iterations=None
):
    """
    Decide stage position units + pixel size (µm/px),
    auto-correct pixel_to_um if overlap large,
    and write LAS/CZI-style TileScanInfo XML with full provenance.

    Returns dict with:
      chosen_unit, to_um, rationale, dx, dy, ovx, ovy,
      pixel_to_um, pixel_to_um_source, tile_width_um, width_px, unit_hint_normalized
    """
    import numpy as _np
    import xml.etree.ElementTree as _ET

    # ---------------- Helpers ----------------
    def _normalize_unit(u: str) -> str:
        if not u: return "unknown"
        u = u.strip().lower().replace("µ", "u")
        if u in {"um", "u", "micron", "microns"} or "micromet" in u: return "microns"
        if u in {"px", "pixel", "pixels"}: return "pixels"
        if u in {"m", "meter", "metre", "meters", "metres"}: return "meters"
        if u in {"mm", "millimeter", "millimetre", "millimeters", "millimetres"}: return "millimeters"
        return "unknown"

    def _robust_step_1d(vals_um, tile_width_um=None):
        """
        Estimate tile step robustly:
        - If tile_width_um known: median of diffs in [0.5, 1.2]×width (fallback to top 30%)
        - Else: median of largest 30% of positive diffs
        """
        if vals_um is None or len(vals_um) < 2:
            return None
        u = _np.unique(_np.round(vals_um, 9))
        if u.size < 2:
            return None
        d = _np.diff(_np.sort(u))
        d = d[d > 0]
        if d.size == 0:
            return None
        if tile_width_um and tile_width_um > 0:
            lo, hi = 0.5 * tile_width_um, 1.2 * tile_width_um
            band = d[(d >= lo) & (d <= hi)]
            if band.size == 0:
                k = max(1, int(0.3 * d.size))
                band = _np.sort(d)[-k:]
            return float(_np.median(band))
        k = max(1, int(0.3 * d.size))
        big = _np.sort(d)[-k:]
        return float(_np.median(big))

    def _ov_pct(step_um, width_um):
        if step_um is None or not width_um:
            return None
        return (1 - step_um / width_um) * 100.0

    def _ov_qual(p):
        if p is None: return "n/a"
        if 5 <= p <= 15: return "typical"
        if p < 0: return "gap?"
        if p > 25: return "large"
        return "ok"

    def _fit_for_scale(scale_um_per_raw, tile_width_um):
        x_um = x_raw * scale_um_per_raw
        y_um = y_raw * scale_um_per_raw
        dx = _robust_step_1d(x_um, tile_width_um)
        dy = _robust_step_1d(y_um, tile_width_um)
        if tile_width_um:
            axis_scores = []
            if dx is not None: axis_scores.append(abs(dx - tile_width_um) / max(tile_width_um, 1e-9))
            if dy is not None: axis_scores.append(abs(dy - tile_width_um) / max(tile_width_um, 1e-9))
            score = min(axis_scores) if axis_scores else float("inf")
        else:
            # fallback order-of-magnitude guess if no width known
            max_abs_raw = float(_np.max(_np.abs(_np.concatenate([x_raw, y_raw])))) if x_raw.size else 0.0
            if _np.isclose(scale_um_per_raw, 1e6): score = 0.0 if max_abs_raw < 1e-3 else 1.0
            elif _np.isclose(scale_um_per_raw, 1.0): score = 0.0 if max_abs_raw >= 1e-3 else 1.0
            else: score = 0.5
        return dict(score=score, dx=dx, dy=dy,
                    ovx=_ov_pct(dx, tile_width_um),
                    ovy=_ov_pct(dy, tile_width_um))

    def _write_xml(*, to_um, chosen_unit, rationale, unit_hint_raw, unit_hint_norm,
                   pixel_to_um, pixel_to_um_source, width_px, tile_width_um, dx, dy,
                   tiles_iter, app_name, out_xml_path,
                   deconvolution_method=None,            
                   deconvolution_iterations=None):
        
        out = _ET.Element("Data")
        img = _ET.SubElement(out, "Image", TextDescription="")
        att = _ET.SubElement(img, "Attachment", Name="TileScanInfo",
                             Application=app_name, FlipX="0", FlipY="0", SwapXY="0")
        att.set("Unit", "micron")
        att.set("DeclaredUnitHint", unit_hint_raw or "unknown")
        att.set("DeclaredUnitNormalized", unit_hint_norm or "unknown")
        att.set("RawPositionUnitUsed", chosen_unit or "unknown")
        att.set("ScaleRawToMicron", f"{to_um:.12g}")
        att.set("DecisionNote", rationale or "")
    
        if deconvolution_method is None:
            att.set("DeconvolutionMethod", "None")
            att.set("DeconvolutionIterations", "0")
        else:
            att.set("DeconvolutionMethod", str(deconvolution_method))
            att.set("DeconvolutionIterations", str(deconvolution_iterations or 0))


        if pixel_to_um is not None:
            att.set("PixelSizeUm", f"{float(pixel_to_um):.10f}")
            if pixel_to_um_source:
                att.set("PixelSizeSource", pixel_to_um_source)
        if width_px is not None:
            att.set("TileWidthPx", str(int(width_px)))
        if tile_width_um is not None:
            att.set("TileWidthUm", f"{float(tile_width_um):.10f}")
        if dx is not None:
            att.set("MedianStepXUm", f"{float(dx):.10f}")
        if dy is not None:
            att.set("MedianStepYUm", f"{float(dy):.10f}")

        for t in tiles_iter or []:
            if isinstance(t, dict):
                fx, fy = int(t["FieldX"]), int(t["FieldY"])
                px_raw, py_raw = float(t["PosX"]), float(t["PosY"])
            else:
                fx, fy, px_raw, py_raw = int(t[0]), int(t[1]), float(t[2]), float(t[3])
            _ET.SubElement(att, "Tile",
                           FieldX=str(fx), FieldY=str(fy),
                           PosX=f"{px_raw * to_um:.10f}",
                           PosY=f"{py_raw * to_um:.10f}")

        _ET.ElementTree(out).write(out_xml_path, encoding="utf-8", xml_declaration=True)
        print(f"[INFO] Wrote TileScanInfo: {out_xml_path} (positions in µm)")

    # ---------------- Decision logic ----------------
    width_px = image_dimensions[0] if isinstance(image_dimensions, (tuple, list)) else None

    # pixel size selection
    pixel_to_um = None
    pixel_to_um_source = "unavailable"
    if pixel_to_um_manual is not None:
        pixel_to_um = float(pixel_to_um_manual)
        pixel_to_um_source = "manual argument"
    elif pixel_to_um_calc is not None:
        pixel_to_um = float(pixel_to_um_calc)
        pixel_to_um_source = "metadata-derived"

    tile_width_um = (width_px * pixel_to_um) if (width_px and pixel_to_um) else None
    unit_hint_norm = _normalize_unit(unit_hint_raw or "")

    # include mm hypothesis
    candidates = {
        "meters": 1e6,
        "millimeters": 1e3,
        "microns": 1.0,
        "pixels": pixel_to_um if pixel_to_um is not None else None
    }

    chosen_unit = None
    rationale = ""
    dx = dy = ovx = ovy = None
    to_um = None

    # Try metadata unit first (if provided and supported)
    if unit_hint_norm in candidates and candidates[unit_hint_norm] is not None:
        r = _fit_for_scale(candidates[unit_hint_norm], tile_width_um)
        if r["score"] <= off_tol:
            chosen_unit = unit_hint_norm
            to_um = candidates[chosen_unit]
            dx, dy, ovx, ovy = r["dx"], r["dy"], r["ovx"], r["ovy"]
            parts = []
            parts.append(f"ΔX≈{dx:.2f} µm (overlap≈{ovx:.1f}% {_ov_qual(ovx)})" if dx is not None else "ΔX=n/a")
            parts.append(f"ΔY≈{dy:.2f} µm (overlap≈{ovy:.1f}% {_ov_qual(ovy)})" if dy is not None else "ΔY=n/a")
            rationale = f"metadata unit '{chosen_unit}' confirmed: {', '.join(parts)} vs width≈{tile_width_um:.2f} µm" if tile_width_um else f"metadata unit '{chosen_unit}' confirmed"
        else:
            print(f"[WARN] Metadata unit '{unit_hint_norm}' inconsistent (overlap X≈{r['ovx']}, Y≈{r['ovy']}) — running hypothesis test.")

    # Hypothesis selection if not confirmed
    if chosen_unit is None:
        scores, details = {}, {}
        for name, scale in candidates.items():
            if scale is None:
                continue
            r = _fit_for_scale(scale, tile_width_um)
            scores[name] = r["score"]
            details[name] = r
        if not scores:
            chosen_unit, to_um, rationale = "microns", 1.0, "no evaluable hypotheses; defaulting to microns"
        else:
            chosen_unit = min(scores, key=scores.get)
            to_um = candidates[chosen_unit]
            dx, dy, ovx, ovy = (details[chosen_unit][k] for k in ("dx", "dy", "ovx", "ovy"))
            parts = []
            parts.append(f"ΔX≈{dx:.2f} µm (overlap≈{ovx:.1f}% {_ov_qual(ovx)})" if dx is not None else "ΔX=n/a")
            parts.append(f"ΔY≈{dy:.2f} µm (overlap≈{ovy:.1f}% {_ov_qual(ovy)})" if dy is not None else "ΔY=n/a")
            if tile_width_um is not None:
                rationale = f"{chosen_unit} chosen by hypothesis: {', '.join(parts)} vs width≈{tile_width_um:.2f} µm"
            else:
                rationale = f"{chosen_unit} chosen by hypothesis: {', '.join(parts)}"

    # Large-overlap auto-correction check (try the other pixel size source if available)
    def _is_large(v): return v is not None and v > 25.0
    if _is_large(ovx) or _is_large(ovy):
        if pixel_to_um_manual is not None and pixel_to_um_calc is not None:
            other_px = float(pixel_to_um_calc) if pixel_to_um_source == "manual argument" else float(pixel_to_um_manual)
            if not _np.isclose(other_px, float(pixel_to_um), rtol=0.05):
                new_tile_width_um = (width_px * other_px) if width_px else None
                r2 = _fit_for_scale(to_um, new_tile_width_um)
                ovx2, ovy2 = r2["ovx"], r2["ovy"]
                if all(v is not None and v <= 25.0 for v in (ovx2, ovy2)):
                    print(
                        f"[WARN] Large overlap ({ovx:.1f}%, {ovy:.1f}%) with {pixel_to_um:.6f} µm/px — "
                        f"switching to {other_px:.6f} µm/px improves overlap "
                        f"({ovx2:.1f}%, {ovy2:.1f}%).\n\033[1m⚠️ Please verify your manual 'pixel_to_um' value.\033[0m"
                    )
                    pixel_to_um = other_px
                    pixel_to_um_source = "metadata-derived" if pixel_to_um_source == "manual argument" else "manual argument"
                    tile_width_um = new_tile_width_um
                    # keep outputs consistent:
                    dx, dy = r2["dx"], r2["dy"]
                    ovx, ovy = ovx2, ovy2
                    rationale = f"{chosen_unit} confirmed; pixel size auto-corrected"

    # Safe print (avoid formatting None)
    px_str = f"{pixel_to_um:.6f}" if pixel_to_um is not None else "NA"
    tw_str = f"{tile_width_um:.2f}" if tile_width_um is not None else "NA"
    print(
        f"[INFO] Position unit decision: {rationale} "
        f"[unit used='{chosen_unit}'; width_px={width_px}; pixel_to_um={px_str} µm/px; tile_width_um={tw_str} µm]"
    )

    if out_xml_path and tiles_iter:
        _write_xml(
            to_um=to_um, chosen_unit=chosen_unit, rationale=rationale,
            unit_hint_raw=unit_hint_raw, unit_hint_norm=unit_hint_norm,
            pixel_to_um=pixel_to_um, pixel_to_um_source=pixel_to_um_source,
            width_px=width_px, tile_width_um=tile_width_um, dx=dx, dy=dy,
            tiles_iter=tiles_iter, app_name=app_name, out_xml_path=out_xml_path,
            deconvolution_method=deconvolution_method,
            deconvolution_iterations=deconvolution_iterations
        )


    return dict(
        chosen_unit=chosen_unit, to_um=to_um, rationale=rationale,
        dx=dx, dy=dy, ovx=ovx, ovy=ovy,
        pixel_to_um=pixel_to_um, pixel_to_um_source=pixel_to_um_source,
        tile_width_um=tile_width_um, width_px=width_px,
        unit_hint_normalized=unit_hint_norm
    )


# -------------------------------------------------------------------------------------
# MAIN FUNCTION FOR PREPROCESSING
# -------------------------------------------------------------------------------------
def preprocessing_main(input_dirs,
                            cycles,
                            output_dir_prefix,
                            mode,
                            n_total_cycles,
                            regions_to_process=None,
                            deconvolution_method=None,
                            num_iterations = 50,
                            PSF_metadata=None, 
                            align_channel=4, 
                            mip=True,
                            tile_dimension=6000, 
                            pixel_to_um = None,
                            chunk_size=None):
    
    """
    Main preprocessing pipeline for microscopy image data.

    This function processes microscopy images stored in various formats/modes 
    (autosaved TIFF, exported TIFF, LIF, CZI and Nd2 files) and performs operations such as 
    region detection, deconvolution, and creation of OME-TIFF files. It organizes 
    outputs into directories, manages PSF generation, and optionally applies 
    Maximum Intensity Projection (MIP).

    Parameters
    ----------
    input_dir : str or Path
        Path to the input directory containing raw microscopy image files.

    output_dir_prefix : str or Path
        Base path prefix where output directories and processed files will be saved.

    cycle : int or str
        Identifier for the current imaging cycle being processed (e.g., cycle number).

    mode : str
        Input data format/mode. Supported values:
        - 'tif_autosaved': TIFF files saved automatically by Leica software.
        - 'tif_exported': TIFF files exported manually.
        - 'lif': Leica Image File (LIF) format.

    deconvolution_method : str or None, optional
        Deconvolution algorithm to use. Supported values:
        - 'deconwolf'
        - 'redlionfish'
        - None (skip deconvolution)

    PSF_metadata : dict or None
        Metadata required to generate the Point Spread Function (PSF) for deconvolution.
        Required if deconvolution is to be performed.

    align_channel : int, optional
        Channel index used for image alignment. Default is 4.

    mip : bool, optional
        Whether to apply Maximum Intensity Projection (MIP) to image stacks. Default is True.

    tile_dimension : int, optional
        Dimension (in pixels) of image tiles for processing. Default is 6000.

    chunk_size : int or None, optional
        Size of chunks for processing large images in segments. Default is None (process whole image).

    Raises
    ------
    ValueError
        If `mode` or `deconvolution_method` is invalid, or required parameters are missing.

    """
    
    script_start_time = time.time()

    valid_modes = {'tif_autosaved', 'tif_exported', 'lif', 'nd2', 'czi'}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}. Choose from {valid_modes}.")

    valid_methods = {'deconwolf', 'redlionfish', None}
    if deconvolution_method not in valid_methods:
        raise ValueError(f"Unsupported deconvolution method: {deconvolution_method}. Choose from {valid_methods - {None}} or None.")

    if deconvolution_method is not None and PSF_metadata is None:
        raise ValueError("PSF_metadata is required to generate PSF when deconvolution method is specified.")

    # DECONVOLUTION
    region_directories = deconvolve_and_mip(
                            input_dirs=input_dirs,
                            cycles=cycles,
                            output_dir_prefix=output_dir_prefix, 
                            mode=mode,
                            regions_to_process=regions_to_process,
                            deconvolution_method=deconvolution_method,
                            num_iterations=num_iterations,
                            PSF_metadata=PSF_metadata, 
                            mip=mip,
                            pixel_to_um=pixel_to_um,
                            chunk_size=chunk_size
                            )

    # OME TIFFS
    mipped_to_OME_tiffs(
        region_directories=region_directories, 
        cycles=cycles,
        pixel_to_um=pixel_to_um)

    # Align and stitch images
    align_and_stitch(region_directories=region_directories, 
                   cycles=cycles,
                   n_total_cycles=n_total_cycles,
                   align_channel=align_channel)

    # retile stitched images
    retile_stitched_images(region_directories=region_directories, 
                    cycles=cycles, 
                    tile_dimension=tile_dimension) 


    # ----- Step 10: Final reporting -----
    script_end_time = time.time()
    print(f"\033[96m[Total Runtime] Full preprocessing pipeline took {(script_end_time - script_start_time)/60:.2f} minutes\033[0m")
    # ----    
    
    return

   

def deconvolve_and_mip(
    input_dirs, 
    cycles ,
    output_dir_prefix: Path,
    mode: str,
    regions_to_process: Optional[List[int]] = None,
    deconvolution_method: Optional[str] = None,
    num_iterations = 50,
    PSF_metadata: Optional[dict] = None, 
    mip: bool = True,
    pixel_to_um = None,
    chunk_size: Optional[int] = None
) -> list:
    """
    Deconvolve Leica microscopy data for a given cycle.
    
    Parameters:
        input_dir (Path): Directory containing input image files.
        output_dir_prefix (Path): output directory.
        mode (str): One of 'tif_autosaved', 'tif_exported', or 'lif'.
        deconvolution_method (str | None): 'redlionfish', 'deconwolf', or None.
        PSF_metadata (dict): Metadata needed to generate PSFs.
        mip (bool): Whether to save maximum intensity projections (MIP).
        chunk_size (int | None): Tile size for Deconwolf processing.
        
    Returns:
        List of directories for each region. Saves output images and metadata files to disk.
    """ 
    print(f"\033[1;96mDeconvolution and mipping\033[0m")
    
    valid_modes = {'tif_autosaved', 'tif_exported', 'lif', 'nd2', 'czi'}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}. Choose from {valid_modes}.")

    valid_methods = {'deconwolf', 'redlionfish', None}
    if deconvolution_method not in valid_methods:
        raise ValueError(f"Unsupported deconvolution method: {deconvolution_method}. Choose from {valid_methods - {None}} or None.")

    for cycle, input_dir in zip(cycles, input_dirs):

        width = 80
      
        print("=" * width + "\033[0m")
        print(f"\033[1;90mProcessing Cycle {cycle}\033[0m")
        print('Processing directory: ', input_dir)  
        print(f"Mode: {mode}".ljust(width))
        print(f"Deconvolution method: {deconvolution_method}, iterations: {num_iterations}".ljust(width))
    
        if deconvolution_method is not None and PSF_metadata is None:
            raise ValueError("PSF_metadata is required to generate PSF when deconvolution method is specified.")
        
        input_dir = Path(input_dir)
        output_dir_prefix = Path(output_dir_prefix)  
    
        # STEP 1: Detect regions to process
        
        # --- Processing Leica .tif files ---
        if mode == 'tif_exported':
            tif_files = [
                f.name
                for f in input_dir.iterdir()
                if f.suffix == '.tif' and 'dw' not in f.name and '.txt' not in f.name
            ]
            # Use underscore split
            region_names = set()
            for f in tif_files:
                base = f.rsplit('.', 1)[0]
                chunks = base.split('_')
                region_name = chunks[0]
                region_names.add(region_name)
            regions = sorted(region_names)
            num_regions = len(regions)
        
        elif mode == 'tif_autosaved':
            tif_files = [
                f.name
                for f in input_dir.iterdir()
                if f.suffix == '.tif' and 'dw' not in f.name and '.txt' not in f.name
            ]
            # Use double-dash split
            region_names = set()
            for f in tif_files:
                base = f.rsplit('.', 1)[0]
                chunks = base.split('--')
                region_name = chunks[0]
                region_names.add(region_name)
            regions = sorted(region_names)
            num_regions = len(regions)
        
        # --- Processing Leica .lif files ---
        elif mode == 'lif':
            lif_files = [f for f in input_dir.iterdir() if f.suffix == '.lif']
            num_files = len(lif_files)
    
            image_names = []   # To store names of images inside .lif files
            
            if num_files > 1:
                # Case: one file per region
                num_regions = num_files                           # one file for each region
                for file in lif_files:
                    lif_file = LifFile(file)
                    image_dict = lif_file.image_list[0]           # Each .lif has one image per region
                    image_names.append(image_dict['name'])    
            
            elif num_files == 1:
                # Case: one .lif file containing multiple regions
                lif_file = LifFile(lif_files[0])
                num_regions = len(lif_file.image_list)            # number of images = number of regions
                for image_dict in lif_file.image_list:
                    image_names.append(image_dict['name'])
    
            # Use unique image names directly as region names 
            regions = sorted(set(image_names))

        # --- Processing Zeiss .czi files ---
        elif mode == 'czi':
            czi_files = [f for f in input_dir.iterdir() if f.suffix == '.czi']
            if not czi_files:
                raise ValueError("No CZI files found in input_dir")
        
            file = czi_files[0] if len(czi_files) == 1 else czi_files[region_index]
            print(f"Using CZI file: {file.name}")
        
            czi = CziFile(str(file))
            dims = normalize_dims_shape(czi)
        
            print("CZI dims:", dims)
        
            # --- Regions (Scenes = S dimension) ---
            num_regions = dims.get("S", 1)
            regions = [f"Region_{i+1}" for i in range(num_regions)]        
    

        # --- Processing Nikon .nd2 files ---
        elif mode == 'nd2':
            nd2_files = [f for f in input_dir.iterdir() if f.suffix == '.nd2']
            num_files = len(nd2_files)
        
            image_names = []
            if num_files > 1:
                # One ND2 per region
                num_regions = num_files
                for file in nd2_files:
                    ndfile = nd2.ND2File(file)
                    image_names.append(file.stem)
                    ndfile.close()
            elif num_files == 1:
                # One ND2 with multiple regions
                ndfile = nd2.ND2File(nd2_files[0])
                num_regions = ndfile.sizes.get("M", 1)  # number of mosaic positions
                image_names = [f"Region_{i+1}" for i in range(num_regions)]
                ndfile.close()
            else:
                raise ValueError("No ND2 files found in input_dir")
        
            regions = sorted(set(image_names))

        # rename regions    
        region_numbers = list(range(1, num_regions + 1))  # [1, 2, ..., num_regions]

        # select a subset of regions to be processed
        if regions_to_process is not None:
            # Convert region_numbers (1-based) into indexes (0-based)
            selected_indices = [i - 1 for i in regions_to_process 
                                if 1 <= i <= len(regions)]
        
            # Reduce regions and region_numbers
            regions = [regions[i] for i in selected_indices]
            region_numbers = [region_numbers[i] for i in selected_indices]
        
            #print(f"User-selected regions_to_process = {regions_to_process}")
            #print(f"Reduced regions = {regions}")
            #print(f"Reduced region_numbers = {region_numbers}")
                
    
        print("Regions to be processed:", regions) 
        print("=" * width + "\033[0m")
    
        region_directories = []  # To collect all processed region directories
    
        # Process each region
        for region_index, region in enumerate(regions):
            print(f"\033[1;90mProcessing R{region_numbers[region_index]}\033[0m")
    
            # Define output directory for this region, always append "R{region_number}" to distinguish them
            region_directory = output_dir_prefix / f"R{region_numbers[region_index]}"
    
            region_directories.append(str(region_directory))
            # Create region directory (with parent folders, if needed)
            region_directory.mkdir(parents=True, exist_ok=True)
    
            # Create cycle directory inside region directory: "preprocessing/Cycle{cycle}"
            cycle_directory = region_directory / 'preprocessing' / f'Cycle{cycle}'
            cycle_directory.mkdir(parents=True, exist_ok=True)  
        
            # Create directory to store MIP (Maximum Intensity Projection) images
            mipped_directory = cycle_directory / '1_mipped'
            mipped_directory.mkdir(exist_ok=True)
    
            # Prepare directory to store stacked images
            stacked_directory = cycle_directory / '1_stacked'
        
            # Create directory to store metadata files
            metadata_directory = cycle_directory / 'MetaData'
            metadata_directory.mkdir(exist_ok=True)
        
            # ----- STEP 1: PREPARE FILE LISTS BASED ON MODE -----
            # --- tif file preparations ---
            if mode in ('tif_autosaved', 'tif_exported'):
                # List all .tif files in input_dir (skip ".txt" and "dw" files)
                tif_files = [
                    f for f in input_dir.iterdir() 
                    if f.suffix == '.tif' and 'dw' not in f.name and not f.name.endswith('.txt')
                ]
                # Filter only files for the current region
                filtered_tifs = [f for f in tif_files if region in f.name]
    
                # --- Find all channels from filenames ---
                channel_set = set()
                if mode == 'tif_autosaved':
                    channel_pattern = re.compile(r'--C(\d{2})')    # channels in format "--C01", "--C02", ...
                elif mode == 'tif_exported':
                    # Match "_ch00", "_Ch00", "_CH00", "_ch0", etc. (case-insensitive)
                    channel_pattern = re.compile(r'_c[hH](\d+)', re.IGNORECASE)

            
                # Populate channel set by scanning filenames
                for f in filtered_tifs:
                    if (m := channel_pattern.search(f.name)):
                        channel_set.add(int(m.group(1)))           # store channels as integers
            
                channels = sorted(channel_set)
                if not channels:
                    raise RuntimeError(f"No channels detected in files for region {region}")

                # --- Detect tiles and find sample tile(s) ---
                if mode == 'tif_autosaved':
                    tile_pattern = re.compile(r'--Stage(\d+)--')      # tile number in "--StageXX--"
                    sample_indicator = re.compile(r'--Stage0+--')     # matches "--Stage0--", "--Stage00--", etc.
                elif mode == 'tif_exported':
                    tile_pattern = re.compile(r'_s(\d+)_')            # capture tile number in "_s###_"
                    sample_indicator = re.compile(r'_s0+_')           # matches "_s0_", "_s00_", "_s000_", etc.
                
                # Extract tile numbers and collect sample tile files
                tiles = set()        # unique tile numbers
                sample_tiles = []    # files belonging to tile 0 (any form of 0-padded index)
                
                for f in filtered_tifs:
                    if tile_pattern.search(f.name):
                        tiles.add(tile_pattern.search(f.name).group(1))   # collect tile numbers
                    if sample_indicator.search(f.name):                   # regex match for "tile 0"
                        sample_tiles.append(f)

                # Safe fallback: if no tile 0 exists, pick the lowest available tile
                if not sample_tiles and tiles:
                    lowest_tile = min(int(t) for t in tiles)
                    # Build regex dynamically depending on mode
                    if mode == 'tif_exported':
                        fallback_pattern = re.compile(rf'_s0*{lowest_tile}_')
                    else:  # tif_autosaved
                        fallback_pattern = re.compile(rf'--Stage0*{lowest_tile}--')
                    sample_tiles = [f for f in filtered_tifs if fallback_pattern.search(f.name)]
        
                # Sort tile list and compute total number of tiles
                tiles = sorted(tiles, key=int)
                n_tiles = len(tiles)
                # Infer Z-size from number of sample_tile files divided by number of channels
                size_z = int(len(sample_tiles) / len(channels))
                # Infer image dimensions (X, Y) from the first sample tile
                sample_tile = tifffile.imread(sample_tiles[0])
                image_dimensions = sample_tile.shape[::-1]  # (width, height)

                print(f"Tiles: {n_tiles}, Z-slices: {size_z}, Channels: {len(channels)}")
                print(f"Image dimensions: {image_dimensions[0]} × {image_dimensions[1]} (X × Y)")
        
                # --- Pre-index files by tile and channel to speed up lookups ---
                tile_to_files = {}
                for tile in tiles:
                    if mode == 'tif_autosaved':
                        tile_files = [f for f in filtered_tifs if f"--Stage{tile}--" in str(f)]
                    else:
                        tile_files = [f for f in filtered_tifs if f"_s{tile}_" in str(f)]
                    tile_to_files[tile] = tile_files
        
                tile_channel_files = {}
                for tile, files_in_tile in tile_to_files.items():
                    for channel in channels:
                
                        if mode == 'tif_autosaved':
                            # Example filenames:  ...--C01--..., ...--C10--...
                            pattern = re.compile(rf"--C{str(channel).zfill(2)}", re.IGNORECASE)
                
                        else:
                            # tif_exported: supports 1–N digits, any zero padding, any case
                            # Matches: _ch0, _ch00, _ch000, _Ch02, _CH2, etc.
                            pattern = re.compile(rf"_ch0*{channel}\b", re.IGNORECASE)
                
                        channel_files = [f for f in files_in_tile if pattern.search(f.name)]
                        tile_channel_files[(tile, channel)] = channel_files

                        
            # --- lif file preparations ---
            elif mode == 'lif':
                # List all .lif files in input_dir
                lif_files = [f for f in input_dir.iterdir() if f.suffix == '.lif']
                num_files = len(lif_files)
                
                if num_files > 1:
                    # Case: multiple .lif files → one file per region
                    filepath = lif_files[region_index]
                    file = LifFile(filepath)
                    image_dict = file.image_list[0]  # always take first image from multi-file set
                    image_name = image_dict['name']
                    image = file.get_image(0)
                elif num_files == 1:
                    # Case: single .lif file → contains multiple regions
                    filepath = lif_files[0]
                    file = LifFile(filepath)
                    image_dict = file.image_list[region_index]  # select image by region_index if single file
                    image_name = image_dict['name']
                    image = file.get_image(region_index)
            
                print(f"Image name: {image_name}")
                # Replace "/" with "_" in image name (prevent file system issues)
                image_name = image_name.replace('/', '_')
        
                dims = image_dict['dims']                        # Extract dimensions
                image_dimensions = (dims.x, dims.y)  # (width, height)
                size_z = dims.z                                  # number of Z slices
                n_tiles = dims.m                                 # number of mosaic tiles (if any)
                tiles = list(range(n_tiles))                     # tile indices 0..n_tiles-1
                tiles = sorted(tiles, key=int)
                mosaic = image_dict.get('mosaic_position', None) # Get mosaic positions
                num_channels = image_dict['channels']
                channels = list(range(num_channels))  # [0, 1, 2, 3, 4, 5]

            # --- czi file preparations ---
            elif mode == 'czi':
               # --- Gather basic CZI info ---
                size_z = dims.get("Z", 1)
                num_channels = dims.get("C", 1)
                n_tiles = dims.get("M", 1)
                image_dimensions = (dims.get("X"), dims.get("Y"))
                channels = list(range(num_channels))
            
                tiles = []
                for m in range(n_tiles):
                    try:
                        # Try reading the first Z and C without specifying S or B
                        img, shp = czi.read_image(M=m, C=0, Z=0)
                        if img is not None:
                            tiles.append(m)
                    except Exception as e:
                        print(f"[INFO] Skipping tile {m} ({e.__class__.__name__})")
            
                print(
                    f"{len(tiles)} valid tiles (out of {n_tiles}), "
                    f"{size_z} Z-slices, {num_channels} channels, "
                    f"image size {image_dimensions[0]} × {image_dimensions[1]}"
                )


            # --- nd2 file preparations ---
            elif mode == 'nd2':
                print(f"\033[1;93m[ND2 MODE] Initializing Nikon ND2 processing for region {region}\033[0m")
            
                # Collect all .nd2 files in the input directory
                nd2_files = [f for f in input_dir.iterdir() if f.suffix == '.nd2']
                print(f"Found {len(nd2_files)} ND2 file(s) in input directory")
            
                # Pick the correct file depending on acquisition setup
                filepath = nd2_files[0] if len(nd2_files) == 1 else nd2_files[region_index]
                print(f"Using ND2 file: {filepath.name}")
            
                # --- Load ND2 and normalize to (M, Z, C, Y, X) ---
                with nd2.ND2File(filepath) as f:
                    sizes = f.sizes
                    print("ND2 sizes:", sizes)   # e.g. {'X': 3789, 'Y': 3789, 'M': 5}
            
                    # Load full array
                    arr = f.to_dask().compute()
                    arr = normalize_nd2_array(arr, sizes)  # -> (M, Z, C, Y, X)
            
                    # Try extracting stage coordinates
                    coords = []
                    exp = f.experiment
                    if hasattr(exp, "points") and exp.points:
                        for p in exp.points:
                            coords.append((p.x, p.y))
                        print(f"Extracted {len(coords)} stage coordinate(s) from experiment.points")
                    else:
                        exp_str = str(exp)
                        for match in re.finditer(r"x=([-+]?\d*\.?\d+), y=([-+]?\d*\.?\d+)", exp_str):
                            coords.append((float(match.group(1)), float(match.group(2))))
                        if coords:
                            print(f"Extracted {len(coords)} stage coordinate(s) from regex parsing")
                        else:
                            print("\033[91m[WARN] No stage coordinates found in ND2 metadata\033[0m")
                            print("Experiment object (repr):", repr(exp))
                            print("Experiment object (str):", exp_str[:500], "..." if len(exp_str) > 500 else "")
            
                            # Debug raw metadata for deeper inspection
                            try:
                                meta = f.metadata
                                print("Top-level ND2 metadata keys:", list(meta.keys()))
                            except Exception as e:
                                print(f"[DEBUG] Could not access f.metadata: {e}")
            
                # --- Assign variables for downstream code ---
                msize = arr.shape[0]                       # number of tiles (M)
                size_z = arr.shape[1]                      # z-slices
                channels = list(range(arr.shape[2]))       # channel indices
                image_dimensions = (arr.shape[4], arr.shape[3])  # (X, Y)
                tiles = list(range(msize))
                n_tiles = msize
            
                print(f"Normalized ND2 array shape: {arr.shape} (M, Z, C, Y, X)")
                print(f"Tiles: {n_tiles}, Z-slices: {size_z}, Channels: {len(channels)}")
                print(f"Image dimensions: {image_dimensions[0]} × {image_dimensions[1]} (X × Y)")


                    
            # ----- STEP 2: COPY METADATA IF AVAILABLE -----
            print("\033[96mExtracting metadata\033[0m")
                        
            # --- tif metadata (Leica → LAS/CZI-style; positions written in µm with explicit provenance) ---
            if mode in ('tif_autosaved', 'tif_exported'):
                # --- Read Leica XML from INPUT/Metadata ---
                # --- Locate Leica metadata folder (case-insensitive) ---
                input_metadata_dir = next(
                    (p for p in input_dir.iterdir()
                     if p.is_dir() and p.name.lower() == 'metadata'),
                    None
                )
                
                if input_metadata_dir is None:
                    print(f"[ERROR] No Leica metadata folder found in {input_dir} (case-insensitive search). Skipping region.")
                    continue

                else:
                    # Gather all plausible XML/XLF files (ignore property dumps)
                    md_files = [
                        f for f in input_metadata_dir.iterdir()
                        if f.suffix.lower() in ('.xml', '.xlif') and 'properties' not in f.name.lower()
                    ]
                
                    if not md_files:
                        print(f"[ERROR] No Leica XML/XLF files found in {input_metadata_dir}. Skipping region.")
                        continue

                    else:
                        # Heuristic: prefer files whose stem contains a region token; else newest file
                        region_token = (str(region) or "").strip()
                        if not region_token and filtered_tifs:
                            region_token = Path(filtered_tifs[0]).stem.split('_')[0]
             
                        # Case-insensitive preference for files whose stem contains the token
                        prio = [f for f in md_files if region_token and region_token.lower() in f.stem.lower()]
                        md_file = prio[0] if prio else max(md_files, key=lambda p: p.stat().st_mtime)
                        
                        print(f"[META] Using Leica XML: {md_file.name} ({md_file})")
                        
                        # Parse robustly; clear error message on failure; also guard empty root
                        try:
                            tree = ET.parse(md_file)
                            root = tree.getroot()
                            if root is None:
                                print(f"[ERROR] Parsed empty/None XML root from {md_file}. Skipping region.")
                                continue
                        except Exception as e:
                            print(f"[ERROR] Failed to parse Leica XML '{md_file}': {e}. Skipping region.")
                            continue
                       

                # ---------- Helpers ----------
                def _f(x):
                    try:
                        return float(x)
                    except Exception:
                        return None
            
                def _px_um_from_dim(dn, axis):
                    """Return pixel size in µm/px from Leica Dimensions, plus a short source tag."""
                    if dn is None:
                        return None, f"{axis}:missing"
                    N = _f(dn.attrib.get("NumberOfElements") or dn.attrib.get("Elements"))
                    L = _f(dn.attrib.get("Length"))
                    if not (N and L):
                        return None, f"{axis}:no_length_or_count"
                    raw = L / N  # length per pixel in declared units
                    # Magnitude-based normalization to µm (independent of Unit label reliability)
                    if 1e-9 <= raw <= 1e-4:      # meters/px (1 nm .. 100 µm)
                        return raw * 1e6, f"{axis}:Length/N (meters→µm ×1e6)"
                    else:                         # treat as µm/px
                        return raw, f"{axis}:Length/N (assumed µm)"
            
                # ---------- 1) Pixel size (µm/px): manual + metadata-derived "calc" ----------
                pixel_to_um_manual = float(pixel_to_um) if pixel_to_um is not None else None
            
                dim_x = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='1']")
                dim_y = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='2']")
                px_um_x, src_x = _px_um_from_dim(dim_x, "X")
                px_um_y, src_y = _px_um_from_dim(dim_y, "Y")
            
                pixel_to_um_calc = None
                if px_um_x and px_um_y:
                    rel = abs(px_um_x - px_um_y) / max(px_um_x, px_um_y)
                    pixel_to_um_calc = (px_um_x + px_um_y) / 2.0
                    print(f"[META] Pixel size from metadata: {pixel_to_um_calc:.6f} µm/px "
                          f"(X={px_um_x:.6f} [{src_x}], Y={px_um_y:.6f} [{src_y}])")
                    if rel > 0.02:
                        print(f"[WARN] X vs Y pixel sizes differ by {rel*100:.2f}% "
                              f"(X={px_um_x:.6f}, Y={px_um_y:.6f}). Using average {pixel_to_um_calc:.6f}.")
                else:
                    pixel_to_um_calc = px_um_x or px_um_y
                    if pixel_to_um_calc is not None:
                        print(f"[META] Pixel size from single axis: {pixel_to_um_calc:.6f} µm/px "
                              f"({src_x if px_um_x is not None else src_y})")
            
                if pixel_to_um_manual is not None:
                    print(f"[META] Manual pixel_to_um: {pixel_to_um_manual:.6f} µm/px")
            
                # ---------- 2) Choose pixel size (prefer manual, warn if mismatch) ----------
                effective_pixel_to_um = None
                if (pixel_to_um_calc is not None) and (pixel_to_um_manual is not None):
                    if not np.isclose(pixel_to_um_calc, pixel_to_um_manual, rtol=0.02):
                        print(f"[WARN] Manual pixel size ({pixel_to_um_manual:.6f} µm/px) differs "
                              f"from metadata value ({pixel_to_um_calc:.6f} µm/px).")
                    effective_pixel_to_um = pixel_to_um_manual
                    print(f"[META] Using manual pixel size: {effective_pixel_to_um:.6f} µm/px")
                elif pixel_to_um_manual is not None:
                    effective_pixel_to_um = pixel_to_um_manual
                    print(f"[META] Using manual pixel size: {effective_pixel_to_um:.6f} µm/px")
                elif pixel_to_um_calc is not None:
                    effective_pixel_to_um = pixel_to_um_calc
                    print(f"[META] No manual pixel size provided — using metadata pixel size: {effective_pixel_to_um:.6f} µm/px")
                else:
                    print(f"[ERROR] No pixel size information available — please provide 'pixel_to_um' manually. Skipping region.")
                    continue

            
                # ---------- 3) Objective magnification info (best-effort) ----------
                mag = None
                for xp in (".//Instrument//Objective",
                           ".//Attachment[@Name='HardwareSetting']//ATLCameraSettingDefinition"):
                    n = root.find(xp)
                    if n is not None:
                        mag = _f(n.attrib.get("Magnification")
                                 or n.attrib.get("NominalMagnification")
                                 or n.attrib.get("TotalVideoMag"))
                        if mag:
                            break
                if mag:
                    print(f"[META] Objective magnification: {mag:g}x")
            
                # ---------- 4) Collect raw stage positions + unit hint ----------
                tile_nodes = root.findall(".//Attachment[@Name='TileScanInfo']//Tile")
                if not tile_nodes:
                    print("[WARN] No <Tile> nodes found under TileScanInfo — nothing to write.")
                    tiles_iter = []
                    x_raw = np.empty((0,), dtype=float)
                    y_raw = np.empty((0,), dtype=float)
                else:
                    x_raw = np.array([float(n.attrib["PosX"]) for n in tile_nodes], dtype=float)
                    y_raw = np.array([float(n.attrib["PosY"]) for n in tile_nodes], dtype=float)
            
                    tiles_iter = [
                        (int(n.attrib["FieldX"]), int(n.attrib["FieldY"]),
                         float(n.attrib["PosX"]), float(n.attrib["PosY"]))
                        for n in sorted(tile_nodes, key=lambda e: (int(e.attrib["FieldY"]), int(e.attrib["FieldX"])))
                    ]
            
                # Leica's declared unit hint (from Dimensions X)
                unit_hint_raw = (dim_x.attrib.get("Unit", "") if dim_x is not None else "").strip().lower()
            
            
                # ---------- 5) Write tilescan XML via your helper ----------
                _ = decide_and_write_tilescan(
                    x_raw=x_raw,
                    y_raw=y_raw,
                    image_dimensions=image_dimensions,           # (X, Y) in pixels
                    pixel_to_um_manual=pixel_to_um_manual,       # keep both for traceability
                    pixel_to_um_calc=pixel_to_um_calc,           # computed above
                    unit_hint_raw=unit_hint_raw,                 # e.g. 'm', 'µm', etc.
                    off_tol=0.25,                                # your overlap tolerance
                    tiles_iter=tiles_iter,
                    app_name="LAS X",
                    out_xml_path=metadata_directory / f"{region}.xml",
                    deconvolution_method=deconvolution_method,       
                    deconvolution_iterations=num_iterations
                    )
            

            # --- lif metadata (Leica LIF → LAS-style XML; include pixel size, magnification, unit conversion) ---
            elif mode == 'lif':
                try:
                    if mosaic is None:
                        print(f"[ERROR] No mosaic information found for LIF region '{image_name}' — skipping XML metadata.")
                        continue
            
                    print("[INFO] Generating LIF TileScanInfo XML metadata")
            
                    # --- 1) Extract pixel size (voxel size) and magnification from LIF metadata ---
                    pixel_to_um_manual = float(pixel_to_um) if pixel_to_um is not None else None
                    pixel_to_um_calc = None
                    mag = None
                        
                    # Load the LIF metadata
                    xml_text = file.xml_header.decode("utf-8", errors="replace") if isinstance(file.xml_header, (bytes, bytearray)) else str(file.xml_header or "")
                    root = getattr(file, "xml_root", None)
            
                    # --- (a) Objective magnification ---
                    m = re.search(r'Magnification\s*=\s*["\']([\d.]+)["\']', xml_text, re.IGNORECASE)
                    if m:
                        try:
                            mag = float(m.group(1))
                            print(f"[META] Objective magnification: {mag:g}x")
                        except Exception:
                            pass
                    else:
                        print("[META] No magnification info found in xml_header.")
            
                    # --- (b) Pixel size from VoxelSizeX/Y/Z (meters → µm) ---
                    voxels = re.findall(r'VoxelSize[XYZ]\s*=\s*["\']([\deE.+-]+)["\']', xml_text, re.IGNORECASE)
                    if voxels:
                        try:
                            vals_um = [float(v) * 1e6 for v in voxels[:2] if v]
                            if vals_um:
                                pixel_to_um_calc = sum(vals_um) / len(vals_um)
                                print(f"[META] Pixel size from VoxelSize*: {pixel_to_um_calc:.6f} µm/px")
                        except Exception:
                            print("[WARN] Could not parse VoxelSize values from LIF xml_header.")
                    
            
                    # --- (c) Fallback: Dimensions (Length/Elements) if no VoxelSize* ---
                    if pixel_to_um_calc is None and root is not None:
                        dim_x = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='1']")
                        dim_y = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='2']")
            
                        def _safe_float(x):
                            try:
                                return float(x)
                            except Exception:
                                return None
            
                        def _dim_to_um(el):
                            if el is None:
                                return None
                            N = _safe_float(el.attrib.get("NumberOfElements") or el.attrib.get("Elements"))
                            L = _safe_float(el.attrib.get("Length"))
                            if not N or not L or N <= 0:
                                return None
                            raw = L / N
                            return raw * 1e6 if 1e-9 <= raw <= 1e-3 else raw  # meters → µm
            
                        vals = [v for v in (_dim_to_um(dim_x), _dim_to_um(dim_y)) if v is not None]
                        if vals:
                            pixel_to_um_calc = sum(vals) / len(vals)

                    if pixel_to_um_calc is not None:
                        print(f"[META] Pixel size from metadata: {pixel_to_um_calc:.6f} µm/px")
            
                    # --- 2) Choose pixel size (prefer manual, warn if mismatch) ---
                    effective_pixel_to_um = None
                    if (pixel_to_um_calc is not None) and (pixel_to_um_manual is not None):
                        if not np.isclose(pixel_to_um_calc, pixel_to_um_manual, rtol=0.02):
                            print(f"[WARN] Manual pixel size ({pixel_to_um_manual:.6f}) differs "
                                  f"from metadata value ({pixel_to_um_calc:.6f}).")
                        effective_pixel_to_um = pixel_to_um_manual
                        print(f"[META] Using manual pixel size: {effective_pixel_to_um:.6f} µm/px")
                    elif pixel_to_um_manual is not None:
                        effective_pixel_to_um = pixel_to_um_manual
                        print(f"[META] Using manual pixel size: {effective_pixel_to_um:.6f} µm/px")
                    elif pixel_to_um_calc is not None:
                        effective_pixel_to_um = pixel_to_um_calc
                        print(f"[META] No manual pixel size provided — using metadata pixel size: {effective_pixel_to_um:.6f} µm/px")
                    else:
                        print("[ERROR] No pixel size information available — please provide 'pixel_to_um' manually. Skipping region")
                        continue
            
            
                    # --- 3) Stage positions from LIF mosaic (raw) ---
                    try:
                        x_raw = np.array([float(p[2]) for p in mosaic], dtype=float)
                        y_raw = np.array([float(p[3]) for p in mosaic], dtype=float)
                        tiles_list = mosaic
                    except Exception as _e:
                        print(f"[WARN] Could not read LIF mosaic positions: {_e}")
                        x_raw = np.array([], dtype=float)
                        y_raw = np.array([], dtype=float)
                        tiles_list = []
            
                    # --- 4) Unit hint ---
                    # LIF mosaic stage coordinates have no declared physical unit in metadata.
                    # Leave this empty to let the hypothesis logic infer it automatically (usually 'pixels').
                    unit_hint_raw = ""
            
                    # --- 5) Write TileScanInfo XML via shared helper ---
                    _ = decide_and_write_tilescan(
                        x_raw=x_raw,
                        y_raw=y_raw,
                        image_dimensions=image_dimensions,
                        pixel_to_um_manual=pixel_to_um_manual,
                        pixel_to_um_calc=pixel_to_um_calc,
                        unit_hint_raw=unit_hint_raw,
                        off_tol=0.25,
                        tiles_iter=tiles_list,
                        app_name="LAS AF",
                        out_xml_path=metadata_directory / f"{image_name}.xml",
                        deconvolution_method=deconvolution_method,       
                        deconvolution_iterations=num_iterations
                    )
            
                    print(f"[INFO] Wrote LIF TileScanInfo for {image_name} (positions in µm)")
            
                except Exception as e:
                    print(f"[WARN] Could not parse/write LIF metadata for '{image_name}': {e}")

            # --- czi metadata ---
            elif mode == 'czi':
                try:
                    # --- 1) Collect stage positions (tile bounding boxes) ---
                    tiles_list = []
                    x_list, y_list = [], []
                    for m in range(dims.get("M", 1)):
                        try:
                            bb = czi.get_mosaic_tile_bounding_box(M=m, Z=0, C=0)
                            x, y = float(bb.x), float(bb.y)
                            tiles_list.append((m, 0, x, y))   # (FieldX, FieldY, PosX_raw, PosY_raw)
                            x_list.append(x)
                            y_list.append(y)
                        except Exception as e:
                            print(f"[WARN] Could not get bounding box for tile {m}: {e}")
            
                    if not tiles_list:
                        print("No CZI tile positions available to write TileScanInfo.")
            
                    x_raw = np.array(x_list, dtype=float)
                    y_raw = np.array(y_list, dtype=float)
                    
            
                    # --- 2) Parse XML metadata directly from czi.meta ---
                    meta_root = getattr(czi, "meta", None)
                    if meta_root is None:
                        print("CZI metadata XML not available (meta is None).")
            
                    if not isinstance(meta_root, ET.Element):
                        meta_root = ET.fromstring(meta_root)  # handles bytes/str
            
                    # Namespace-agnostic helpers
                    def _local(tag):
                        return tag.split('}')[-1] if '}' in tag else tag
            
                    def _iter(root, name):
                        for n in root.iter():
                            if _local(n.tag) == name:
                                yield n
            
                    def _first_text_path(root, path_elems):
                        node = root
                        for name in path_elems:
                            found = None
                            for ch in node:
                                if _local(ch.tag) == name:
                                    found = ch
                                    break
                            if found is None:
                                return None
                            node = found
                        return (node.text.strip() if node is not None and node.text else None)
            
                    # --- 3) Extract pixel size (prefer X/Y from Scaling/Items/Distance) ---
                    def _get_pixel_to_um(root):
                        vals_um = []
                        for dist in _iter(root, "Distance"):
                            axis = dist.get("Id") or dist.get("Dimension") or dist.get("Axis")
                            if axis not in ("X", "Y"):
                                continue
                            val_txt = None
                            unit_txt = None
                            for ch in dist:
                                lname = _local(ch.tag)
                                if lname in ("Value", "MeasuredValue") and ch.text:
                                    val_txt = ch.text.strip()
                                elif lname in ("DefaultUnit", "Unit") and ch.text:
                                    unit_txt = ch.text.strip().lower().replace("μ", "µ")
                            if val_txt:
                                try:
                                    v = float(val_txt)
                                    if unit_txt in (None, "", "m", "meter", "metre", "meters", "metres"):
                                        vals_um.append(v * 1e6)  # m → µm
                                    elif unit_txt in ("µm", "um", "micrometer", "micrometre", "micron", "microns"):
                                        vals_um.append(v)        # already µm
                                    elif unit_txt in ("nm", "nanometer", "nanometre", "nanometers", "nanometres"):
                                        vals_um.append(v * 1e-3) # nm → µm
                                    else:
                                        vals_um.append(v * 1e6)  # unknown → assume meters
                                except Exception:
                                    print(f"[WARN] Could not parse pixel size for axis {axis}: {val_txt}")
                        return (sum(vals_um) / len(vals_um)) if vals_um else None
            
                    pixel_to_um_calc = _get_pixel_to_um(meta_root)
                    if pixel_to_um_calc is not None:
                        print(f"[META] Pixel size from CZI metadata: {pixel_to_um_calc:.6f} µm/px")
                    else:
                        print("[WARN] No pixel size information available from metadata")
            
            
                    # --- 4) Extract objective magnification (numeric only) ---
                    mag = None
                    root = meta_root  # alias for findtext calls
            
                    for path in [
                        ".//Information/Instrument/Objectives/Objective/NominalMagnification",
                        ".//Information/Instrument/Objectives/Objective/ManufacturerData/Magnification",
                        ".//Scaling/Objectives/Objective/NominalMagnification",
                    ]:
                        txt = root.findtext(path)
                        if txt:
                            m = re.search(r"([\d.]+)", txt)
                            if m:
                                mag = float(m.group(1))
                                break
            
                    if mag:
                        print(f"[META] Objective magnification: {mag:g}x")
                    else:
                        print("[META] No magnification info found.")
            
                    # --- 5) Choose pixel size (prefer manual, warn if mismatch) ---
                    pixel_to_um_manual = float(pixel_to_um) if pixel_to_um is not None else None
                    effective_pixel_to_um = None
            
                    if pixel_to_um_calc and pixel_to_um_manual:
                        if not np.isclose(pixel_to_um_calc, pixel_to_um_manual, rtol=0.02):
                            print(f"[WARN] Manual pixel size ({pixel_to_um_manual:.6f}) differs "
                                  f"from metadata value ({pixel_to_um_calc:.6f}).")
                        effective_pixel_to_um = pixel_to_um_manual
                        print(f"[META] Using manual pixel size: {effective_pixel_to_um:.6f} µm/px")
                    elif pixel_to_um_manual:
                        effective_pixel_to_um = pixel_to_um_manual
                        print(f"[META] Using manual pixel size: {effective_pixel_to_um:.6f} µm/px")
                    elif pixel_to_um_calc:
                        effective_pixel_to_um = pixel_to_um_calc
                        print(f"[META] No manual pixel size provided — using metadata pixel size: {effective_pixel_to_um:.6f} µm/px")
                    else:
                        print("[ERROR] No pixel size information available — please provide 'pixel_to_um' manually. Skipping region")
                        continue
                        
                    # --- 6) Unit hint for stage positions (for TileScanInfo generation) ---
                    # Mosaic tile bounding boxes lack a declared unit; leave empty so the hypothesis test
                    # infers the correct interpretation (often pixels → µm/px).
                    unit_hint_raw = ""
            
                    # --- 7) Write TileScanInfo XML via helper ---
                    _ = decide_and_write_tilescan(
                        x_raw=x_raw,
                        y_raw=y_raw,
                        image_dimensions=image_dimensions,
                        pixel_to_um_manual=pixel_to_um_manual,
                        pixel_to_um_calc=pixel_to_um_calc,
                        unit_hint_raw=unit_hint_raw,
                        off_tol=0.25,
                        tiles_iter=tiles_list,
                        app_name="Zeiss CZI",
                        out_xml_path=metadata_directory / f"{region}.xml",
                        deconvolution_method=deconvolution_method,       
                        deconvolution_iterations=num_iterations
                        )
            
                except Exception as e:
                    print(f"[WARN] Could not parse/write CZI metadata: {e}")
                                                   
            # --- nd2 metadata ---
            elif mode == 'nd2':
                data = ET.Element("Data")
                image_elem = ET.SubElement(data, "Image", TextDescription="")
            
                attachment = ET.SubElement(
                    image_elem,
                    "Attachment",
                    Name="TileScanInfo",
                    Application="NIS-Elements",
                    FlipX="0", FlipY="0", SwapXY="0"
                )
            
                if coords:
                    print(f"Extracting {len(coords)} stage coordinate(s) from ND2")
                    for idx, (x, y) in enumerate(coords):
                        ET.SubElement(
                            attachment, "Tile",
                            FieldX=str(idx),
                            FieldY="0",
                            PosX=f"{x:.10f}",
                            PosY=f"{y:.10f}"
                        )
                else:
                    if n_tiles == 1:
                        print("[INFO] ND2 file has no stage coords but only one tile → writing dummy (0,0)")
                        ET.SubElement(
                            attachment, "Tile",
                            FieldX="0",
                            FieldY="0",
                            PosX="0.0000000000",
                            PosY="0.0000000000"
                        )
                    else:
                        raise ValueError(
                            f"No stage coordinates found in ND2 file {filepath.name}, "
                            f"but multiple tiles detected ({n_tiles}). Cannot generate OME-TIFF without positions."
                        )
            
                # Save XML file
                xml_path = metadata_directory / f"{region}.xml"
                tree = ET.ElementTree(data)
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)
                print(f"Metadata XML written: {xml_path}")

            
            # ----- STEP 3: SKIP EXISTING FILES -----
            print("\033[96mProcessing files\033[0m")
            # Check what files are expected to exist
            expected_files = [
                            (mipped_directory if mip else stacked_directory) / f"Cycle{cycle}_s{tile}_ch{channel}.tif"
                            for tile in tiles
                            for channel in channels
                        ]
                        
            print(f"Expected number of output files in {mipped_directory if mip else stacked_directory}: {len(expected_files)} ({len(tiles)} tiles × {len(channels)} channels)")
            
            # Identify which files are missing
            missing_files = [f for f in expected_files if not f.exists()]
            
            if not missing_files:
                print(f"All expected files for Cycle {cycle} already exist in {mipped_directory if mip else stacked_directory} directory. Skipping processing.")
                continue
            
            # Extract unique tile numbers from missing file names
            missing_tiles = sorted(set(
                match.group(1)
                for f in missing_files
                if (match := re.search(r'_s(\d+)_ch', f.name))
            ))
            
            # Update the tiles list to only those with missing outputs
            tiles = missing_tiles
            
            print(f"{len(tiles)} tile(s) have missing outputs. Proceeding with processing only these.")

            # ----- STEP 4: GENERATE PSFS FOR ALL CHANNELS -----
            print("Calculating the PSF")
        
            if deconvolution_method is None:
                print("Skipping PSF generation — deconvolution method is None.")
                psf_dict = {}  # Initialize empty dict for compatibility
        
            elif deconvolution_method == 'redlionfish': 
                psf_dict = {}
                for channel, info in PSF_metadata['channels'].items():
        
                    print(f"Generating PSF for channel {channel}")
                    psf_volume = fd_psf.GibsonLanni(
                        na=float(PSF_metadata['na']),
                        m=float(PSF_metadata['m']),
                        ni0=float(PSF_metadata['ni0']),
                        res_lateral=float(PSF_metadata['res_lateral']),
                        res_axial=float(PSF_metadata['res_axial']),
                        wavelength=float(info['wavelength']),
                        size_x=image_dimensions[0],
                        size_y=image_dimensions[1],
                        size_z=size_z
                    ).generate()
        
                    psf_dict[channel] = psf_volume  
                    
            elif deconvolution_method == 'deconwolf':
        
                # Prepare output directory for PSF files
                psf_dir = cycle_directory / 'PSF'
                psf_dir.mkdir(parents=True, exist_ok=True)
                
                psf_dict = {}
                # Generate PSF files for each channel using the external generate_psf function
                for channel, info in PSF_metadata['channels'].items():
                    wavelength_nm = float(info['wavelength']) * 1000      # Convert wavelength to nanometers
                    psf_filename = psf_dir / f"PSF_channel_{channel}.tif" # Output file path for this channel's PSF
                    
                    # Call PSF generation function with parameters in nanometers
                    generate_psf(
                        psf_output=psf_filename,
                        resxy=PSF_metadata['res_lateral'] * 1000,         # Lateral resolution in nm
                        resz=PSF_metadata['res_axial'] * 1000,            # Axial resolution in nm
                        wavelength=wavelength_nm,
                        NA=PSF_metadata['na'],
                        ni=PSF_metadata['ni0'])
                    
                    # Store path to generated PSF file in dictionary
                    psf_dict[channel] = psf_filename
        
            # ----- STEP 5: DECONVOLVE EACH TILE AND CHANNEL -----
            print("Single tile imaging." if n_tiles == 1 else f"Number of tiles to process: {n_tiles}")
    
            # Prepare directory to save stacked images
            
            stacked_directory.mkdir(exist_ok=True, parents=True)
    
            # Loop over each tile (spatial subdivision of the image)
            for tile in tqdm(tiles, desc="Processing tiles", leave=False):
               
                # Loop over each fluorescence channel in the PSF metadata
                for channel in channels:
                    print(f"\033[90m[\033[96mCycle {cycle}\033[90m] Tile {tile}, Channel {channel}...\033[0m")
                    tile_channel_start = time.time()
                    
                    # Choose output path depending on whether MIP (max intensity projection) is requested
                    output_file_path = (mipped_directory if mip else stacked_directory) / f'Cycle{cycle}_s{tile}_ch{int(channel)}.tif'
        
                    # Skip processing if output file already exists
                    if output_file_path.exists():
                        print(f"File {output_file_path} already exists. Skipping.")
                        continue
        
                    # Load stacked images depending on mode
                    if mode in ('tif_autosaved', 'tif_exported'):
                        channel_files = tile_channel_files.get((tile, channel), [])
                        stacked_images = np.stack([tifffile.imread(f) for f in channel_files])
                    elif mode == 'lif':
                        # For lif files, iterate through z-planes in the tile and channel
                        z_planes = [np.array(z_frame) for z_frame in image.get_iter_z(m=tile, c=channel)]
                        stacked_images = np.stack(z_planes, axis=0)
                    
                    elif mode == 'czi':
                        dim_sizes = dict(zip(czi.dims, czi.size))
                        max_Z = dim_sizes.get("Z", 1)
                    
                        z_planes = []
                        for z in range(max_Z):
                            try:
                                # Build safe kwargs — only include dimensions that exist
                                kwargs = {"Z": int(z)}
                                if "S" in dim_sizes:
                                    kwargs["S"] = int(region_index)
                                if "M" in dim_sizes:
                                    kwargs["M"] = int(tile)
                                if "C" in dim_sizes:
                                    kwargs["C"] = int(channel)
                                if "B" in dim_sizes:
                                    kwargs["B"] = 0  # base resolution if pyramid exists
                    
                                # Keep only valid keys that appear in czi.dims
                                kwargs = {k: v for k, v in kwargs.items() if k in czi.dims}
                    
                                # Try reading — if overspecified, drop S and retry
                                try:
                                    img, shp = czi.read_image(**kwargs)
                                except Exception as e:
                                    if "S value" in str(e):
                                        kwargs.pop("S", None)
                                        img, shp = czi.read_image(**kwargs)
                                    else:
                                        raise
                    
                                z_planes.append(img.squeeze())
                    
                            except RuntimeError as e:
                                if "Not enough data read" in str(e):
                                    print(f"[WARN] Missing plane at Tile {tile}, Channel {channel}, Z {z}. Using {len(z_planes)} planes.")
                                    break
                                raise
                    
                        if not z_planes:
                            print(f"[WARN] No Z planes for Tile {tile}, Channel {channel} (Cycle {cycle}). Skipping.")
                            continue
                    
                        stacked_images = np.stack(z_planes, axis=0)  # (Z_actual, Y, X)


                    elif mode == 'nd2':
                        # ND2 array has shape (M, Z, C, Y, X)
                        # Select one tile (M), all Z, one channel (C), full XY
                        stacked_images = arr[int(tile), :, int(channel), :, :]

        
                    # Deconvolution with RedLionFish method
                    if deconvolution_method == 'redlionfish':
                        deconvolved_images = rl.doRLDeconvolutionFromNpArrays(stacked_images, psf_dict[str(channel)], niter=num_iterations)
                        # Save max projection if MIP requested, otherwise full stack
                        processed_img = np.max(deconvolved_images, axis=0).astype('uint16') if mip else deconvolved_images.astype('uint16')
                        tifffile.imwrite(output_file_path, processed_img)
                        print(f"{'Mipped' if mip else 'Stacked'} images saved in directory: {mipped_directory if mip else stacked_directory}")
                        
        
                    # Deconvolution with Deconwolf method
                    elif deconvolution_method == 'deconwolf':
                        # Create temporary directory for Deconwolf input
                        dw_input_directory = cycle_directory / 'deconwolf input tmp'
                        dw_input_directory.mkdir(parents=True, exist_ok=True)
                        
                        dw_input_path = dw_input_directory / f'Cycle{cycle}_s{tile}_ch{channel}.tif'
                        tifffile.imwrite(dw_input_path, stacked_images)    # Write input stack for Deconwolf
                        
                        dw_output_path = stacked_directory / f'Cycle{cycle}_s{tile}_ch{channel}.tif'
        
                        # Run Deconwolf deconvolution externally
                        deconvolve_image(
                            input_image=dw_input_path,
                            psf_image=psf_dict[str(channel)],
                            output_image=dw_output_path,
                            iterations=20,
                            tilesize=chunk_size)
        
                        # If MIP requested, generate max projection from deconvolved images and save
                        if mip:
                            deconvolved_images = tifffile.imread(dw_output_path)
                            mipped_img = np.max(deconvolved_images, axis=0).astype('uint16')
                            tifffile.imwrite(output_file_path, mipped_img)
                            print(f"Mipped images saved in directory: {mipped_directory}")
                            
                        else:
                            print(f"Stacked files saved in directory: {stacked_directory}")
        
                        # Remove temporary Deconwolf input directory after processing
                        if dw_input_directory.exists():
                            shutil.rmtree(dw_input_directory)
                            print(f"Deleted directory: {dw_input_directory}")
        
                    # No deconvolution, just save max projection or stack
                    elif deconvolution_method is None:
                        processed_img = np.max(stacked_images, axis=0).astype('uint16') if mip else stacked_images.astype('uint16')
                        tifffile.imwrite(output_file_path, processed_img)
                        print(f"{'Mipped' if mip else 'Stacked'} images saved in directory: {mipped_directory if mip else stacked_directory}")
    
    
                    tile_channel_end = time.time()
                    print(f"\033[1;37m[Timing] Full deconvolution/mipping cycle for Tile {tile}, Channel {channel} took {tile_channel_end - tile_channel_start:.2f} seconds\033[0m")
            
            # After all tiles and channels are done
            if mip and stacked_directory.exists():
                shutil.rmtree(stacked_directory)
                print(f"Deleted stacked directory: {stacked_directory}")
    
    print("\n🔹 region_directories:\n", region_directories)
    
    return region_directories




def mipped_to_OME_tiffs(region_directories, cycles, pixel_to_um=None):
    """
    Convert per-tile MIPs into an OME-TIFF with spatial metadata.
    Assumes metadata Tile positions (PosX/PosY) are already in microns.
    No unit detection or scaling is performed here; we only read/choose pixel size.
    """

    print("\033[1;96mConverting to OME-TIFFs\033[0m")

    for cycle in cycles:
        print(f"\033[1;90mProcessing Cycle {cycle}\033[0m")

        for region_directory in region_directories:
            region_suffix = region_directory[-2:]
            if re.match(r"R\d+", region_suffix):
                print(f"\033[1mProcessing {region_suffix}\033[0m")

            region_directory = Path(region_directory)
            cycle_directory    = region_directory / 'preprocessing' / f'Cycle{cycle}'
            mipped_directory   = cycle_directory / '1_mipped'
            ome_tiff_directory = cycle_directory / '2_ome_tiffs'
            ome_tiff_path      = ome_tiff_directory / f'Cycle{cycle}.ome.tiff'
            metadata_directory = cycle_directory / 'MetaData'

            ome_tiff_directory.mkdir(parents=True, exist_ok=True)

            if ome_tiff_path.exists():
                print(f"OME-TIFF already exists: {ome_tiff_path}. Skipping.")
                continue

            tif_files = list(mipped_directory.glob('*.tif'))
            if not tif_files:
                print(f"No TIFF files found in {mipped_directory}. Skipping.")
                continue

            # Build file index: tile -> channel -> path
            file_index = defaultdict(dict)
            for f in tif_files:
                m = re.search(r'_s0*(\d+)_ch0*(\d+)', f.name, re.IGNORECASE)
                if m:
                    tile, channel = map(int, m.groups())
                    file_index[tile][channel] = f


            tiles = sorted(file_index.keys(), key=int)
            channels = sorted({ch for chs in file_index.values() for ch in chs}, key=int)
            if not channels:
                print(f"[WARN] No channels found in {mipped_directory}. Check filename pattern.")
                continue

            # ----- Load positions (already µm) -----
            md_candidates = sorted(metadata_directory.glob('*.xml'))
            if not md_candidates:
                print(f"No metadata found in {metadata_directory}. Skipping.")
                continue

            # Prefer a file whose stem matches the region token in the image filenames (e.g., "Region1_*")
            region_token = None
            if tiles:
                any_tile_any_ch = next(iter(file_index[tiles[0]].values()))
                region_token = any_tile_any_ch.stem.split('_')[0]  # ".../MyRegion_s0_ch0.tif" → "MyRegion"
            md_file = next((p for p in md_candidates if region_token and region_token in p.stem), md_candidates[0])

            try:
                root = ET.parse(md_file).getroot()
            except Exception as e:
                print(f"[WARN] Could not parse XML {md_file.name}: {e}. Skipping.")
                continue

            att = root.find(".//Attachment[@Name='TileScanInfo']")
            unit_attr = (att.attrib.get("Unit") if att is not None else None)
            if unit_attr and unit_attr.lower() not in ("µm", "um", "micron", "microns", "micron(s)", "micrometer", "micrometers"):
                print(f"[WARN] Metadata Unit='{unit_attr}' but this function assumes microns. Proceeding as µm.")

            # --- Read pixel size (µm/px) from XML if available (or compute from TileWidthUm/TileWidthPx) ---
            pixel_to_um_from_xml = None
            pixel_source = None
            if att is not None:
                raw_px = (att.attrib.get("PixelSizeUm") or "").strip()
                if raw_px:
                    cleaned = re.sub(r"[^\d.,+\-eE]", "", raw_px).replace(",", ".")
                    try:
                        pixel_to_um_from_xml = float(cleaned)
                        pixel_source = "XML"
                        print(f"[META] Pixel size read from XML: {pixel_to_um_from_xml:.6f} µm/px")
                    except Exception:
                        print(f"[WARN] Could not parse PixelSizeUm='{raw_px}' in {md_file.name}")
                        pixel_to_um_from_xml = None

                # Fallback: compute from TileWidthUm/TileWidthPx if present
                if pixel_to_um_from_xml is None:
                    tpx = att.attrib.get("TileWidthPx")
                    tum = att.attrib.get("TileWidthUm")
                    if tpx and tum:
                        try:
                            tpx_i = int(re.sub(r"[^\d]", "", tpx))
                            tum_f = float(re.sub(r"[^\d.,+\-eE]", "", tum).replace(",", "."))
                            if tpx_i > 0:
                                pixel_to_um_from_xml = tum_f / tpx_i
                                pixel_source = "XML (TileWidthUm/TileWidthPx)"
                                print(f"[META] Pixel size computed from XML widths: {pixel_to_um_from_xml:.6f} µm/px")
                        except Exception:
                            pass

            # Choose effective pixel_to_um: XML > argument
            effective_pixel_to_um = None
            if pixel_to_um_from_xml is not None:
                effective_pixel_to_um = pixel_to_um_from_xml
                source = pixel_source or "XML"
            elif pixel_to_um is not None:
                effective_pixel_to_um = float(pixel_to_um)
                source = "argument"
            else:
                source = None

            if source:
                print(f"[META] Using PixelSizeUm from {source}: {effective_pixel_to_um:.6f} µm/px")
            else:
                print("[WARN] No pixel size available (XML/argument). Overlap estimation and OME scale will be omitted.")

            # Sort tiles by FieldY then FieldX (raster order)
            tile_nodes = root.findall(".//Tile")
            if not tile_nodes:
                print(f"[WARN] No <Tile> elements found in {md_file.name}. Skipping.")
                continue

            tile_nodes_sorted = sorted(
                tile_nodes,
                key=lambda e: (int(e.attrib.get("FieldY", "0")), int(e.attrib.get("FieldX", "0")))
            )

            # Positions (normalize origin to 0,0)
            try:
                x_um_raw = np.array([float(n.attrib['PosX']) for n in tile_nodes_sorted], dtype=float)
                y_um_raw = np.array([float(n.attrib['PosY']) for n in tile_nodes_sorted], dtype=float)
            except Exception as e:
                print(f"[WARN] Malformed <Tile> positions in {md_file.name}: {e}. Skipping.")
                continue

            x_um = x_um_raw - x_um_raw.min() if x_um_raw.size else x_um_raw
            y_um = y_um_raw - y_um_raw.min() if y_um_raw.size else y_um_raw

            # Sanity: counts match?
            if len(x_um) != len(tiles):
                print(f"[WARN] Tile count mismatch: metadata has {len(x_um)} tiles, files show {len(tiles)} tiles.")
                n = min(len(x_um), len(tiles))
                x_um = x_um[:n]; y_um = y_um[:n]; tiles = tiles[:n]

            # Image size (from any file)
            first_tile_dict = next(iter(file_index.values()))
            first_img_path  = next(iter(first_tile_dict.values()))
            try:
                height_px, width_px = tifffile.imread(first_img_path).shape
            except Exception as e:
                print(f"[WARN] Could not read image to get dimensions ({first_img_path}): {e}. Skipping.")
                continue

            # Tile width in µm if we have pixel size
            tile_width_um = None
            if effective_pixel_to_um is not None:
                tile_width_um = width_px * float(effective_pixel_to_um)

            # --- Robust overlap estimation (median of valid tile steps) ---
            def robust_step(vals_um, expected_width_um):
                """Estimate spacing using robust filtering near the expected width."""
                if vals_um is None or len(vals_um) < 2:
                    return None
                u = np.unique(np.round(vals_um, 9))
                if u.size < 2:
                    return None
                d = np.diff(np.sort(u))
                d = d[d > 0]
                if d.size == 0:
                    return None
                if expected_width_um:
                    lo, hi = 0.5 * expected_width_um, 1.2 * expected_width_um
                    band = d[(d >= lo) & (d <= hi)]
                    if band.size == 0:
                        # fallback: largest 30% of steps
                        k = max(1, int(0.3 * d.size))
                        band = np.sort(d)[-k:]
                    return float(np.median(band))
                # no expected width: fall back to the median of the largest 30% of steps
                k = max(1, int(0.3 * d.size))
                return float(np.median(np.sort(d)[-k:]))

            def ov_pct(step_um, width_um):
                if step_um is None or not width_um:
                    return None
                return (1.0 - step_um / width_um) * 100.0

            dx = robust_step(x_um, tile_width_um)
            dy = robust_step(y_um, tile_width_um)

            if effective_pixel_to_um is not None:
                if dx is not None:
                    ovx = ov_pct(dx, tile_width_um)
                    print(f"[INFO] Tile spacing X≈{dx:.2f} µm; est width≈{tile_width_um:.2f} µm → overlap≈{ovx:.1f}%")
                if dy is not None:
                    ovy = ov_pct(dy, tile_width_um)
                    print(f"[INFO] Tile spacing Y≈{dy:.2f} µm; est width≈{tile_width_um:.2f} µm → overlap≈{ovy:.1f}%")
            else:
                if dx is not None:
                    print(f"[INFO] Tile spacing X≈{dx:.2f} µm (pixel size unknown)")
                if dy is not None:
                    print(f"[INFO] Tile spacing Y≈{dy:.2f} µm (pixel size unknown)")

            # Optional: write pixel-grid indices (debug) if we know pixel size
            if effective_pixel_to_um is not None:
                x_idx = (x_um / float(effective_pixel_to_um))
                y_idx = (y_um / float(effective_pixel_to_um))
                pd.DataFrame({'x': x_idx, 'y': y_idx}).to_csv(
                    ome_tiff_directory / f'Cycle{cycle}_coords.csv', index=False
                )

            # ----- Write OME-TIFF -----
            with tifffile.TiffWriter(ome_tiff_path, bigtiff=True, ome=True) as tif:
                for tile_idx, tile in enumerate(tiles):
                    image_stack = np.empty((len(channels), height_px, width_px), dtype=np.uint16)

                    for ci, ch in enumerate(channels):
                        try:
                            image_stack[ci] = tifffile.imread(file_index[tile][ch]).astype(np.uint16)
                        except Exception as e:
                            print(f"[WARN] Tile {tile}, Channel {ch} missing or unreadable: {e}")
                            image_stack[ci] = np.zeros((height_px, width_px), dtype=np.uint16)

                    posX_um = float(x_um[tile_idx]) if tile_idx < len(x_um) else 0.0
                    posY_um = float(y_um[tile_idx]) if tile_idx < len(y_um) else 0.0

                    # Build OME metadata
                    pixels_md = {}
                    if effective_pixel_to_um is not None:
                        pixels_md = {
                            'PhysicalSizeX': float(effective_pixel_to_um),
                            'PhysicalSizeXUnit': 'µm',
                            'PhysicalSizeY': float(effective_pixel_to_um),
                            'PhysicalSizeYUnit': 'µm',
                        }

                    plane_md = {
                        'PositionX': [posX_um] * len(channels),
                        'PositionY': [posY_um] * len(channels)
                    }

                    metadata = {'Pixels': pixels_md, 'Plane': plane_md} if pixels_md else {'Plane': plane_md}
                    tif.write(image_stack, metadata=metadata)

            print(f"[DONE] Wrote OME-TIFF: {ome_tiff_path}")
#---------


def align_and_stitch(
    region_directories,
    cycles,
    n_total_cycles,
    align_channel=4, 
    flip_x=False, 
    flip_y=True, 
    output_channels=None, 
    maximum_shift=500, 
    filter_sigma=5, 
    pyramid=False,
    tile_size=None,
    ffp=None,
    dfp=None,
    plates=False,
    quiet=True,
    version=False):
    """
    Wrapper function for the Ashlar tool for image alignment and mosaicking.

    Args:
        region_directories (list): List of directories for all regions.
        cycles (list): List of cycle numbers to identify the correct TIFFs.
        align_channel (int): Channel to use for alignment.
        flip_x (bool): Flip images along the X-axis.
        flip_y (bool): Flip images along the Y-axis.
        output_channels (list or None): List of channels to include in output.
        maximum_shift (int): Max shift in pixels allowed for tile alignment.
        filter_sigma (float): Sigma for Gaussian filter used in alignment.
        pyramid (bool): Whether to generate pyramid TIFFs.
        tile_size (int or None): Tile size for pyramid TIFFs. Required if pyramid=True.
        ffp (list or None): Flat-field profiles.
        dfp (list or None): Dark-field profiles.
        plates (bool): Whether to use plate processing mode.
        quiet (bool): Suppress verbose output.
        version (bool): Print version (not used in this wrapper).

    Returns:
        int: 1 on error, otherwise result of Ashlar processing.
    """

    print("\033[1;96mAligning and stitching tiles\033[0m")
    print("\033[1mProcessing all cycles \033[0m")
    ashlar.configure_terminal()

    maximum_shift=200
    filter_sigma=3
    
    for region_directory in region_directories:
        region_suffix = region_directory[-2:]
        if re.match(r"R\d+", region_suffix):
            print(f"\033[1mProcessing {region_suffix}\033[0m")
        
        region_directory = Path(region_directory)

        # --- STEP 1: Make directories for each cycle ---
        for cycle in cycles:
            cycle_directory = region_directory / 'preprocessing' / f'Cycle{cycle}'
            ome_tiff_directory = cycle_directory / '2_ome_tiffs'
            stitched_directory = cycle_directory / '3_stitched'
            stitched_directory.mkdir(exist_ok=True)

        # --- STEP 2: Collect OME-TIFFs and validate cycles ---
        ome_tiffs = natsorted([
            f for f in (region_directory / "preprocessing").rglob("*.ome.tiff")
        ])
        
        # Extract cycle numbers from filenames like Cycle1_xxx.ome.tif
        found_cycles = sorted(set(
            int(re.search(r"Cycle(\d+)", f.name).group(1))
            for f in ome_tiffs if re.search(r"Cycle(\d+)", f.name)
        ))
        
        # Sanity check: must match the expected number of cycles
        if len(found_cycles) != n_total_cycles:
            print(f"Expected {n_total_cycles} cycles, but found {len(found_cycles)}: {found_cycles}")
            raise RuntimeError(
                f"Cycle mismatch. Expected {n_total_cycles} cycles, but found {found_cycles}."
            )
        else:
            print(f"Found all {n_total_cycles} cycles: {found_cycles}")



        # --- STEP 3: Define expected outputs ---
        # Get number of channels from first OME-TIFF
        with tifffile.TiffFile(ome_tiffs[0]) as tif:
            n_channels = tif.series[0].shape[tif.series[0].axes.index('C')]

       # Define the stitched output pattern as a format string
        ashlar_filename_pattern = str(
            region_directory / "preprocessing" / "Cycle{cycle}" / "3_stitched" / "Cycle{cycle}_ch{channel}.tif"
        )
        
        # Build a list of expected outputs for all cycles + channels
        expected_outputs = [
            Path(ashlar_filename_pattern.format(cycle=cyc, channel=ch))
            for cyc in cycles
            for ch in range(n_channels)
        ]


        # Skip only if *all* expected outputs exist
        if all(p.exists() for p in expected_outputs):
            print(f"Stitched images already exist for all {n_total_cycles} cycles. Skipping.")
            continue
       

        # --- STEP 4: Validate Ashlar parameters ---
        # **Validate pyramid/tile size configuration**
        warnings.filterwarnings("ignore")
        if tile_size and not pyramid:
            ashlar.print_error("--tile-size can only be used with --pyramid")
            continue
        if pyramid and tile_size is None:
            ashlar.print_error("--tile-size must be specified when --pyramid is enabled")
            continue
    
        # **Normalize FFP/DFP paths if provided**
        ffp_paths = ffp
        if ffp_paths:
            if len(ffp_paths) not in (0, 1, len(ome_tiffs)):
                ashlar.print_error(f"Wrong number of flat-field profiles. Must be 1, or {len(ome_tiffs)}")
                continue
            if len(ffp_paths) == 1:
                ffp_paths *= len(ome_tiffs)
    
        dfp_paths = dfp
        if dfp_paths:
            if len(dfp_paths) not in (0, 1, len(ome_tiffs)):
                ashlar.print_error(f"Wrong number of dark-field profiles. Must be 1, or {len(ome_tiffs)}")
                continue
            if len(dfp_paths) == 1:
                dfp_paths *= len(ome_tiffs)
    
        # **Set Ashlar aligner and mosaic parameters**
        aligner_args = {
            'channel': align_channel,
            'verbose': not quiet,
            'max_shift': maximum_shift,
            'filter_sigma': filter_sigma
        }
    
        mosaic_args = {}
        if output_channels:
            mosaic_args['channels'] = output_channels
        if pyramid:
            mosaic_args['tile_size'] = tile_size
        if not quiet:
            mosaic_args['verbose'] = True

        # Define temporary Ashlar output pattern (0-indexed cycles)
        tmp_path = region_directory / "preprocessing" / "ashlar_tmp"
        tmp_pattern = str(
            tmp_path / "Cycle{cycle}_ch{channel}.tif"
        )
        Path(tmp_pattern).parent.mkdir(parents=True, exist_ok=True)
    
        # --- STEP 5: Run Ashlar ---
        try:
            ome_tiff_files = [str(f) for f in ome_tiffs]
            if not ome_tiff_files:
                raise RuntimeError("No OME-TIFF inputs found for this region.")
        
            if plates:
                rc = ashlar.process_plates(
                    ome_tiff_files,
                    None,                  # no base output dir
                    tmp_pattern,           # full pattern string
                    flip_x, flip_y,
                    ffp_paths, dfp_paths,
                    0.0,                   # ✅ barrel_correction explicitly included
                    aligner_args, mosaic_args,
                    pyramid, quiet
                )
            else:
                rc = ashlar.process_single(
                    ome_tiff_files,
                    tmp_pattern,           # same pattern string
                    flip_x, flip_y,
                    ffp_paths, dfp_paths,
                    0.0,                   # ✅ barrel_correction explicitly included
                    aligner_args, mosaic_args,
                    pyramid, quiet
                )
        
            # Optional: Check return code
            if rc not in (None, 0):
                raise RuntimeError(f"Ashlar returned non-zero status code: {rc}")
        
        except ashlar.ProcessingError as e:
            ashlar.print_error(str(e))
            continue
        
        except Exception as e:
            ashlar.print_error(f"Unexpected error during Ashlar run: {e}")
            continue
        
         
        # --- STEP 6: Remap cycles provided by user ---
        tmp_dir = Path(tmp_pattern).parent
        for cyc_idx, cyc in enumerate(cycles):
            stitched_dir = region_directory / "preprocessing" / f"Cycle{cyc}" / "3_stitched"
            stitched_dir.mkdir(parents=True, exist_ok=True)


            for ch in range(n_channels):
                tmp_file = tmp_dir / f"Cycle{cyc_idx}_ch{ch}.tif"
                if not tmp_file.exists():
                    raise FileNotFoundError(f"Expected {tmp_file} not found")
                final_file = stitched_dir / f"Cycle{cyc}_ch{ch}.tif"
                tmp_file.rename(final_file)
        print(f"Moved stitched and aligned images from {tmp_path} → {stitched_dir}")

        # Clean up temporary folder
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        
    
def retile_stitched_images(
    region_directories,
    cycles,
    tile_dimension=6000
):
    """
    Tiles stitched .tif images from a directory and saves them with a specific naming convention.

    Args:
        region_directories (list of Path): List of region base directories.
        cycle (int): Cycle number for naming.
        tile_dimension (int): Tile dimension size. Default 6000.

    Returns:
        None. Saves tiled images and tile positions CSV.
    """
    print(f"\033[1;96mRetiling stitched images\033[0m")

    for cycle in cycles:
        print(f"\033[1;90mProcessing Cycle {cycle}\033[0m")

        for region_directory in region_directories:
            region_suffix = region_directory[-2:]
            if re.match(r"R\d+", region_suffix):
                print(f"\033[1mProcessing {region_suffix}\033[0m")
            
            region_directory = Path(region_directory)
    
            cycle_directory = region_directory / 'preprocessing' / f'Cycle{cycle}'
            stitched_directory = cycle_directory / '3_stitched'
            retiled_directory = cycle_directory / '4_retiled'
            retiled_directory.mkdir(exist_ok=True, parents=True)
    
            tif_files = sorted([
                f for f in stitched_directory.iterdir()
                if f.is_file() and f.suffix == '.tif'
            ])
    
            if not tif_files:
                print(f"No stitched TIFFs found for cycle {cycle} in {stitched_directory}")
                continue
    
            # === Pre-check: Skip if all expected tile files already exist ===
            sample_img = tifffile.imread(tif_files[0])  # input stitched image
            pad_height = math.ceil(sample_img.shape[0] / tile_dimension) * tile_dimension - sample_img.shape[0]
            pad_width = math.ceil(sample_img.shape[1] / tile_dimension) * tile_dimension - sample_img.shape[1]
            padded_height = sample_img.shape[0] + pad_height
            padded_width = sample_img.shape[1] + pad_width
            
            expected_tiles_per_img = (padded_height // tile_dimension) * (padded_width // tile_dimension)
            expected_total_tiles = expected_tiles_per_img * len(tif_files)
    
            existing_tiles = list(retiled_directory.glob(f'Cycle{cycle}_s*_ch*.tif'))
    
            # If the number of tiles is correct, only then sample-check 1–2 tile shapes
            if len(existing_tiles) == expected_total_tiles:
                sample_tile = tifffile.imread(existing_tiles[0])
                if sample_tile.shape != (tile_dimension, tile_dimension):
                    print(f"[WARN] Sample tile shape mismatch: expected ({tile_dimension}, {tile_dimension}), got {sample_tile.shape}")
                    for tile in existing_tiles:
                        tile.unlink()
                    print(f"Reprocessing due to tile shape mismatch.")
                else:
                    print(f"All expected tiles found and shape of first tile is correct (tile_dimension = {tile_dimension}) in {retiled_directory}. Skipping.")
                    continue
            else:
                print(f"Missing tiles (expected {expected_total_tiles}, found {len(existing_tiles)}). Reprocessing all.")
                for tile in existing_tiles:
                    tile.unlink()
    
    
            # === Begin tiling ===
            x_positions = []
            y_positions = []
    
            for tif_path in tif_files:
                try:
                    image = tifffile.imread(tif_path)
                    print(f"Tiling: {tif_path.name}")
    
                    pad_height = math.ceil(image.shape[0] / tile_dimension) * tile_dimension - image.shape[0]
                    pad_width = math.ceil(image.shape[1] / tile_dimension) * tile_dimension - image.shape[1]
    
                    image_padded = cv2.copyMakeBorder(
                        image,
                        top=0, bottom=pad_height,
                        left=0, right=pad_width,
                        borderType=cv2.BORDER_CONSTANT
                    )
    
                    img_height, img_width = image_padded.shape
                    nrows = img_height // tile_dimension
                    ncols = img_width // tile_dimension
    
                    tiled_array = image_padded.reshape(nrows, tile_dimension, ncols, tile_dimension)
                    tiled_array = tiled_array.swapaxes(1, 2)
    
                    filename_stem = tif_path.stem
                    channel_match = re.search(r'ch(\d+)', filename_stem)
                    channel = int(channel_match.group(1)) if channel_match else 0
    
                    tile_count = 0
                    # Reset coordinates for each image (only last one will remain)
                    x_positions, y_positions = [], []

                    for row in range(nrows):
                        for col in range(ncols):
                            x_positions.append(col * tile_dimension)
                            y_positions.append(row * tile_dimension)

                            tile_img = tiled_array[row, col]
                            tile_filename = retiled_directory / f'Cycle{cycle}_s{tile_count}_ch{channel}.tif'
                            tifffile.imwrite(tile_filename, tile_img)
                            tile_count += 1
    
                except Exception as e:
                    print(f"[ERROR] Processing {tif_path.name}: {e}")
                    continue

            # Always overwrite the coordinates CSV with the most recent image's tile grid
            tile_positions_df = pd.DataFrame({'x': x_positions, 'y': y_positions})
            coords_csv_path = retiled_directory / f'Cycle{cycle}_retiled_coords.csv'
            tile_positions_df.to_csv(coords_csv_path, header=False, index=False)
 
    
            print(f"Tiling complete. Positions saved to {coords_csv_path}")
    






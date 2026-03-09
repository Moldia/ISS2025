"""
Microscopy preprocessing pipeline 

Goals
-----
- Support input formats in one unified pipeline:
    TIFF (.tif), Leica LIF (.lif), Zeiss CZI (.czi), Nikon ND2 (.nd2)
- Keep the "main" pipeline readable by pushing format-specific logic into handlers
- Centralize TileScanInfo metadata writing via decide_and_write_tilescan()
- Preserve existing behavior:
    deconvolution (RedLionFish / Deconwolf / None), MIP, OME-TIFF, Ashlar stitching, retiling

Key Contracts
-------------
- decide_and_write_tilescan() is the *only* TileScanInfo XML writer.
- TileScanInfo output positions are ALWAYS written in microns (µm), regardless of input units.
- Handlers that provide mosaic/stage positions must ultimately supply STRICT 5-tuples:
    (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
  where TileIndex MUST match the tile ids used by downstream filenames (e.g. `_s{tile}`).

Notes
-----
- External dependencies assumed installed:
  tifffile, numpy, pandas, cv2, tqdm, natsort, aicspylibczi, readlif, nd2, ashlar, skimage
- Local modules assumed available:
  RedLionfishDeconv as rl
  ISS_preprocessing.psf as fd_psf
  ashlar.scripts.ashlar as ashlar
"""


# ============================
# --- Standard Library ---
# ============================
import os
import re
import math
import time
import shutil
import warnings
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Any, Iterable, Tuple

import xml.etree.ElementTree as ET

# ============================
# --- Third-Party ---
# ============================
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from natsort import natsorted


# ============================
# --- Local Modules ---
# ============================

import ISS_preprocessing.psf as fd_psf

from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity


# ======================================================================================
# GPU selection
# ======================================================================================


def choose_gpu_for_rl(
    preferred_max_mem_mb=2000,
    preferred_max_util=20,
):
    result = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,nounits,noheader"
    ]).decode("utf-8").strip().split("\n")

    rows = []
    for line in result:
        idx, mem, util = [x.strip() for x in line.split(",")]
        rows.append((int(idx), int(mem), int(util)))

    # Prefer GPUs that look truly free
    preferred = [
        r for r in rows
        if r[1] <= preferred_max_mem_mb and r[2] <= preferred_max_util
    ]

    if preferred:
        preferred.sort(key=lambda x: (x[1], x[2], x[0]))  # memory first
        gpu_id = preferred[0][0]
    else:
        # fallback: choose least memory used overall
        rows.sort(key=lambda x: (x[1], x[2], x[0]))
        gpu_id = rows[0][0]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYOPENCL_CTX"] = f"0:{gpu_id}"

    RED_BOLD = "\033[1;31m"
    RED = "\033[31m"
    RESET = "\033[0m"

    print(f"{RED_BOLD}Selected GPU {gpu_id}{RESET}")
    print(f"{RED}CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}{RESET}")
    print(f"{RED}PYOPENCL_CTX = {os.environ['PYOPENCL_CTX']}{RESET}")

    return gpu_id

# ======================================================================================
# Small, reusable utilities
# ======================================================================================
BOLD = "\033[1m"
RESET = "\033[0m"

def safe_mkdir(p: Path) -> Path:
    p.mkdir(exist_ok=True, parents=True)
    return p

def file_exists_and_valid(path: Path, min_size: int = 1024) -> bool:
    """Fast corruption guard for outputs from previous runs."""
    try:
        return path.exists() and path.stat().st_size > int(min_size)
    except Exception:
        return False


def to_uint16_safe(
    arr: np.ndarray,
    *,
    context: str = "",
) -> np.ndarray:
    """
    Safely cast an image array to uint16.

    This function:
      - Detects NaN / ±Inf values
      - Replaces them with safe numeric values
      - Clips intensities to the uint16 range [0, 65535]
      - Emits a single [INFO] message only if correction was needed

    Parameters
    ----------
    arr : np.ndarray
        Image array (2D or 3D), typically float after deconvolution.
    context : str
        Short identifier for logging (e.g. "tile=0 ch=1").

    Returns
    -------
    np.ndarray
        uint16 array with the same shape as input.
    """
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()

    if has_nan or has_inf:
        print(
            f"[INFO] to_uint16_safe"
            f"{f' ({context})' if context else ''}: "
            f"nan={has_nan}, inf={has_inf}"
        )

    safe = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=65535.0,
        neginf=0.0,
    )

    safe = np.clip(safe, 0, 65535)

    return safe.astype(np.uint16)

# ======================================================================================
# Deconwolf
# ======================================================================================

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


# ======================================================================================
# TileScanInfo writer (your “smart” position unit logic)
# ======================================================================================
def decide_and_write_tilescan(
    *,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    image_dimensions: Tuple[int, int],
    pixel_to_um_manual: Optional[float] = None,
    pixel_to_um_calc: Optional[float] = None,
    unit_hint_raw: str = "",
    off_tol: float = 0.35,
    tiles_iter: Optional[Iterable] = None,
    app_name: str = "LAS X",
    out_xml_path: Optional[Path] = None,
    deconvolution_method: Optional[str] = None,
    deconvolution_iterations: Optional[int] = None,
    objective_mag: Optional[float] = None,
    objective_mag_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decide stage position units + select pixel size (µm/px), then write TileScanInfo XML.

    What this does
    --------------
    1) Selects an effective pixel size (µm/px):
       - pixel_to_um_manual overrides pixel_to_um_calc
       - provenance is recorded in the output XML attributes
    2) Chooses a raw-position unit scale (raw → µm):
       - Uses unit_hint_raw if consistent
       - Otherwise runs a hypothesis test when tile_width_um is available
         (requires pixel size and image width to estimate tile width in µm)
    3) Writes TileScanInfo XML with positions ALWAYS expressed in microns (µm).

    Output XML provenance attributes
    -------------------------------
    - DeclaredUnitHint / DeclaredUnitNormalized
    - RawPositionUnitUsed / ScaleRawToMicron
    - PixelSizeUm / PixelSizeSource
    - TileWidthPx / TileWidthUm
    - MedianStepXUm / MedianStepYUm (when computable)

    Tile records contract (writer policy)
    -------------------------------------
    - If writing XML (out_xml_path and tiles_iter provided), tiles_iter MUST contain TileIndex.
    - Supported tile formats:
        * 5-tuple: (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
        * or dict with keys: TileIndex, FieldX, FieldY, PosX, PosY
    - Output tiles are written exactly as provided (no inference, no reordering).

    Returns
    -------
    Dict[str, Any]
        Decision summary (unit choice, scale, pixel size choice, inferred steps/overlap if available).
    """

    # Make sure these are arrays (defensive + consistent)
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    def _normalize_unit(u: str) -> str:
        if not u:
            return "unknown"
        u = u.strip().lower().replace("µ", "u")
        if u in {"um", "u", "micron", "microns"} or "micromet" in u:
            return "microns"
        if u in {"px", "pixel", "pixels"}:
            return "pixels"
        if u in {"m", "meter", "metre", "meters", "metres"}:
            return "meters"
        if u in {"mm", "millimeter", "millimetre", "millimeters", "millimetres"}:
            return "millimeters"
        return "unknown"

    def _robust_step_1d(vals_um, tile_width_um=None):
        if vals_um is None or len(vals_um) < 2:
            return None
        u = np.unique(np.round(vals_um, 9))
        if u.size < 2:
            return None
        d = np.diff(np.sort(u))
        d = d[d > 0]
        if d.size == 0:
            return None
        if tile_width_um and tile_width_um > 0:
            lo, hi = 0.5 * tile_width_um, 1.2 * tile_width_um
            band = d[(d >= lo) & (d <= hi)]
            if band.size == 0:
                k = max(1, int(0.3 * d.size))
                band = np.sort(d)[-k:]
            return float(np.median(band))
        k = max(1, int(0.3 * d.size))
        return float(np.median(np.sort(d)[-k:]))

    def _ov_pct(step_um, width_um):
        if step_um is None or not width_um:
            return None
        return (1 - step_um / width_um) * 100.0

    def _ov_qual(p):
        if p is None:
            return "n/a"
        if 5 <= p <= 15:
            return "typical"
        if p < 0:
            return "gap?"
        if p > 25:
            return "large"
        return "ok"

    def _fit_for_scale(scale_um_per_raw, tile_width_um):
        """
        Evaluate how well a given scale (raw units → µm) matches the expected tile width.
        Assumes tile_width_um is not None when used for scoring.
        """
        x_um = x_raw * scale_um_per_raw
        y_um = y_raw * scale_um_per_raw
        dx = _robust_step_1d(x_um, tile_width_um)
        dy = _robust_step_1d(y_um, tile_width_um)

        if tile_width_um:
            axis_scores = []
            if dx is not None:
                axis_scores.append(abs(dx - tile_width_um) / max(tile_width_um, 1e-9))
            if dy is not None:
                axis_scores.append(abs(dy - tile_width_um) / max(tile_width_um, 1e-9))
            score = min(axis_scores) if axis_scores else float("inf")
        else:
            score = float("inf")

        return dict(
            score=score,
            dx=dx,
            dy=dy,
            ovx=_ov_pct(dx, tile_width_um),
            ovy=_ov_pct(dy, tile_width_um),
        )

    def _write_xml(
        *,
        to_um,
        chosen_unit,
        rationale,
        unit_hint_raw,
        unit_hint_norm,
        pixel_to_um,
        pixel_to_um_source,
        width_px,
        tile_width_um,
        dx,
        dy,
        tiles_iter,
        out_xml_path,
        app_name="LAS X",
        deconvolution_method=None,
        deconvolution_iterations=None,
        objective_mag=None,
        objective_mag_source=None,
    ):
    
        # ------------------------------------------------------------------
        # Normalize tiles into records (STRICT: require TileIndex)
        #
        # - tiles_iter MUST be an iterable of STRICT 5-tuples:
        #     (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
        # ------------------------------------------------------------------
        tiles_list = list(tiles_iter or [])
        if not tiles_list:
            raise ValueError("tiles_iter is empty — cannot write TileScanInfo without TileIndex records.")
    
        recs = []
        for enum_i, t in enumerate(tiles_list):
            rec = dict(enum_i=int(enum_i), tile_index=None, fx=None, fy=None, px=None, py=None)
    
            # STRICT 5-tuple ONLY
            tt = tuple(t)
            if len(tt) != 5:
                raise ValueError(
                    "Tile entry must be a STRICT 5-tuple (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw); "
                    f"got len={len(tt)}: {tt}"
                )
    
            rec["tile_index"] = int(tt[0])
            rec["fx"] = int(tt[1])
            rec["fy"] = int(tt[2])
            rec["px"] = float(tt[3])
            rec["py"] = float(tt[4])
    
            # final sanity
            if rec["tile_index"] is None:
                raise ValueError(f"Tile entry missing TileIndex after parsing: {t}")
            if rec["px"] is None or rec["py"] is None:
                raise ValueError(f"Tile entry missing position: {t}")
    
            recs.append(rec)
    

        # ------------------------------------------------------------------
        # Build XML
        # ------------------------------------------------------------------
        out = ET.Element("Data")
        img = ET.SubElement(out, "Image", TextDescription="")
        
        app = str(app_name).strip().lower()
        
        att = ET.SubElement(
            img,
            "Attachment",
            Name="TileScanInfo",
            Application=str(app_name),
        
            # keep your existing NIS special-case
            FlipX="1" if app == "nis-elements" else "0",
        
            FlipY="0" ,
        
            SwapXY="0",
        )

    
        # Global metadata
        att.set("Unit", "micron")
        att.set("DeclaredUnitHint", unit_hint_raw or "unknown")
        att.set("DeclaredUnitNormalized", unit_hint_norm or "unknown")
        att.set("RawPositionUnitUsed", chosen_unit or "unknown")
        att.set("ScaleRawToMicron", f"{to_um:.12g}")
        att.set("DecisionNote", rationale or "")
    
        if deconvolution_method:
            att.set("DeconvolutionMethod", str(deconvolution_method))
            att.set("DeconvolutionIterations", str(deconvolution_iterations or 0))
        else:
            att.set("DeconvolutionMethod", "None")
            att.set("DeconvolutionIterations", "0")
    
        if pixel_to_um is not None:
            att.set("PixelSizeUm", f"{float(pixel_to_um):.10f}")
            att.set("PixelSizeSource", pixel_to_um_source or "unknown")
    
        if objective_mag is not None:
            att.set("ObjectiveMagnification", f"{float(objective_mag):.10g}")
            att.set("ObjectiveMagnificationSource", objective_mag_source or "metadata-derived")
    
        if width_px is not None:
            att.set("TileWidthPx", str(int(width_px)))
        if tile_width_um is not None:
            att.set("TileWidthUm", f"{float(tile_width_um):.10f}")
        if dx is not None:
            att.set("MedianStepXUm", f"{float(dx):.10f}")
        if dy is not None:
            att.set("MedianStepYUm", f"{float(dy):.10f}")
    
        # Tiles (exactly as given)
        for r in recs:
            ET.SubElement(
                att,
                "Tile",
                TileIndex=str(int(r["tile_index"])),
                FieldX=str(int(r["fx"])),
                FieldY=str(int(r["fy"])),
                PosX=f"{r['px'] * to_um:.10f}",
                PosY=f"{r['py'] * to_um:.10f}",
            )
    
        ET.ElementTree(out).write(out_xml_path, encoding="utf-8", xml_declaration=True)
        print(f"[INFO] Wrote TileScanInfo: {out_xml_path} (positions in µm)")

    # --- pixel size selection ---
    width_px = image_dimensions[0] if isinstance(image_dimensions, (tuple, list)) else None
    
    pixel_to_um = None
    pixel_to_um_source = "unavailable"
    if pixel_to_um_manual is not None:
        pixel_to_um = float(pixel_to_um_manual)
        pixel_to_um_source = "manual argument"
    elif pixel_to_um_calc is not None:
        pixel_to_um = float(pixel_to_um_calc)
        pixel_to_um_source = "metadata-derived"
    
    # ------------------------------------------------------------------
    # RESTORED PRINTS: pixel size provenance + magnification
    # ------------------------------------------------------------------
    if objective_mag is not None:
        src = objective_mag_source or "metadata-derived"
        print(f"[META] Objective magnification: {float(objective_mag):g}x (source={src})")
    else:
        print("[META] Objective magnification: unavailable")
    

    if pixel_to_um_calc is not None:
        print(f"[META] Pixel size from metadata: {float(pixel_to_um_calc):.6f} µm/px")
    else:
        print("{BOLD}[WARN]⚠️ {RESET} No pixel size information available from metadata")
    
    if pixel_to_um_manual is not None:
        print(f"[INFO] Manual pixel_to_um: {float(pixel_to_um_manual):.6f} µm/px")
        if pixel_to_um_calc is not None and not np.isclose(
            float(pixel_to_um_manual), float(pixel_to_um_calc), rtol=0.02
        ):
            print(
                f"{BOLD}[WARN]⚠️ {RESET} Manual pixel size ({float(pixel_to_um_manual):.6f} µm/px) differs "
                f"from metadata value ({float(pixel_to_um_calc):.6f} µm/px)."
            )
    
    
    # ------------------------------------------------------------------
    # RESTORED PRINT #1: explicit pixel size decision 
    # ------------------------------------------------------------------
    if pixel_to_um is not None:
        if pixel_to_um_manual is not None:
            msg = f"[INFO] Pixel size decision: using manual pixel_to_um={pixel_to_um:.6f} µm/px"
            if pixel_to_um_calc is not None:
                msg += f" (metadata={float(pixel_to_um_calc):.6f} µm/px)"
            print(msg)
        else:
            print(f"[INFO] Pixel size decision: using metadata-derived pixel_to_um={pixel_to_um:.6f} µm/px")
    else:
        print("{BOLD}[WARN]⚠️ {RESET} Pixel size decision: no pixel size available (manual=None, metadata=None)")
    
    tile_width_um = (width_px * pixel_to_um) if (width_px and pixel_to_um) else None
    unit_hint_norm = _normalize_unit(unit_hint_raw or "")
    
    # Hypotheses: scale raw units → µm
    candidates = {
        "meters": 1e6,
        "millimeters": 1e3,
        "microns": 1.0,
        "pixels": pixel_to_um if pixel_to_um is not None else None,
    }
    
    chosen_unit = None
    rationale = ""
    dx = dy = ovx = ovy = None
    to_um = None
    
    # ---------------------------
    # Unit decision
    # ---------------------------
    if tile_width_um is None:
    # No tile width => hypothesis test is not meaningful.
    # Prefer metadata hint if it maps to a valid scale, else default to microns.
        if unit_hint_norm in candidates and candidates[unit_hint_norm] is not None:
            chosen_unit = unit_hint_norm
            to_um = candidates[chosen_unit]
    
            # Make it explicit that this is a hint-based choice (not “confirmed”)
            # and record the original hint value for debugging.
            rationale = (
                f"{chosen_unit} chosen from declared unit hint "
                f"(DeclaredUnitNormalized='{unit_hint_norm}'; no tile_width_um)"
            )
        else:
            chosen_unit = "microns"
            to_um = candidates.get("microns", 1.0) or 1.0
    
            # Make it explicit we are defaulting because hint was unusable.
            rationale = (
                "microns chosen by default (declared unit hint unusable or missing; "
                "no tile_width_um; hypothesis test disabled)"
            )

    else:
        # Try hint first (if it corresponds to a candidate with a scale)
        if unit_hint_norm in candidates and candidates[unit_hint_norm] is not None:
            r = _fit_for_scale(candidates[unit_hint_norm], tile_width_um)
            if r["score"] <= off_tol:
                chosen_unit = unit_hint_norm
                to_um = candidates[chosen_unit]
                dx, dy, ovx, ovy = r["dx"], r["dy"], r["ovx"], r["ovy"]
                rationale = f"metadata unit '{chosen_unit}' confirmed"
            else:
                print(f"{BOLD}[WARN]⚠️ {RESET} Metadata unit '{unit_hint_norm}' inconsistent — running hypothesis test.")
    
        # Otherwise (or if hint failed): choose best hypothesis
        if chosen_unit is None:
            scores, details = {}, {}
            for name, scale in candidates.items():
                if scale is None:
                    continue
                r = _fit_for_scale(scale, tile_width_um)
                scores[name] = r["score"]
                details[name] = r
            if scores:
                chosen_unit = min(scores, key=scores.get)
                to_um = candidates.get(chosen_unit, 1.0) or 1.0
                d = details.get(chosen_unit, {})
                dx, dy, ovx, ovy = d.get("dx"), d.get("dy"), d.get("ovx"), d.get("ovy")
                rationale = f"{chosen_unit} chosen by hypothesis"
            else:
                chosen_unit = "microns"
                to_um = 1.0
                rationale = "no valid unit candidates; defaulting to microns"
    
    px_str = f"{pixel_to_um:.6f}" if pixel_to_um is not None else "NA"
    tw_str = f"{tile_width_um:.2f}" if tile_width_um is not None else "NA"
    
    # ------------------------------------------------------------------
    # RESTORED PRINT #2: compact unit decision line 
    # ------------------------------------------------------------------

    if (dx is not None) and (dy is not None) and (tile_width_um is not None) and (tile_width_um > 0):
        ovx_s = f"{ovx:.1f}% {_ov_qual(ovx)}" if ovx is not None else "n/a"
        ovy_s = f"{ovy:.1f}% {_ov_qual(ovy)}" if ovy is not None else "n/a"
        print(
            f"[INFO] Position unit decision (detail): {rationale}: "
            f"ΔX≈{dx:.2f} µm (overlap≈{ovx_s}), "
            f"ΔY≈{dy:.2f} µm (overlap≈{ovy_s}) "
            f"vs width≈{tile_width_um:.2f} µm "
            f"[unit used='{chosen_unit}'; width_px={width_px}; pixel_to_um={px_str} µm/px; tile_width_um={tw_str} µm]"
        )
    else:
        print(
            f"[INFO] Position unit decision: {rationale} "
            f"[unit used='{chosen_unit}'; width_px={width_px}; pixel_to_um={px_str} µm/px; tile_width_um={tw_str} µm]"
        )

    # NOTE (TileScanInfo writer policy)
    # --------------------------------
    # _write_xml() writes exactly what tiles_iter provides (no inference, no sorting).
    # Handlers must provide valid ints for TileIndex/FieldX/FieldY.

    if out_xml_path and tiles_iter is not None:
        _write_xml(
            to_um=to_um,
            chosen_unit=chosen_unit,
            rationale=rationale,
            unit_hint_raw=unit_hint_raw,
            unit_hint_norm=unit_hint_norm,
            pixel_to_um=pixel_to_um,
            pixel_to_um_source=pixel_to_um_source,
            width_px=width_px,
            tile_width_um=tile_width_um,
            dx=dx,
            dy=dy,
            tiles_iter=tiles_iter,
            out_xml_path=out_xml_path,
            app_name=app_name,
            deconvolution_method=deconvolution_method,
            deconvolution_iterations=deconvolution_iterations,
            objective_mag=objective_mag,
            objective_mag_source=objective_mag_source,
        )

    return dict(
        chosen_unit=chosen_unit,
        to_um=to_um,
        rationale=rationale,
        dx=dx,
        dy=dy,
        ovx=ovx,
        ovy=ovy,
        pixel_to_um=pixel_to_um,
        pixel_to_um_source=pixel_to_um_source,
        tile_width_um=tile_width_um,
        width_px=width_px,
        unit_hint_normalized=unit_hint_norm,
    )

# ==================================================================================
# Shared HELPERS 
# ==================================================================================

def normalize_pixel_size_to_um(
    raw_length_per_pixel: float,
    *,
    source: str,
    meters_range: Tuple[float, float] = (1e-9, 1e-4),
) -> Tuple[Optional[float], str]:
    """
    Normalize a raw physical length-per-pixel value to microns (µm/px)
    using magnitude-based heuristics.

    This function centralizes the unit-inference logic used across
    TIFF / LIF / CZI handlers to avoid format-dependent drift.

    Heuristic
    ---------
    - If raw_length_per_pixel falls within a plausible meters-per-pixel
      range (default: 1e-9 .. 1e-4), interpret it as meters and convert
      to microns (× 1e6).
    - Otherwise, assume the value is already expressed in microns.

    IMPORTANT DESIGN NOTES
    ----------------------
    - This function performs *no metadata parsing* — only magnitude-based
      normalization.
    - The meters_range is intentionally conservative and shared across
      all formats for consistency.
    - If raw_length_per_pixel is non-positive or NaN, returns (None, reason).

    Parameters
    ----------
    raw_length_per_pixel : float
        Raw length-per-pixel value extracted from metadata.
    source : str
        Human-readable source label (used only for provenance strings).
    meters_range : (float, float)
        Inclusive range treated as meters-per-pixel.

    Returns
    -------
    pixel_size_um : float or None
        Pixel size in microns per pixel, or None if invalid.
    provenance : str
        Description of how the value was interpreted.
    """

    try:
        v = float(raw_length_per_pixel)
    except Exception:
        return None, f"{source}: invalid raw value"

    if not np.isfinite(v) or v <= 0:
        return None, f"{source}: non-positive or non-finite value"

    lo, hi = meters_range

    if lo <= v <= hi:
        # Interpreted as meters-per-pixel → convert to µm
        return v * 1e6, f"{source}: meters-per-pixel (×1e6 → µm)"
    else:
        # Interpreted as already in µm-per-pixel
        return v, f"{source}: assumed µm-per-pixel (outside meter range)"


# ==================================================================================
# TIFF HELPERS (Leica XML / TIFF modes)
# ==================================================================================

def tiff_find_metadata_dir_case_insensitive(input_dir: Path, folder_name: str = "metadata") -> Optional[Path]:
    """
    Find Leica's metadata folder inside input_dir, case-insensitively.

    Leica commonly uses folder names like:
      - "Metadata"
      - "MetaData"
      - "metadata"

    We scan only one level under input_dir.
    """
    input_dir = Path(input_dir)
    wanted = (folder_name or "").strip().lower()

    for p in input_dir.iterdir():
        if p.is_dir() and p.name.strip().lower() == wanted:
            return p
    return None

def tiff_pick_leica_xml(input_metadata_dir: Path, *, region_token: str = "") -> Optional[Path]:
    """
    Pick the best Leica XML/XLF file from a Metadata folder.

    Rules:
      - Accept: .xml, .xlif
      - Ignore: any file with "properties" in the name (LAS X dumps)
      - Prefer: a file whose name contains region_token (case-insensitive)
      - Fallback: newest file by modification time
    """
    input_metadata_dir = Path(input_metadata_dir)

    md_files = [
        f for f in input_metadata_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".xml", ".xlf", ".xlif")
        and "properties" not in f.name.lower()
    ]
    if not md_files:
        return None

    region_token = (region_token or "").strip()
    if region_token:
        prio = [f for f in md_files if region_token.lower() in f.stem.lower()]
        if prio:
            # If multiple match, keep deterministic ordering
            return sorted(prio)[0]

    return max(md_files, key=lambda p: p.stat().st_mtime)

def tiff_parse_xml_safe(md_file: Path) -> Optional[ET.Element]:
    """Parse Leica XML safely. Return root element or None on failure."""
    try:
        root = ET.parse(str(md_file)).getroot()
        return root if root is not None else None
    except Exception:
        return None

def tiff_safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def tiff_px_um_from_dim(dim_node: Optional[ET.Element], axis: str) -> Tuple[Optional[float], str]:
    """
    Compute Leica pixel size from a DimensionDescription node using Length / NumberOfElements.

    Leica exports are inconsistent about physical units. In practice:
      - If Length/Elements is extremely small (typical meters/px scale), treat it as meters/px
        and convert to µm/px by multiplying by 1e6.
      - Otherwise assume the value is already in µm/px.

    Returns
    -------
    (pixel_size_um_per_px, source_string)
        pixel_size_um_per_px may be None if required fields are missing.

    DESIGN NOTES
    ------------
    - Pure extraction helper: NO printing, NO logging, NO file I/O.
    - All unit-magnitude inference is delegated to normalize_pixel_size_to_um()
      to keep TIFF/LIF/CZI behavior consistent.
    """

    if dim_node is None:
        return None, f"{axis}:missing"

    # Leica XML uses either NumberOfElements or Elements depending on export flavor.
    N = tiff_safe_float(dim_node.attrib.get("NumberOfElements") or dim_node.attrib.get("Elements"))
    L = tiff_safe_float(dim_node.attrib.get("Length"))

    # Guard: require both values and require positivity
    if N is None or L is None:
        return None, f"{axis}:no_length_or_count"
    if N <= 0 or L <= 0:
        return None, f"{axis}:nonpositive_length_or_count"

    raw = L / N  # raw pixel size in unknown units (meters/px or µm/px)

    # Guard: reject non-finite values early
    if not np.isfinite(raw) or raw <= 0:
        return None, f"{axis}:invalid_raw"

    # ------------------------------------------------------------------
    # Delegate unit normalization to the shared helper
    # ------------------------------------------------------------------
    px_um, src = normalize_pixel_size_to_um(
        raw,
        source=f"{axis}:Length/N",
    )

    return px_um, src


def tiff_extract_pixel_size_and_magnification(
    root: ET.Element,
    *,
    pixel_to_um_manual: Optional[float] = None,
    rtol_warn: float = 0.02,
) -> Dict[str, Any]:
    """
    Extract pixel size (µm/px), objective magnification, and unit hints from Leica TIFF XML.

    IMPORTANT
    ---------
    - Extraction only: this function MUST NOT print, log, or write files.
    - It MUST NOT apply the manual pixel size override: pixel_to_um_manual is handled later
      in decide_and_write_tilescan() to keep all decisions and warnings centralized.

    Returns
    -------
    Dict[str, Any]
        Includes:
          - pixel_to_um_calc (float or None)
          - magnification (float or None)
          - unit_hint_raw (str)
          - px_um_x/px_um_y + per-axis source strings + mismatch indicator
    """

    def tiff_extract_objective_magnification(root: ET.Element) -> Optional[float]:
        """
        Best-effort objective magnification extraction from Leica XML.
    
        We try a few plausible locations/attributes. Returns magnification (e.g. 20.0) or None.
        """
        for xp in (
            ".//Instrument//Objective",
            ".//Attachment[@Name='HardwareSetting']//ATLCameraSettingDefinition",
        ):
            n = root.find(xp)
            if n is None:
                continue
            mag = tiff_safe_float(
                n.attrib.get("Magnification")
                or n.attrib.get("NominalMagnification")
                or n.attrib.get("TotalVideoMag")
            )
            if mag:
                return mag
        return None
    
    # ------------------------------------------------------------
    # Pixel size extraction from DimensionDescription nodes
    # ------------------------------------------------------------
    dim_x = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='1']")
    dim_y = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='2']")

    px_um_x, src_x = tiff_px_um_from_dim(dim_x, "X")
    px_um_y, src_y = tiff_px_um_from_dim(dim_y, "Y")

    pixel_to_um_calc = None
    warn_xy_mismatch = None

    if px_um_x is not None and px_um_y is not None:
        rel = abs(px_um_x - px_um_y) / max(px_um_x, px_um_y)
        pixel_to_um_calc = (px_um_x + px_um_y) / 2.0
        if rel > rtol_warn:
            warn_xy_mismatch = rel
    else:
        pixel_to_um_calc = px_um_x if px_um_x is not None else px_um_y

    # ------------------------------------------------------------
    # Objective magnification (best-effort)
    # ------------------------------------------------------------
    mag = tiff_extract_objective_magnification(root)

    # ------------------------------------------------------------
    # Raw unit hint from metadata (may be unreliable)
    # ------------------------------------------------------------
    unit_hint_raw = (
        dim_x.attrib.get("Unit", "") if dim_x is not None else ""
    ).strip().lower()

    # ------------------------------------------------------------
    # Return EVERYTHING needed for later decision + logging
    # ------------------------------------------------------------
    return dict(
        pixel_to_um_calc=pixel_to_um_calc,
        magnification=mag,
        unit_hint_raw=unit_hint_raw,

        # Provenance for later logging
        px_um_x=px_um_x,
        px_um_y=px_um_y,
        src_x=src_x,
        src_y=src_y,
        warn_xy_mismatch=warn_xy_mismatch,
    )


def tiff_collect_tiles_from_tilescaninfo(root: ET.Element) -> List[Tuple]:
    """
    Read Leica TileScanInfo tile positions from XML.
    
    Returns
    -------
    List[Tuple]
        If TileIndex is present on tiles:
            (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
    
        Else:
            (FieldX, FieldY, PosX_raw, PosY_raw)
    
    Ordering
    --------
    If TileIndex is present:
        Tiles are sorted by TileIndex (deterministic identity order).
    
    If TileIndex is NOT present:
        XML document order is preserved exactly.
    
        This is intentional: Leica LAS AF / LAS X often writes tiles in
        acquisition order (commonly serpentine). Sorting by (FieldY, FieldX)
        would silently convert serpentine into raster order and break mapping
        to on-disk filename tile ids (e.g., s0000, s0001, ...).
    
    Validation
    ----------
    - If ANY <Tile> has TileIndex then ALL must have TileIndex (else ValueError).
    - FieldX, FieldY, PosX, PosY are required on every <Tile>.
    - Types are normalized:
        FieldX / FieldY / TileIndex -> int
        PosX / PosY -> float
    """

    def _require_attr(node: ET.Element, name: str) -> str:
        v = node.attrib.get(name, None)
        if v is None:
            raise ValueError(
                f"<Tile> missing required attribute {name!r}: "
                f"{ET.tostring(node, encoding='unicode')}"
            )
        return v

    tile_nodes = root.findall(".//Attachment[@Name='TileScanInfo']//Tile")
    if not tile_nodes:
        return []

    # Check TileIndex consistency (either present on all tiles or none)
    has_any_ti = any("TileIndex" in n.attrib for n in tile_nodes)
    has_all_ti = all("TileIndex" in n.attrib for n in tile_nodes)
    if has_any_ti and not has_all_ti:
        raise ValueError(
            "Inconsistent TileScanInfo: some <Tile> have TileIndex and others do not."
        )

    # Decide ordering policy based on presence of TileIndex.
    #
    # - If TileIndex exists: it's the only safe identity key. We sort by TileIndex so the
    #   returned list is deterministic and aligns with filename tile ids when those are
    #   acquisition-indexed (common in Leica exports).
    #
    # - If TileIndex does NOT exist: Leica often writes <Tile> elements in acquisition order
    #   (commonly serpentine). Sorting by (FieldY, FieldX) would silently convert serpentine
    #   into raster order and break the mapping to on-disk tile ids (s0000, s0001, ...).
    #   Therefore we preserve the XML document order exactly.
    if has_all_ti:
        tile_nodes_sorted = sorted(tile_nodes, key=lambda n: int(_require_attr(n, "TileIndex")))
    else:
        tile_nodes_sorted = list(tile_nodes)  # preserve XML order (acquisition/path order)


    tiles_iter: List[Tuple] = []

    for n in tile_nodes_sorted:
        # Required core attributes
        fx = int(_require_attr(n, "FieldX"))
        fy = int(_require_attr(n, "FieldY"))
        px = float(_require_attr(n, "PosX"))
        py = float(_require_attr(n, "PosY"))

        if has_all_ti:
            ti = int(_require_attr(n, "TileIndex"))
            tiles_iter.append((ti, fx, fy, px, py))
        else:
            tiles_iter.append((fx, fy, px, py))

    return tiles_iter

# ======================================================================================
# LIF helpers (pixel size + objective magnification) 
# ======================================================================================

def lif_get_mag_and_pixel_to_um(ctx: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract LIF objective magnification + pixel size (µm/px), using the keys your LifHandler sets.

    Expects (from LifHandler.open_region):
      - ctx["lif_file"] (readlif LifFile)   [optional but preferred]
      - ctx["lif_xml_root"]                [optional]
      - ctx["lif_image_dict"]              [optional]
      - ctx["lif_filepath"]                [optional, only for messaging]

    Returns:
      (objective_mag, pixel_to_um_calc)
    """
    lf = ctx.get("lif_file", None)

    # ---------- xml_header as text (best-effort) ----------
    xml_text = ""
    try:
        xml_header = getattr(lf, "xml_header", None) if lf is not None else None
        if isinstance(xml_header, (bytes, bytearray)):
            xml_text = xml_header.decode("utf-8", errors="replace")
        else:
            xml_text = str(xml_header or "")
    except Exception:
        xml_text = ""

    # Prefer ctx-provided root, fallback to lf.xml_root
    root = ctx.get("lif_xml_root", None)
    if root is None and lf is not None:
        root = getattr(lf, "xml_root", None)

    def _try_float(x) -> Optional[float]:
        try:
            if x is None:
                return None
            if isinstance(x, str):
                x = x.strip().replace(",", ".")
            return float(x)
        except Exception:
            return None

    def _plausible_mag(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        try:
            if not np.isfinite(v):
                return None
        except Exception:
            return None
        # typical objectives ~1..200; allow a bit wider
        return float(v) if (0.25 <= float(v) <= 400.0) else None

    def _local(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    # =============================================================================
    # (1) Magnification from xml_root (namespace-agnostic)
    # =============================================================================
    mag: Optional[float] = None
    if isinstance(root, ET.Element):
        objective_like = []
        for n in root.iter():
            lname = _local(n.tag).lower()
            if "objective" in lname:
                objective_like.append(n)

        scan_nodes = objective_like if objective_like else list(root.iter())

        attr_keys = (
            "Magnification",
            "NominalMagnification",
            "ObjectiveMagnification",
            "TotalMagnification",
            "TotalVideoMagnification",
        )

        for n in scan_nodes:
            # 1) attributes on node
            for k in attr_keys:
                if k in n.attrib:
                    mag = _plausible_mag(_try_float(n.attrib.get(k)))
                    if mag is not None:
                        break
            if mag is not None:
                break

            # 2) child text nodes
            for child in list(n):
                lk = _local(child.tag)
                if lk in attr_keys and (child.text or "").strip():
                    mag = _plausible_mag(_try_float(child.text))
                    if mag is not None:
                        break
            if mag is not None:
                break

    # =============================================================================
    # (2) Magnification regex fallback over xml_header text
    # =============================================================================
    if mag is None and xml_text:
        patterns = [
            r'NominalMagnification\s*=\s*["\']([\d.,]+)["\']',
            r'ObjectiveMagnification\s*=\s*["\']([\d.,]+)["\']',
            r'TotalMagnification\s*=\s*["\']([\d.,]+)["\']',
            r'Magnification\s*=\s*["\']([\d.,]+)["\']',
            r'<NominalMagnification>\s*([\d.,]+)\s*</NominalMagnification>',
            r'<Magnification>\s*([\d.,]+)\s*</Magnification>',
        ]
        for pat in patterns:
            m = re.search(pat, xml_text, re.IGNORECASE)
            if m:
                mag = _plausible_mag(_try_float(m.group(1)))
                if mag is not None:
                    break

    # =============================================================================
    # Pixel size (µm/px)
    # =============================================================================
    pixel_to_um_calc: Optional[float] = None

    # (a) Look for voxel size in xml_header (often meters)
    if xml_text:
        voxels = re.findall(r'VoxelSize[XY]\s*=\s*["\']([\deE.+-]+)["\']', xml_text, re.IGNORECASE)
        if voxels:
            try:
                # IMPORTANT:
                # - LIF xml_header voxel sizes are often meters/px, but not guaranteed.
                # - We delegate unit normalization to normalize_pixel_size_to_um()
                #   to keep behavior consistent with TIFF/CZI.
                vals_um: List[float] = []
                for i, v in enumerate(voxels[:2]):  # X, Y only
                    if not v:
                        continue
                    raw = _try_float(v)
                    if raw is None or not np.isfinite(raw) or raw <= 0:
                        continue
                    um, _src = normalize_pixel_size_to_um(
                        float(raw),
                        source=f"LIF xml_header:VoxelSize{'XY'[i]}",
                    )
                    if um is not None:
                        vals_um.append(float(um))

                if vals_um:
                    pixel_to_um_calc = float(sum(vals_um) / len(vals_um))
                    return mag, pixel_to_um_calc
            except Exception:
                pass

    # (b) Common readlif image_dict keys (varies by version/data)
    d = ctx.get("lif_image_dict", None)
    if isinstance(d, dict):
        # Try a handful of common patterns; accept either meters or microns
        candidate_keys = ("voxel_size", "voxelSize", "pixel_size", "pixelsize", "scale", "scales")
        for key in candidate_keys:
            v = d.get(key, None)
            if v is None:
                continue
            try:
                if isinstance(v, dict):
                    x = v.get("x") or v.get("X") or v.get("0")
                else:
                    x = v[0]  # list/tuple/np array

                raw = _try_float(x)
                if raw is None or not np.isfinite(raw) or raw <= 0:
                    continue

                # IMPORTANT:
                # - readlif sometimes stores meters/px, sometimes µm/px depending on version/data.
                # - Delegate to shared helper to avoid drifting heuristics.
                pixel_to_um_calc, _src = normalize_pixel_size_to_um(
                    float(raw),
                    source=f"LIF image_dict:{key}",
                )
                if pixel_to_um_calc is not None and pixel_to_um_calc > 0:
                    return mag, float(pixel_to_um_calc)
            except Exception:
                continue

    # (c) Fallback: xml_root DimensionDescription (Length / Elements) -> meters or microns
    def _safe_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    def _dim_to_um(el: Optional[ET.Element], axis: str) -> Optional[float]:
        if el is None:
            return None
        N = _safe_float(el.attrib.get("NumberOfElements") or el.attrib.get("Elements"))
        L = _safe_float(el.attrib.get("Length"))
        if N is None or L is None or N <= 0 or L <= 0:
            return None

        raw = L / N  # could be meters/px or already microns/px depending on file

        # IMPORTANT:
        # - Delegate normalization to shared helper (same logic as TIFF).
        um, _src = normalize_pixel_size_to_um(
            float(raw),
            source=f"LIF xml_root:{axis}:Length/N",
        )
        return um

    if isinstance(root, ET.Element):
        try:
            # Leica LIF often uses DimID 1=X and 2=Y, but we keep it defensive
            dim_x = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='1']")
            dim_y = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='2']")
            vals = [v for v in (_dim_to_um(dim_x, "X"), _dim_to_um(dim_y, "Y")) if v is not None]
            if vals:
                pixel_to_um_calc = float(sum(vals) / len(vals))
        except Exception:
            pixel_to_um_calc = None

    return mag, (float(pixel_to_um_calc) if pixel_to_um_calc is not None else None)

# ======================================================================================
# CZI helpers (pixel size + objective magnification + mosaic tile positions + dims normalization)
# ======================================================================================

def czi_get_mag_and_pixel_to_um(czi: "CziFile") -> Tuple[Optional[float], Optional[float]]:
    """
    Extract objective magnification + pixel size (µm/px) from CZI metadata.

    Policy
    ------
    - We DO NOT guess *stage* units here (that belongs to decide_and_write_tilescan).
    - We DO try hard to recover pixel size from scaling metadata:
        * If unit is explicit: convert deterministically.
        * If unit is missing/unknown: accept only if the numeric magnitude is
          strongly indicative (meters-like vs microns-like vs nm-like).
          Otherwise return None (fail closed).

    Returns
    -------
    (mag, pixel_to_um_calc)
      - mag: objective magnification (e.g., 20, 40)
      - pixel_to_um_calc: µm/px (float) or None
    """
    meta_root = getattr(czi, "meta", None)
    if meta_root is None:
        return None, None

    if not isinstance(meta_root, ET.Element):
        try:
            meta_root = ET.fromstring(meta_root)  # bytes/str
        except Exception:
            return None, None

    def _local(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _iter(root: ET.Element, name: str):
        for n in root.iter():
            if _local(n.tag) == name:
                yield n

    # ----------------------------
    # Pixel size (Scaling/Items/Distance)
    # ----------------------------
    def _parse_float(txt: Optional[str]) -> Optional[float]:
        if not txt:
            return None
        try:
            return float(txt.strip().replace(",", "."))
        except Exception:
            return None

    def _unit_to_um(v: float, unit_txt: Optional[str]) -> Optional[float]:
        """
        Deterministic conversion if unit is known.
        Returns None if unit missing/unknown.
        """
        u = (unit_txt or "").strip().lower().replace("μ", "µ")
        if u in ("m", "meter", "metre", "meters", "metres"):
            return v * 1e6
        if u in ("mm", "millimeter", "millimetre", "millimeters", "millimetres"):
            return v * 1e3
        if u in ("µm", "um", "micrometer", "micrometre", "micron", "microns"):
            return v
        if u in ("nm", "nanometer", "nanometre", "nanometers", "nanometres"):
            return v * 1e-3
        return None

    def _magnitude_to_um(v: float) -> Optional[float]:
        """
        Unit-missing fallback:
        Accept only if value magnitude is very clearly in one of these regimes:
    
          - meters/px:   ~1e-9 .. 1e-3  (typical is ~1e-7 .. 1e-5)
          - microns/px:  ~1e-3 .. 50     (typical is ~0.05 .. 5)
          - nanometers/px: extremely small in microns; rarely used here
    
        We return µm/px or None if ambiguous.
    
        IMPORTANT DESIGN NOTE
        ---------------------
        All magnitude-based unit inference is delegated to the shared helper
        normalize_pixel_size_to_um() so TIFF / LIF / CZI behave identically.
        This wrapper exists only to preserve the CZI-specific calling contract.
        """
        if not np.isfinite(v) or v <= 0:
            return None
    
        # Delegate magnitude-based unit inference to shared helper
        px_um, _src = normalize_pixel_size_to_um(
            float(v),
            source="CZI scaling",
        )
    
        return px_um


    def _get_pixel_to_um(root: ET.Element) -> Optional[float]:
        vals_um: List[float] = []

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
                    unit_txt = ch.text.strip()

            v = _parse_float(val_txt)
            if v is None:
                continue

            um = _unit_to_um(v, unit_txt)
            if um is None:
                # Unit missing/unknown: use magnitude-based acceptance (fail closed)
                um = _magnitude_to_um(v)

            if um is not None and np.isfinite(um) and um > 0:
                vals_um.append(float(um))

        if not vals_um:
            return None

        # If X and Y differ wildly, refuse (metadata likely inconsistent)
        if len(vals_um) >= 2:
            a, b = vals_um[0], vals_um[1]
            if max(a, b) / max(min(a, b), 1e-12) > 1.5:
                return None

        return float(sum(vals_um) / len(vals_um))

    pixel_to_um = _get_pixel_to_um(meta_root)

    # ----------------------------
    # Magnification (best-effort, as before)
    # ----------------------------
    mag = None
    for path in (
        ".//Information/Instrument/Objectives/Objective/NominalMagnification",
        ".//Information/Instrument/Objectives/Objective/ManufacturerData/Magnification",
        ".//Scaling/Objectives/Objective/NominalMagnification",
    ):
        txt = meta_root.findtext(path)
        if not txt:
            continue
        m = re.search(r"([\d.,]+)", txt)
        if not m:
            continue
        try:
            v = float(m.group(1).replace(",", "."))
        except Exception:
            continue
        if np.isfinite(v) and (0.25 <= v <= 400):
            mag = float(v)
            break

    return mag, pixel_to_um
    
def czi_get_mosaic_positions(
    czi: "CziFile",
    *,
    n_tiles: int,
    scene_index: Optional[int] = None,
    block_index: Optional[int] = None,
) -> Tuple[
    List[Tuple[int, int, int, float, float]],
    np.ndarray,
    np.ndarray,
    str,
]:
    """
    Extract raw mosaic tile positions from a Zeiss CZI file using aicspylibczi.

    This function queries the mosaic tile bounding boxes directly from the CZI
    container and converts them into STRICT 5-tuples suitable for downstream
    TileScanInfo writing:

        (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)

    IMPORTANT DESIGN NOTES
    ----------------------
    - This function performs *no unit guessing* and *no scaling*.
      Raw coordinates are returned exactly as reported by the CZI.
    - The returned unit_hint_raw is intentionally conservative ("unknown").
      Unit normalization happens later in decide_and_write_tilescan().
    - TileIndex is taken directly from the mosaic index (M dimension).
    - FieldX / FieldY are *placeholders* here (Zeiss does not expose a grid);
      spatial layout is determined solely from (PosX_raw, PosY_raw).

    SCENE / BLOCK HANDLING
    ----------------------
    - If `scene_index` is provided, bounding boxes are queried with S=scene_index.
      This is REQUIRED for multi-scene CZIs; otherwise all scenes incorrectly
      reuse scene 0 tile positions.
    - If `block_index` is provided (usually B=0), it is also passed explicitly.
    - If the installed aicspylibczi version does not accept S/B for this call,
      the function transparently retries without them.

    Parameters
    ----------
    czi : CziFile
        Open CZI file handle.
    n_tiles : int
        Number of mosaic tiles (M dimension).
    scene_index : int or None
        Scene index (S dimension) corresponding to the region being processed.
    block_index : int or None
        Block index (B dimension), if present in the dataset.

    Returns
    -------
    tiles_iter : List[Tuple[int, int, int, float, float]]
        STRICT 5-tuples suitable for decide_and_write_tilescan().
    x_raw : np.ndarray
        Raw X positions (as reported by CZI).
    y_raw : np.ndarray
        Raw Y positions (as reported by CZI).
    unit_hint_raw : str
        Always "unknown" for CZI (unit inference happens later).
    """

    tiles_iter: List[Tuple[int, int, int, float, float]] = []
    x_raw_list: List[float] = []
    y_raw_list: List[float] = []

    missing = 0
    errors_preview: List[str] = []

    # Iterate over mosaic tiles by M index
    for m in range(int(n_tiles)):
        try:
            # ------------------------------------------------------------------
            # Build arguments for get_mosaic_tile_bounding_box()
            #
            # Z=0, C=0 are generally safe for positional metadata.
            # S/B are included *only if explicitly requested* to avoid
            # breaking older aicspylibczi signatures.
            # ------------------------------------------------------------------
            kwargs = {
                "M": int(m),
                "Z": 0,
                "C": 0,
            }

            if scene_index is not None:
                kwargs["S"] = int(scene_index)

            if block_index is not None:
                kwargs["B"] = int(block_index)

            # ------------------------------------------------------------------
            # Query bounding box; retry without S/B if the signature rejects them
            # ------------------------------------------------------------------
            try:
                bb = czi.get_mosaic_tile_bounding_box(**kwargs)
            except TypeError:
                # Older aicspylibczi versions may not accept S/B for this call
                kwargs.pop("S", None)
                kwargs.pop("B", None)
                bb = czi.get_mosaic_tile_bounding_box(**kwargs)

            # ------------------------------------------------------------------
            # Extract raw stage coordinates (NO scaling here)
            # ------------------------------------------------------------------
            x = float(bb.x)
            y = float(bb.y)

            # ------------------------------------------------------------------
            # STRICT 5-tuple:
            # - TileIndex = m (matches on-disk `_s{tile}` convention for CZI)
            # - FieldX / FieldY are placeholders (Zeiss does not expose a grid)
            # - PosX_raw / PosY_raw are raw stage coordinates
            # ------------------------------------------------------------------
            tiles_iter.append((int(m), int(m), 0, x, y))
            x_raw_list.append(x)
            y_raw_list.append(y)

        except Exception as e:
            missing += 1
            if len(errors_preview) < 5:
                errors_preview.append(f"M={m}: {e!r}")
            continue

    if missing > 0:
        msg = (
            f"{BOLD}[WARN]⚠️ {RESET} CZI mosaic position extraction: "
            f"{missing}/{int(n_tiles)} tile(s) missing bounding boxes."
        )
        if errors_preview:
            msg += " Examples: " + "; ".join(errors_preview)
        print(msg)

    unit_hint_raw = "unknown"

    return (
        tiles_iter,
        np.asarray(x_raw_list, dtype=float),
        np.asarray(y_raw_list, dtype=float),
        unit_hint_raw,
    )


def normalize_dims_shape(czi: "CziFile") -> Dict[str, int]:
    """
    Normalize CziFile.get_dims_shape() across aicspylibczi versions into {axis: size}.

    Handler expectations
    --------------------
    CziHandler expects a dict like:
      {"X": 2048, "Y": 2048, "Z": 7, "C": 5, "M": 4, "S": 2, ...}

    Notes
    -----
    aicspylibczi has returned shapes in different forms:
      - dict: {"X": 2048, "Y": (0, 2048), ...}
      - list of dicts: [{"X": (0, 2048)}, {"Y": (0, 2048)}, ...]
      - list of tuples: [("X", (0, 2048)), ("Y", 2048), ...]

    We treat (start, end) as size=end (matches versions commonly seen in practice).
    """
    dims_shape = czi.get_dims_shape()

    def _as_size(v) -> int:
        # v may be int or (start, end)
        if isinstance(v, tuple) and len(v) == 2:
            return int(v[1])
        return int(v)

    if isinstance(dims_shape, dict):
        out: Dict[str, int] = {}
        for axis, rng in dims_shape.items():
            out[str(axis)] = _as_size(rng)
        return out

    if isinstance(dims_shape, list):
        out: Dict[str, int] = {}
        for elem in dims_shape:
            if isinstance(elem, dict):
                for axis, rng in elem.items():
                    out[str(axis)] = _as_size(rng)
            elif isinstance(elem, tuple) and len(elem) >= 2:
                axis, size_or_rng = elem[0], elem[1]
                out[str(axis)] = _as_size(size_or_rng)
            else:
                raise ValueError(f"Unexpected dims_shape element: {elem!r}")
        return out

    raise TypeError(f"Unexpected dims_shape type: {type(dims_shape)!r}")

# ======================================================================================
# ND2 helpers (NO PRINTS; match Nd2Handler contract)
# ======================================================================================

def _nd2_try_float(x: Any) -> Optional[float]:
    """Best-effort float conversion for ND2 metadata fields."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def nd2_get_pixel_to_um_calc(f: "nd2.ND2File") -> Optional[float]:
    """
    Best-effort ND2 pixel size (µm/px) extraction WITHOUT printing.
    Returns None if not found.

    Priority:
      1) f.metadata.channels[0].volume.axesCalibration
      2) voxel_size / pixel_size style attributes
      3) xarray coords spacing (rare)
    """
    # (1) axesCalibration (most reliable for your pipeline)
    try:
        dx_um, dy_um, _dz_um = f.metadata.channels[0].volume.axesCalibration
        dx_um = _nd2_try_float(dx_um)
        dy_um = _nd2_try_float(dy_um)
        if (
            dx_um is not None and dy_um is not None
            and np.isfinite(dx_um) and np.isfinite(dy_um)
            and dx_um > 0 and dy_um > 0
        ):
            return float((dx_um + dy_um) / 2.0)
    except Exception:
        pass

    # (2) direct voxel/pixel size fields (version-dependent)
    for attr in ("voxel_size", "voxelsize", "pixel_size", "pixelsize"):
        try:
            v = getattr(f, attr, None)
            if v is None:
                continue

            if isinstance(v, (tuple, list)) and len(v) >= 2:
                dx_um = _nd2_try_float(v[0])
                dy_um = _nd2_try_float(v[1])
            else:
                dx_um = _nd2_try_float(getattr(v, "x", None))
                dy_um = _nd2_try_float(getattr(v, "y", None))

            if (
                dx_um is not None and dy_um is not None
                and np.isfinite(dx_um) and np.isfinite(dy_um)
                and dx_um > 0 and dy_um > 0
            ):
                return float((dx_um + dy_um) / 2.0)
        except Exception:
            pass

    # (3) xarray coords spacing fallback (rare)
    try:
        darr = f.to_dask(copy=False)
        if hasattr(darr, "coords") and "X" in darr.coords and len(darr.coords["X"]) > 1:
            xs = np.asarray(darr.coords["X"].values, dtype=float)
            xs = np.sort(xs)
            step = float(np.median(np.diff(xs)))
            if np.isfinite(step) and step > 0:
                # coords might be meters; convert if it looks like meters
                if step < 1e-3:
                    step *= 1e6
                return float(step)
    except Exception:
        pass

    return None


def nd2_get_objective_magnification(f: "nd2.ND2File") -> Optional[float]:
    """
    Best-effort objective magnification extraction WITHOUT printing.
    Returns None if not found.
    """

    def _plausible_mag(v: Any) -> Optional[float]:
        val = _nd2_try_float(v)
        if val is None or not np.isfinite(val):
            return None
        return float(val) if 0.25 <= val <= 400 else None

    mag: Optional[float] = None

    # (A) structured metadata object graph (preferred)
    try:
        md = getattr(f, "metadata", None)
        objs: List[Any] = []
        if md is not None:
            objs.append(md)
            try:
                objs.append(md.channels[0])
            except Exception:
                pass

        containers: List[Any] = []
        for obj in objs:
            if obj is None:
                continue
            for attr in ("objective", "objectives", "microscope", "instrument", "optics", "volume"):
                containers.append(getattr(obj, attr, None))

        flat: List[Any] = []
        for c in containers:
            if c is None:
                continue
            if isinstance(c, (list, tuple)):
                flat.extend([x for x in c if x is not None])
            else:
                flat.append(c)

        for obj in flat:
            for key in (
                "magnification", "Magnification",
                "nominalMagnification", "NominalMagnification",
                "objectiveMagnification", "ObjectiveMagnification",
            ):
                m_val = _plausible_mag(getattr(obj, key, None))
                if m_val is not None:
                    mag = m_val
                    break
            if mag is not None:
                break
    except Exception:
        mag = None

    # (B) raw metadata text fallback (last resort)
    if mag is None:
        try:
            raw = ""
            for attr in ("raw_metadata", "metadata_raw", "xml", "meta", "_meta", "_metadata", "_raw_metadata"):
                v = getattr(f, attr, None)
                if v is None:
                    continue
                raw = v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                if raw:
                    break

            if raw:
                m = re.search(r"(NominalMagnification|Magnification)\D{0,10}([\d.,]+)", raw, re.IGNORECASE)
                if m:
                    mag = _plausible_mag(m.group(2).replace(",", "."))
        except Exception:
            mag = None

    return mag


def nd2_get_stage_positions_um(f: "nd2.ND2File") -> Optional[List[Tuple[float, float]]]:
    """
    Extract stage coordinates from ND2 experiment metadata as (x_um, y_um), WITHOUT printing.

    Matches Nd2Handler.infer_tiles_channels() logic:
      - f.experiment XYPosLoop
      - p.stagePositionUm.x / .y
    """
    try:
        xy_loop = next((e for e in f.experiment if getattr(e, "type", None) == "XYPosLoop"), None)
        if xy_loop is None:
            return None
        pts = xy_loop.parameters.points
        coords = [(float(p.stagePositionUm.x), float(p.stagePositionUm.y)) for p in pts]
        return coords if coords else None
    except Exception:
        return None


def nd2_get_mag_and_pixel_to_um(f: "nd2.ND2File") -> Tuple[Optional[float], Optional[float]]:
    """
    Convenience wrapper (NO PRINTS):
      returns (objective_magnification, pixel_to_um_calc)
    """
    return nd2_get_objective_magnification(f), nd2_get_pixel_to_um_calc(f)


# ======================================================================================
# Format handlers: each handler knows how to
#   1) discover regions
#   2) prepare per-region reading context
#   3) read a single tile/channel stack into (Z,Y,X)
#   4) provide metadata inputs to write_region_metadata()
# ======================================================================================

# -----------------------------
# Base handler 
# -----------------------------
from abc import ABC, abstractmethod

class BaseHandler(ABC):
    """
    Abstract base class for format handlers.

    Handlers define how to:
      1) discover_regions(input_dir) -> List[str]
      2) open_region(input_dir, region_index, region_name) -> ctx dict
      3) infer_tiles_channels(ctx) -> tiles/channels/Z/image_dimensions (+ anything else)
      4) read_stack(ctx, tile, channel) -> (Z, Y, X) uint16
      5) close_region(ctx) -> release resources

    Optional metadata support
    -------------------------
    Handlers may implement build_metadata_args(ctx, ...) to return a kwargs dict for
    decide_and_write_tilescan(). The handler MUST NOT write XML directly.
    """


    #: Human-readable mode / format name, e.g. "lif", "czi", "nd2"
    mode: str

    @abstractmethod
    def discover_regions(self, input_dir: Path) -> List[str]:
        """
        Inspect the input directory and return a list of region identifiers
        (e.g. ["R1", "R2"] or arbitrary strings), in processing order.
        """
        raise NotImplementedError

    @abstractmethod
    def open_region(self, *, input_dir: Path, region_index: int, region_name: str) -> Dict[str, Any]:
        """
        Open resources for a given region and return a context dict.

        The context is handler-specific, but must contain everything needed for:
          - infer_tiles_channels(ctx)
          - read_stack(ctx, tile, channel)
          - close_region(ctx)
        """
        raise NotImplementedError

    @abstractmethod
    def infer_tiles_channels(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspect the region-level context and return at least:
            {
              "tiles": List[int],              # tile identifiers
              "channels": List[int],           # channel identifiers
              "size_z": int,                   # number of z-planes
              "image_dimensions": Tuple[int, int],  # (X, Y) in pixels
              ...
            }

        Handlers may include additional keys as needed.
        """
        raise NotImplementedError

    @abstractmethod
    def read_stack(self, ctx: Dict[str, Any], tile: int, channel: int) -> np.ndarray:
        """
        Read and return the (Z, Y, X) uint16 stack for a single (tile, channel).
        The caller assumes:
          - shape is (size_z, Y, X) as reported by infer_tiles_channels
          - dtype is np.uint16
        """
        raise NotImplementedError

    def close_region(self, ctx: Dict[str, Any]) -> None:
        """
        Close any file handles / resources associated with this region.
        Default implementation does nothing; override if your handler opens files.
        """
        return None


    def build_metadata_args(
        self,
        ctx: Dict[str, Any],
        *,
        pixel_to_um_manual: Optional[float],
        deconvolution_method: Optional[str],
        num_iterations: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """
        Return kwargs for decide_and_write_tilescan(), or None to skip this region.
        Must NOT call decide_and_write_tilescan itself.
        Default: no metadata support.
        """
        return None


# -----------------------------
# TIFF handler 
# -----------------------------
class TiffHandler(BaseHandler):
    """
    Supports BOTH tif_autosaved + tif_exported, but keeps mode as provided
    via handler instance (see get_handler()).
    """
    def __init__(self, *, mode: str):
        """
        Initialize a TIFF handler for Leica TIFF datasets.
    
        Leica produces two distinct TIFF filename conventions depending on how
        data were generated:
          - "tif_autosaved": direct LAS X autosave during acquisition
          - "tif_exported": manual export from LAS X
    
        The only difference between these modes is how tile and channel numbers
        are encoded in the filenames. We select regex patterns once here and
        reuse them everywhere to ensure consistent parsing.
        """
        assert mode in ("tif_autosaved", "tif_exported")
        self.mode = mode
    
        # ------------------------------------------------------------------
        # Filename parsing patterns
        #
        # These regexes are used to EXTRACT NUMBERS from filenames.
        # They must be used consistently everywhere to avoid subtle bugs.
        #
        # Example autosaved:
        #   R1--Stage0003--Z0005--C02.tif
        #
        # Example exported:
        #   R1_s0003_ch02.tif
        # ------------------------------------------------------------------
    
        if mode == "tif_autosaved":
            # Match channel number from "--C02", "--C2", etc.
            self.channel_pattern = re.compile(r"--C0*(\d+)(?=\D|$)", re.IGNORECASE)
    
            # Match tile number from "--Stage0003--"
            self.tile_pattern = re.compile(r"--Stage(\d+)--")
    
            # Identify files belonging to "tile 0".
            # These are used only as a REFERENCE to estimate Z count.
            self.sample_indicator = re.compile(r"--Stage0+--")
    
        else:  # tif_exported
            # Match channel number from "_ch02", "_cH2", etc.
            self.channel_pattern = re.compile(r"_ch0*(\d+)(?=\D|$)", re.IGNORECASE)
    
            # Match tile number from "_s0003_"
            self.tile_pattern = re.compile(r"_s(\d+)_")
    
            # Identify files belonging to "tile 0" for Z estimation.
            self.sample_indicator = re.compile(r"_s0+_")

    # ---- internal helper ----
    def _iter_tiffs(self, input_dir: Path) -> Iterable[Path]:
        """
        Yield all TIFF image files belonging to the dataset.
    
        This reproduces the original filtering logic used in the old pipeline:
          - Only '.tif' files
          - Exclude 'dw' in name
          - Exclude LAS X text sidecar files ('.txt')
    
        IMPORTANT:
        This function does NOT filter by region, tile, or channel.
        It only answers: "Which files are valid TIFF image planes?"
        """
        for f in input_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() != ".tif":
                continue
            if "dw" in f.name:
                continue
            yield f


    def discover_regions(self, input_dir: Path) -> List[str]:
        """
        Discover region identifiers from TIFF filenames.
    
        Rules:
          - tif_autosaved: region = everything before "--Stage"
          - tif_exported:  region = everything before "_s"
        """
    
        tif_files = [f.name for f in self._iter_tiffs(input_dir)]
        region_names = set()
    
        for fn in tif_files:
            base = fn.rsplit(".", 1)[0]
    
            if self.mode == "tif_autosaved":
                if "--Stage" in base:
                    region = base.split("--Stage", 1)[0]
                else:
                    # fallback safety
                    region = base.split("--", 1)[0]
    
            else:  # tif_exported
                if "_s" in base:
                    region = base.split("_s", 1)[0]
                else:
                    # fallback safety
                    region = base.split("_", 1)[0]
    
            region = region.strip()
            region_names.add(region)
    
        return sorted(region_names)

    
    def open_region(self, *, input_dir: Path, region_index: int, region_name: str) -> Dict[str, Any]:
        """
        Open TIFF region context and (optionally) pre-extract Leica metadata.
    
        Behavior
        --------
        - Always returns a ctx dict.
        - If Leica metadata folder/XML is missing or cannot be parsed, returns ctx with
          metadata fields left as None/empty (pipeline can still run).
    
        If Leica XML is available, extracts:
          - TileScanInfo tile positions (4-tuples or 5-tuples depending on TileIndex presence)
          - metadata-derived pixel size (µm/px) and unit hint
          - objective magnification (best-effort)
    
        Notes on TileIndex
        ------------------
        - No reconciliation to on-disk tile ids happens here (ctx['tiles'] not known yet).
        - Mapping/subsetting is performed later in build_metadata_args() after infer_tiles_channels().
        """

        ctx: Dict[str, Any] = dict(
            input_dir=Path(input_dir),
            region_index=int(region_index),
            region=str(region_name),
            mode=self.mode,
        )
    
        # Core "uniform" metadata fields (same semantics as other handlers)
        ctx["objective_mag"] = None
        ctx["objective_mag_source"] = None
        ctx["pixel_to_um_calc"] = None
        ctx["unit_hint_raw"] = ""
    
        # Optional Leica metadata (TIFF-specific but harmlessly present)
        ctx["tiff_metadata_dir"] = None
        ctx["tiff_metadata_file"] = None
        ctx["tiff_xml_root"] = None
    
        # Mosaic fields (kept consistent with other handlers)
        # tiles_iter may be 5-tuples (TileIndex, FieldX, FieldY, PosX, PosY)
        # or 4-tuples (FieldX, FieldY, PosX, PosY) if TileIndex is absent in XML.
        ctx["mosaic_tiles_iter"] = None
        ctx["mosaic_x_raw"] = None
        ctx["mosaic_y_raw"] = None
    
        try:
            # 1) Locate Leica metadata folder (case-insensitive "metadata" under input_dir)
            input_metadata_dir = tiff_find_metadata_dir_case_insensitive(ctx["input_dir"], "metadata")
            if input_metadata_dir is None:
                return ctx  # no Leica metadata
    
            # 2) Choose Leica XML/XLF file that matches this region (ignores Properties dumps)
            region_token = (region_name or "").strip()
            md_file = tiff_pick_leica_xml(input_metadata_dir, region_token=region_token)
            if md_file is None:
                return ctx
    
            root = tiff_parse_xml_safe(md_file)
            if root is None:
                return ctx
    
            ctx["tiff_metadata_dir"] = input_metadata_dir
            ctx["tiff_metadata_file"] = md_file
            ctx["tiff_xml_root"] = root
    
            # 3) Extract tile stage positions from Leica TileScanInfo
            #
            # Contract of tiff_collect_tiles_from_tilescaninfo():
            #   - If XML has TileIndex: returns 5-tuples:
            #       (TileIndex, FieldX, FieldY, PosX, PosY)
            #   - Else: returns 4-tuples:
            #       (FieldX, FieldY, PosX, PosY)
            #
            # NOTE:
            # - tiff_collect_tiles_from_tilescaninfo() already:
            #     * sorts deterministically (FieldY, FieldX)
            #     * normalizes types (ints/floats)
            # - We do NOT assign TileIndex here; that's done later in build_metadata_args().
            tiles_iter = list(tiff_collect_tiles_from_tilescaninfo(root) or [])
            if not tiles_iter:
                raise ValueError("TileScanInfo contained no tiles (tiles_iter empty).")
    
            n = len(tiles_iter[0])
            if n == 5:
                # (TileIndex, FieldX, FieldY, PosX, PosY)
                x_raw = np.asarray([t[3] for t in tiles_iter], dtype=float)
                y_raw = np.asarray([t[4] for t in tiles_iter], dtype=float)
            elif n == 4:
                # (FieldX, FieldY, PosX, PosY)
                x_raw = np.asarray([t[2] for t in tiles_iter], dtype=float)
                y_raw = np.asarray([t[3] for t in tiles_iter], dtype=float)
            else:
                raise ValueError(f"Unexpected tiles_iter tuple length: {n} (expected 4 or 5).")
    
            ctx["mosaic_tiles_iter"] = tiles_iter
            ctx["mosaic_x_raw"] = x_raw
            ctx["mosaic_y_raw"] = y_raw
    
            # 4) Pixel size + magnification (metadata-derived)
            meta = tiff_extract_pixel_size_and_magnification(
                root,
                pixel_to_um_manual=None,  # manual override comes from pipeline arg, not here
                rtol_warn=0.02,
            )
    
            ctx["pixel_to_um_calc"] = meta.get("pixel_to_um_calc", None)
            ctx["unit_hint_raw"] = (meta.get("unit_hint_raw", "") or "")
            mag = meta.get("magnification", None)
            ctx["objective_mag"] = mag
            ctx["objective_mag_source"] = "Leica XML" if mag is not None else None
    
        except Exception as e:
            print(f"{BOLD}[WARN]⚠️ {RESET} TIFF handler: failed to pre-extract Leica metadata for region '{region_name}': {e}")
    
        return ctx
    
    def _matches_channel(self, filename: str, ch: int) -> bool:
        """
        Return True if the filename belongs to the given channel number.
    
        This MUST use self.channel_pattern instead of string matching
        because Leica uses inconsistent zero-padding and capitalization:
          - C2, C02, ch2, ch02, cH2, ...
    
        By always parsing the number and comparing integers, we ensure
        that channel detection and file assignment remain consistent.
        """
        m = self.channel_pattern.search(filename)
        return (m is not None) and (int(m.group(1)) == int(ch))


    def infer_tiles_channels(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer dataset structure for a single region.
    
        This method answers four concrete questions for the pipeline:
          1) Which tile indices exist in this region?
          2) Which channel indices exist?
          3) How many Z planes belong to each (tile, channel) stack?
          4) What are the image dimensions in pixels (X, Y)?
    
        IMPORTANT:
        - This uses ONLY filename patterns and TIFF headers.
        - No Leica XML, no metadata inference, no assumptions.
        - Behavior is intentionally equivalent to the old preprocessing code.
        """
    
        input_dir: Path = ctx["input_dir"]
        region: str = ctx["region"]
    
        # ------------------------------------------------------------
        # 1) Collect all TIFF files belonging to this region
        #
        # We rely on the region token being part of the filename,
        # exactly as in the legacy pipeline.
        # ------------------------------------------------------------
        tif_files = list(self._iter_tiffs(input_dir))
        prefix = f"{region}--" if self.mode == "tif_autosaved" else f"{region}_"
        filtered_tifs = [f for f in tif_files if f.name.startswith(prefix)]

        if not filtered_tifs:
            raise RuntimeError(f"No TIFFs found for region token '{region}'")
    
        # ------------------------------------------------------------
        # 2) Detect available channels from filenames
        #
        # Example matches:
        #   tif_autosaved  → "--C01", "--C002"
        #   tif_exported   → "_ch1", "_CH002"
        #
        # Zero-padding is ignored by int().
        # ------------------------------------------------------------
        channel_set = set()
        for f in filtered_tifs:
            m = self.channel_pattern.search(f.name)
            if m:
                channel_set.add(int(m.group(1)))
    
        channels = sorted(channel_set)
        if not channels:
            raise RuntimeError(f"No channels detected for region '{region}'")
    
        # ------------------------------------------------------------
        # 3) Detect tile indices and identify "sample" tiles
        #
        # Tiles are inferred from filename patterns:
        #   tif_autosaved  → "--Stage###--"
        #   tif_exported   → "_s###_"
        #
        # Sample tiles are those belonging to the lowest tile index,
        # used to estimate Z depth and image shape.
        # ------------------------------------------------------------
        tiles = set()
        sample_tiles: List[Path] = []
    
        for f in filtered_tifs:
            m = self.tile_pattern.search(f.name)
            if m:
                tiles.add(int(m.group(1)))
    
            # Sample tiles are typically Stage000 / s000
            if self.sample_indicator.search(f.name):
                sample_tiles.append(f)
    
        # If no explicit sample tiles were found, fall back to
        # the lowest tile index present.
        if not sample_tiles and tiles:
            lowest = min(tiles)
            if self.mode == "tif_exported":
                fallback = re.compile(rf"_s0*{lowest}_")
            else:
                fallback = re.compile(rf"--Stage0*{lowest}--")
    
            sample_tiles = [f for f in filtered_tifs if fallback.search(f.name)]
    
        tiles = sorted(tiles)
        if not tiles:
            raise RuntimeError(f"No tiles detected for region '{region}'")
    
        # ------------------------------------------------------------
        # 4) Infer Z depth
        #
        # Old logic (kept intentionally):
        #   number of planes for sample tile / number of channels
        #
        # This assumes:
        #   - one TIFF per (Z, C) plane
        #   - consistent ordering across tiles
        # ------------------------------------------------------------
        if sample_tiles:
            if len(sample_tiles) % len(channels) != 0:
                print(
                    f"{BOLD}[WARN]⚠️ {RESET} TIFF handler: sample_tiles ({len(sample_tiles)}) not divisible by "
                    f"channels ({len(channels)}); inferred size_z may be wrong."
                )
            size_z = max(1, len(sample_tiles) // len(channels))
        else:
            size_z = 1

    
        # ------------------------------------------------------------
        # 5) Infer image dimensions from a single TIFF header
        #
        # We only read the TIFF header (no pixel data),
        # and expect a single 2D plane (Y, X).
        # ------------------------------------------------------------
        sample_path = sample_tiles[0] if sample_tiles else filtered_tifs[0]
    
        with tifffile.TiffFile(str(sample_path)) as tf:
            shp = tf.pages[0].shape  # (Y, X)
    
        if len(shp) != 2:
            raise RuntimeError(
                f"Expected 2D plane for '{sample_path.name}', got shape={shp}"
            )
    
        image_dimensions = (int(shp[1]), int(shp[0]))  # (X, Y)
    
        # ------------------------------------------------------------
        # 6) Build a lookup table:
        #
        #   (tile_index, channel_index) → list of TIFF planes (Z order)
        #
        # This allows read_stack() to be simple and fast.
        # ------------------------------------------------------------
        tile_to_files: Dict[int, List[Path]] = {t: [] for t in tiles}
    
        for f in filtered_tifs:
            m = self.tile_pattern.search(f.name)
            if not m:
                continue
            t = int(m.group(1))
            if t in tile_to_files:
                tile_to_files[t].append(f)
    
        tile_channel_files: Dict[Tuple[int, int], List[Path]] = {}
        for t, files_in_tile in tile_to_files.items():
            for ch in channels:
                tile_channel_files[(t, ch)] = [
                    f for f in files_in_tile
                    if self._matches_channel(f.name, ch)
                ]
    
        # ------------------------------------------------------------
        # 7) Store everything needed by read_stack() in ctx
        #
        # The returned dict is intentionally minimal and
        # contains only what the pipeline needs.
        # ------------------------------------------------------------
        ctx.update(dict(
            filtered_tifs=filtered_tifs,
            tiles=tiles,
            channels=channels,
            size_z=size_z,
            image_dimensions=image_dimensions,
            tile_channel_files=tile_channel_files,
        ))
    
        return {
            "tiles": tiles,
            "channels": channels,
            "size_z": size_z,
            "image_dimensions": image_dimensions,
        }


    def read_stack(self, ctx: Dict[str, Any], tile: int, channel: int) -> np.ndarray:
        """
        Read a (Z, Y, X) stack for one tile and one channel.
    
        This is intentionally the *simple, old-style* TIFF reader:
          - Each Z plane is stored as a separate TIFF file
          - Files are read sequentially with tifffile.imread
          - No multiprocessing, no retries, no shape repair
          - Behavior matches the legacy pipeline exactly
    
        Assumptions:
          - ctx["tile_channel_files"] was built by infer_tiles_channels()
          - Each file contains a single 2D plane (Y, X)
          - All planes for a given (tile, channel) have identical shape
        """
        tile = int(tile)
        channel = int(channel)
    
        # Look up all TIFF files belonging to this (tile, channel) pair.
        # This should already represent one full Z stack.
        tile_channel_files = ctx["tile_channel_files"]
        files = tile_channel_files.get((tile, channel), [])
    
        if not files:
            raise FileNotFoundError(
                f"No TIFF planes found for tile={tile}, channel={channel}"
            )
    
        # Sort filenames to ensure deterministic Z ordering.
        # This preserves historical behavior and avoids OS-dependent ordering.
        files = sorted(files, key=lambda p: str(p))
    
        # Read each plane and stack into a 3D array: (Z, Y, X)
        stack = np.stack(
            [tifffile.imread(str(f)) for f in files],
            axis=0
        )
    
        # Defensive check: each stack must be exactly 3D
        if stack.ndim != 3:
            raise ValueError(
                f"Expected (Z,Y,X) stack, got shape={stack.shape} "
                f"for tile={tile}, channel={channel}"
            )
    
        # Ensure uint16 without copying if already correct
        return stack.astype(np.uint16, copy=False)

    def build_metadata_args(
        self,
        ctx: Dict[str, Any],
        *,
        pixel_to_um_manual: Optional[float],
        deconvolution_method: Optional[str],
        num_iterations: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare keyword arguments for decide_and_write_tilescan().

        Responsibilities
        ----------------
        - Validate Leica metadata presence
        - Normalize tile records to STRICT 5-tuples:
            (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
        - Align TileIndex with on-disk tile ids (ctx["tiles"]) so downstream mapping is stable
        - Return a kwargs dict for decide_and_write_tilescan() (caller supplies out_xml_path)

        Subset policy
        -------------
        - If XML provides TileIndex (5-tuples):
            * Treat TileIndex as an identity key.
            * Keep only those tiles whose TileIndex exists on disk (supports subsets safely).
            * Preserve on-disk ordering (sorted ctx["tiles"]) for deterministic downstream behavior.

        - If XML lacks TileIndex (4-tuples):
            * Preserve the XML <Tile> element order exactly.
              IMPORTANT: Leica often writes tiles in acquisition order (commonly serpentine).
              Sorting by (FieldY, FieldX) can silently convert serpentine → raster and corrupt
              the mapping between filename tile ids (_s0000_, _s0001_, ...) and stage positions.
            * ASSUME on-disk filename tile ids are 0-based indices into this XML order:
                on_disk_tile_id t -> xml_tiles[t]
            * Supports subsets only if the on-disk tile ids remain a subset of [0..n_xml-1].
              If filename ids are re-labeled / non-contiguous / not acquisition-indexed, mapping
              may be invalid and should be rejected.

        Returns
        -------
        Optional[Dict[str, Any]]
            Kwargs for decide_and_write_tilescan(), or None to skip metadata writing.
        """

    
        # ------------------------------------------------------------
        # Basic sanity check: image geometry must already be known
        # ------------------------------------------------------------
        image_dimensions: Optional[Tuple[int, int]] = ctx.get("image_dimensions", None)
        if image_dimensions is None:
            print(
                "[ERROR] TIFF handler: image_dimensions missing in ctx "
                "(infer_tiles_channels() must be called first)."
            )
            return None
    
        # ------------------------------------------------------------
        # Leica metadata requirements
        # ------------------------------------------------------------
        md_file = ctx.get("tiff_metadata_file", None)
        tiles_iter = ctx.get("mosaic_tiles_iter", None)
    
        if md_file is None:
            print(
                f"[ERROR] TIFF handler: no Leica XML/XLF metadata for region "
                f"'{ctx.get('region')}'. Skipping metadata."
            )
            return None
    
        if tiles_iter is None:
            print(
                "[ERROR] TIFF handler: missing tile positions "
                "(TileScanInfo not found or incomplete). Skipping metadata."
            )
            return None
    
        # ------------------------------------------------------------
        # Normalize tiles_iter to 5-tuples:
        #   (TileIndex, FieldX, FieldY, PosX, PosY)
        # ------------------------------------------------------------
        tiles_iter = list(tiles_iter or [])
        if not tiles_iter:
            print("[ERROR] TIFF handler: tiles_iter empty. Skipping metadata.")
            return None
    
        t0 = tiles_iter[0]
        if (not isinstance(t0, tuple)) or (len(t0) not in (4, 5)):
            print(
                f"[ERROR] TIFF handler: unexpected tiles_iter entry: {type(t0)} / {t0}. "
                "Expected tuples of length 4 or 5. Skipping metadata."
            )
            return None
    
        # On-disk tiles MUST be known here (infer_tiles_channels ran)
        file_tiles = ctx.get("tiles", None)
        if not file_tiles:
            print(
                "[ERROR] TIFF handler: ctx['tiles'] missing or empty; "
                "infer_tiles_channels() must run first. Skipping metadata."
            )
            return None
        file_tiles_sorted = sorted(int(t) for t in file_tiles)
    
        # Defensive init (avoids UnboundLocalError if code changes later)
        x_raw = None
        y_raw = None
    
        # ----------------------------
        # Case A: 5-tuples already have TileIndex
        # ----------------------------
        if len(t0) == 5:
            # Normalize types
            tiles_5 = [(int(a), int(b), int(c), float(d), float(e)) for (a, b, c, d, e) in tiles_iter]
    
            # Allow subset: keep only tiles whose TileIndex exists on disk
            file_set = set(file_tiles_sorted)
            kept = [t for t in tiles_5 if t[0] in file_set]
            dropped = [t[0] for t in tiles_5 if t[0] not in file_set]
    
            if not kept:
                print(
                    "[ERROR] TIFF handler: TileScanInfo TileIndex values do not overlap with on-disk tiles. "
                    "Skipping metadata to avoid wrong tile mapping."
                )
                return None
    
            if dropped:
                print(
                    f"{BOLD}[WARN]⚠️ {RESET} TIFF handler: XML contains {len(dropped)} tile(s) not present on disk; "
                    f"dropping them. Example dropped TileIndex: {dropped[:10]}{'...' if len(dropped) > 10 else ''}"
                )
    
            tiles_iter = kept
    
            # Keep the same tile ordering as on-disk filenames for determinism downstream.
            rank = {t: i for i, t in enumerate(file_tiles_sorted)}
            tiles_iter = sorted(tiles_iter, key=lambda t: rank.get(int(t[0]), 10**12))
    
            # Rebuild x_raw/y_raw in EXACTLY this kept order
            x_raw = np.asarray([t[3] for t in tiles_iter], dtype=float)
            y_raw = np.asarray([t[4] for t in tiles_iter], dtype=float)
    
        # ----------------------------
        # Case B: 4-tuples (FieldX, FieldY, PosX, PosY)
        # ----------------------------
        else:
            # ------------------------------------------------------------------
            # IMPORTANT ORDERING POLICY (Case B: no TileIndex in XML)
            #
            # When Leica XML omits TileIndex, the <Tile> elements are often stored
            # in *acquisition order* (commonly serpentine). In those files FieldX/FieldY
            # may not represent a true 2D grid ordering, and sorting by (FieldY, FieldX)
            # can silently corrupt the mapping from on-disk tile id -> stage position.
            #
            # Therefore:
            #   - Preserve the XML order exactly.
            #   - Assume filename tile ids are 0-based indices into that list:
            #       on_disk_tile_id t  ->  xml_tiles[t]
            # ------------------------------------------------------------------

            # Normalize XML tiles but DO NOT sort them
            xml_tiles = [(int(fx), int(fy), float(px), float(py)) for (fx, fy, px, py) in tiles_iter]
            n_xml = len(xml_tiles)

            print(
                "[META] Case B: XML has no TileIndex — preserving XML acquisition order."
            )

            if not file_tiles_sorted:
                print("[ERROR] TIFF handler: no on-disk tiles found for mapping. Skipping metadata.")
                return None

            if n_xml <= 0:
                print("[ERROR] TIFF handler: XML contains zero tiles (n_xml=0). Skipping metadata.")
                return None

            # Validate mapping range (tile id must be a valid index into xml_tiles)
            bad = [t for t in file_tiles_sorted if not (0 <= t < n_xml)]
            if bad:
                print(
                    "[ERROR] TIFF handler: filename tile ids exceed XML tile count "
                    f"(bad={bad[:10]}{'...' if len(bad) > 10 else ''}, "
                    f"max_file_tile={max(file_tiles_sorted)}, n_xml={n_xml}). "
                    "Cannot safely map tiles. Skipping metadata."
                )
                return None


            # INFO: explain mapping decision
            preview = file_tiles_sorted[:10]
            print(
                "[INFO] TIFF handler: mapping XML tiles by filename tile ids (assumed 0-based indexing): "
                f"file_tiles={preview}{'...' if len(file_tiles_sorted) > 10 else ''}, n_xml={n_xml}"
            )
            print(f"[INFO] TIFF handler: max on-disk tile id = {max(file_tiles_sorted)}")
            print(f"[INFO] TIFF handler: mapping {len(file_tiles_sorted)} on-disk tile(s) onto {n_xml} XML tile(s).")

            # Build final tiles_iter using filename tile id as TileIndex
            mapped_tiles_iter: List[Tuple[int, int, int, float, float]] = []
            for tid in file_tiles_sorted:
                fx, fy, px, py = xml_tiles[tid]
                mapped_tiles_iter.append((tid, fx, fy, px, py))
            tiles_iter = mapped_tiles_iter

            # INFO: show a few concrete mappings
            k = min(10, len(tiles_iter))
            pairs = [
                f"{t[0]}->(fx={t[1]},fy={t[2]},x={t[3]:.6f},y={t[4]:.6f})"
                for t in tiles_iter[:k]
            ]
            print(
                "[INFO] TIFF handler: tile mapping (filename->xml): "
                + "; ".join(pairs)
                + (f"; ... (+{len(tiles_iter)-k} more)" if len(tiles_iter) > k else "")
            )

            # Rebuild x_raw / y_raw aligned with final tile order
            x_raw = np.asarray([t[3] for t in tiles_iter], dtype=float)
            y_raw = np.asarray([t[4] for t in tiles_iter], dtype=float)

    
        # Defensive: ensure we set x_raw/y_raw
        if x_raw is None or y_raw is None:
            print("[ERROR] TIFF handler: internal error: x_raw/y_raw not set. Skipping metadata.")
            return None
    
        # More informative mismatch reporting (actionable)
        if len(tiles_iter) != len(file_tiles_sorted):
            file_set = set(file_tiles_sorted)
            mapped_set = set(int(t[0]) for t in tiles_iter)  # TileIndex after normalization
    
            missing_in_xml = sorted(file_set - mapped_set)
            extra_in_xml = sorted(mapped_set - file_set)
    
            if missing_in_xml:
                print(
                    f"{BOLD}[WARN]⚠️ {RESET} TIFF handler: {len(missing_in_xml)} on-disk tile(s) have no matching XML TileIndex; "
                    f"missing={missing_in_xml[:10]}{'...' if len(missing_in_xml) > 10 else ''}"
                )
            if extra_in_xml:
                print(
                    f"{BOLD}[WARN]⚠️ {RESET} TIFF handler: {len(extra_in_xml)} XML tile(s) are not present on disk after mapping; "
                    f"extra={extra_in_xml[:10]}{'...' if len(extra_in_xml) > 10 else ''}"
                )
    
        # Store normalized result back into ctx so downstream sees canonical form
        ctx["mosaic_tiles_iter"] = tiles_iter
        ctx["mosaic_x_raw"] = x_raw
        ctx["mosaic_y_raw"] = y_raw
    
        # Final sanity check: numeric arrays must be non-empty
        if x_raw.size == 0 or y_raw.size == 0:
            print(
                "[ERROR] TIFF handler: tile position arrays are empty after normalization. "
                "Skipping metadata."
            )
            return None
    
        # Optional metadata (used if present)
        pixel_to_um_calc = ctx.get("pixel_to_um_calc", None)
        unit_hint_raw = ctx.get("unit_hint_raw", "")
    
        mag = ctx.get("objective_mag", None)
        mag_src = ctx.get(
            "objective_mag_source",
            "Leica XML" if mag is not None else None,
        )
    
        print(
            f"[INFO] Generating TIFF TileScanInfo XML metadata "
            f"from Leica XML '{md_file.name}'"
        )
    
        return dict(
            x_raw=x_raw,
            y_raw=y_raw,
            image_dimensions=image_dimensions,
            pixel_to_um_manual=pixel_to_um_manual,
            pixel_to_um_calc=pixel_to_um_calc,
            unit_hint_raw=unit_hint_raw,
            off_tol=0.25,
            tiles_iter=tiles_iter,
            app_name="LAS X",
            deconvolution_method=deconvolution_method,
            deconvolution_iterations=num_iterations,
            objective_mag=mag,
            objective_mag_source=mag_src,
        )
    

# -----------------------------
# LIF handler (region-aware)
# -----------------------------
class LifHandler(BaseHandler):
    
    mode = "lif"

    # ------------------------------------------------------------------
    # Lazy import
    # ------------------------------------------------------------------
    @staticmethod
    def _LifFile():
        """
        Lazy import to avoid crashing the Jupyter kernel during module import.
        Only imports readlif when LIF mode is actually used.
        """
        try:
            from readlif.reader import LifFile
        except Exception as e:
            raise ImportError(
                "LIF mode requires 'readlif'. Install it to use mode='lif'."
            ) from e
        return LifFile


    # -----------------------------
    # Region discovery
    # -----------------------------
    def discover_regions(self, input_dir: Path) -> List[str]:
        LifFile = self._LifFile()

        lif_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".lif"])
        if not lif_files:
            return []

        names: List[str] = []

        # case 1: multiple LIF files -> treat as one region per file (first image)
        if len(lif_files) > 1:
            for fp in lif_files:
                lf = LifFile(fp)
                try:
                    if not getattr(lf, "image_list", None):
                        nm = fp.stem
                    else:
                        first = lf.image_list[0]
                        nm = str(first.get("name") or fp.stem)
        
                    nm = nm.replace("/", "_")
                    # ensures uniqueness across files
                    nm = f"{fp.stem}__{nm}"
        
                    names.append(nm)
        
                finally:
                    try:
                        lf.close()
                    except Exception:
                        pass


        # case 2: single LIF file -> multiple images = multiple regions
        else:
            fp = lif_files[0]
            lf = LifFile(fp)
            try:
                img_list = getattr(lf, "image_list", None) or []
                for d in img_list:
                    nm = str(d.get("name") or fp.stem)
                    names.append(nm.replace("/", "_"))
            finally:
                try:
                    lf.close()
                except Exception:
                    pass

        # stable unique (preserve order)
        seen = set()
        out: List[str] = []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out

    # -----------------------------
    # Open a specific region
    # -----------------------------
    def open_region(self, *, input_dir: Path, region_index: int, region_name: str) -> Dict[str, Any]:
        LifFile = self._LifFile()

        lif_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".lif"])
        if not lif_files:
            raise ValueError("No .lif files found")

        if region_index < 0:
            raise IndexError("region_index must be >= 0")

        # --- pick file + image index ---
        if len(lif_files) > 1:
            # one file per region
            if region_index >= len(lif_files):
                raise IndexError(f"region_index {region_index} out of range for {len(lif_files)} .lif files")
            filepath = lif_files[region_index]
            lf = LifFile(filepath)
            if not getattr(lf, "image_list", None):
                try:
                    lf.close()
                except Exception:
                    pass
                raise RuntimeError(f"LIF file '{filepath.name}' has no images (image_list empty).")

            image_dict = lf.image_list[0]
            image = lf.get_image(0)
        else:
            # one file with many images
            filepath = lif_files[0]
            lf = LifFile(filepath)
            img_list = getattr(lf, "image_list", None) or []
            if region_index >= len(img_list):
                try:
                    lf.close()
                except Exception:
                    pass
                raise IndexError(f"region_index {region_index} out of range for LIF image_list size {len(img_list)}")

            image_dict = img_list[region_index]
            image = lf.get_image(region_index)

        # normalize name
        image_name = str(image_dict.get("name") or region_name or filepath.stem).replace("/", "_")

        ctx: Dict[str, Any] = dict(
            input_dir=input_dir,
            mode=self.mode,
            region_index=int(region_index),
            region=image_name,          # prefer a single standard key: ctx["region"]
            lif_filepath=filepath,

            lif_file=lf,                # kept open; closed in close_region()
            lif_image=image,
            lif_image_dict=image_dict,

            # you can keep aliases if older code expects them:
            image=image,
            image_dict=image_dict,
            region_name=image_name,
        )

        # optional: expose raw XML header/root if readlif provides it
        ctx["lif_xml_root"] = getattr(lf, "xml_root", None)

        # best-effort: objective mag + pixel size from metadata
        mag = None
        pixel_to_um_calc = None
        try:
            # If you already wrote this helper, keep using it.
            # Should return: (objective_mag, pixel_to_um)
            mag, pixel_to_um_calc = lif_get_mag_and_pixel_to_um(ctx)
        except Exception:
            pass

        ctx["objective_mag"] = mag
        ctx["objective_mag_source"] = "LIF xml_header/xml_root" if mag is not None else None
        ctx["pixel_to_um_calc"] = pixel_to_um_calc

        return ctx

    # -----------------------------
    # Infer tiles/channels/dims
    # -----------------------------
    def infer_tiles_channels(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        d = ctx["lif_image_dict"]
        dims = d.get("dims", None)
        if dims is None:
            raise RuntimeError(f"LIF image_dict has no 'dims' for region '{ctx.get('region')}'")

        # dims.x / dims.y
        try:
            x_px = int(getattr(dims, "x"))
            y_px = int(getattr(dims, "y"))
        except Exception as e:
            raise RuntimeError(f"Invalid LIF dims.x/dims.y for region '{ctx.get('region')}': {e!r}") from e

        if x_px <= 0 or y_px <= 0:
            raise RuntimeError(f"LIF image has non-positive dimensions (x={x_px}, y={y_px})")

        image_dimensions = (x_px, y_px)  # (width, height)

        # dims.z / dims.m
        size_z = int(getattr(dims, "z", 1) or 1)
        n_tiles = int(getattr(dims, "m", 1) or 1)
        if n_tiles <= 0:
            n_tiles = 1

        # channels field
        n_channels = int(d.get("channels", 1) or 1)
        if n_channels <= 0:
            n_channels = 1

        mosaic = d.get("mosaic_position", None)

        # IMPORTANT: if mosaic exists, prefer it for tile count (often more reliable)
        if mosaic is not None:
            try:
                n_tiles = max(1, len(mosaic))
            except Exception:
                pass

        ctx.update(
            tiles=list(range(n_tiles)),
            channels=list(range(n_channels)),
            size_z=size_z,
            image_dimensions=image_dimensions,
            mosaic=mosaic,
        )
        
        return {
            "tiles": ctx["tiles"],
            "channels": ctx["channels"],
            "size_z": ctx["size_z"],
            "image_dimensions": ctx["image_dimensions"],
        }


    # -----------------------------
    # Read a (Z,Y,X) stack
    # -----------------------------
    def read_stack(self, ctx: Dict[str, Any], tile: int, channel: int) -> np.ndarray:
        image = ctx["lif_image"]
        tile = int(tile)
        channel = int(channel)

        z_planes = [np.asarray(zf) for zf in image.get_iter_z(m=tile, c=channel)]
        if not z_planes:
            raise RuntimeError(
                f"LIF read_stack(): no Z planes for tile={tile}, channel={channel} (region='{ctx.get('region')}')"
            )

        stack = np.stack(z_planes, axis=0).astype(np.uint16, copy=False)
        return stack
    
    # -----------------------------
    # Build args for decide_and_write_tilescan()
    # (robust: TileIndex matches ctx["tiles"] / on-disk tile ids)
    # -----------------------------
    def build_metadata_args(
        self,
        ctx: Dict[str, Any],
        *,
        pixel_to_um_manual: Optional[float],
        deconvolution_method: Optional[str],
        num_iterations: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        mosaic = ctx.get("mosaic", None)
        region = ctx.get("region", "unknown")
        image_dimensions: Tuple[int, int] = ctx["image_dimensions"]
    
        if mosaic is None:
            print(f"{BOLD}[WARN]⚠️ {RESET} No mosaic_position found for LIF region '{region}'. Skipping TileScanInfo.")
            return None
    
        if not isinstance(mosaic, (list, tuple)) or len(mosaic) == 0:
            print(f"{BOLD}[WARN]⚠️ {RESET} Empty/invalid mosaic_position for LIF region '{region}'. Skipping TileScanInfo.")
            return None
    
        # Tiles we actually process / write to disk (these must match filenames _sNNN_)
        tiles_ctx = ctx.get("tiles", None)
        if tiles_ctx is None:
            tiles_ctx = list(range(len(mosaic)))
        else:
            tiles_ctx = [int(t) for t in tiles_ctx]
    
        # Sanity / mismatch handling
        if len(mosaic) != len(tiles_ctx):
            print(
                f"{BOLD}[WARN]⚠️ {RESET} Tile count mismatch for '{region}': "
                f"len(mosaic_position)={len(mosaic)} vs len(ctx['tiles'])={len(tiles_ctx)}. "
                f"Will only use overlapping indices."
            )
    
        tiles_iter = []
        try:
            # IMPORTANT:
            # - TileIndex is the tile id used by the pipeline (ctx["tiles"]) and thus filenames.
            # - We index mosaic by that same tile id when possible.
            for tile_id in tiles_ctx:
                if tile_id < 0 or tile_id >= len(mosaic):
                    # skip tiles that exist on disk but have no mosaic position entry
                    continue
                p = mosaic[tile_id]  # expected: (FieldX, FieldY, PosX, PosY) with Pos in meters
                tiles_iter.append((int(tile_id), int(p[0]), int(p[1]), float(p[2]), float(p[3])))
    
            if not tiles_iter:
                print(f"[ERROR] No overlapping tiles between ctx['tiles'] and mosaic_position for '{region}'. Skipping.")
                return None
    
            x_raw = np.asarray([t[3] for t in tiles_iter], dtype=float)
            y_raw = np.asarray([t[4] for t in tiles_iter], dtype=float)
    
        except Exception as e:
            print(f"[ERROR] Could not parse mosaic_position for '{region}': {e!r}. Skipping TileScanInfo.")
            return None
    
        if x_raw.size == 0 or y_raw.size == 0:
            print(f"[ERROR] Parsed mosaic positions are empty for '{region}'. Skipping TileScanInfo.")
            return None
    
        return dict(
            x_raw=x_raw,
            y_raw=y_raw,
            image_dimensions=image_dimensions,
            pixel_to_um_manual=pixel_to_um_manual,
            pixel_to_um_calc=ctx.get("pixel_to_um_calc", None),
            unit_hint_raw="m",  # LIF mosaic positions are meters
            off_tol=0.25,
            tiles_iter=tiles_iter,
            app_name="LAS AF",
            # NOTE: out_xml_path intentionally NOT included (pipeline passes it)
            deconvolution_method=deconvolution_method,
            deconvolution_iterations=num_iterations,
            objective_mag=ctx.get("objective_mag", None),
            objective_mag_source=ctx.get("objective_mag_source", None),
        )


    # -----------------------------
    # Close resources
    # -----------------------------
    def close_region(self, ctx: Dict[str, Any]) -> None:
        lf = ctx.get("lif_file", None)
        try:
            if lf is not None:
                lf.close()
        except Exception:
            pass
        finally:
            ctx["lif_file"] = None
            ctx["lif_image"] = None
            ctx["lif_image_dict"] = None
            ctx["lif_xml_root"] = None
            # keep compatibility aliases clean too
            ctx["image"] = None
            ctx["image_dict"] = None


# ----------------------------
# CZI handler 
# ----------------------------
class CziHandler(BaseHandler):
    """
    Design goals (match LIF/TIFF)
    -----------------------------
    - discover_regions(): returns a stable list of region names (pipeline uses indices)
    - open_region(): opens the dataset for a given region_index; keeps file open until close_region()
    - infer_tiles_channels(): ONLY infers tiles/channels/Z/image_dimensions (NO mosaic parsing here)
    - build_metadata_args(): parses/normalizes mosaic positions and returns kwargs for decide_and_write_tilescan()
        * MUST NOT depend on ctx["out_xml_path"]
        * MUST return STRICT 5-tuples: (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
        * TileIndex MUST match on-disk tile ids (the `_s{tile}` numbers in filenames)
          because mipped_to_OME_tiffs maps positions by TileIndex identity.

    Notes on units
    --------------
    - CZI stage/mosaic coordinate units are not guaranteed. Therefore:
        * We do NOT hardcode unit_hint_raw="m".
        * We prefer the helper czi_get_mosaic_positions() to return a unit hint if it can.
        * Otherwise we pass unit_hint_raw="unknown" and let decide_and_write_tilescan()
          use its hypothesis test (if pixel size is available).
    """
    mode = "czi"

    # ------------------------------------------------------------------
    # Lazy import
    # ------------------------------------------------------------------
    @staticmethod
    def _CziFile():
        """
        Lazy import to avoid crashing the Jupyter kernel during module import.
        Only imports aicspylibczi when CZI mode is actually used.
        """
        try:
            from aicspylibczi import CziFile
        except Exception as e:
            raise ImportError(
                "CZI mode requires 'aicspylibczi'. Install it to use mode='czi'."
            ) from e
        return CziFile

    # ------------------------------------------------------------------
    # Region discovery
    # ------------------------------------------------------------------
    def discover_regions(self, input_dir: Path) -> List[str]:
        CziFile = self._CziFile()

        # Match LIF/TIFF: use sorted() for stable ordering
        czi_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".czi"])
        if not czi_files:
            return []

        # For now: assume one CZI per directory; "S" (scene) gives region count.
        czi = CziFile(str(czi_files[0]))
        try:
            dims = normalize_dims_shape(czi)
            n_scenes = int(dims.get("S", 1) or 1)
            if n_scenes <= 0:
                n_scenes = 1
            return [f"Region_{i+1}" for i in range(n_scenes)]
        finally:
            try:
                czi.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Open region
    # ------------------------------------------------------------------
    def open_region(self, *, input_dir: Path, region_index: int, region_name: str) -> Dict[str, Any]:
        CziFile = self._CziFile()

        czi_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".czi"])
        if not czi_files:
            raise ValueError("No .czi files found")

        if region_index < 0:
            raise IndexError("region_index must be >= 0")

        filepath = czi_files[0]
        czi = CziFile(str(filepath))
        try:
            dims = normalize_dims_shape(czi)
        except Exception:
            try:
                czi.close()
            except Exception:
                pass
            raise

        # Best-effort objective magnification + pixel size (µm/px)
        try:
            mag, pixel_to_um_calc = czi_get_mag_and_pixel_to_um(czi)
        except Exception:
            mag, pixel_to_um_calc = None, None

        # Match LIF: keep a consistent ctx schema and keep file handle open
        ctx: Dict[str, Any] = dict(
            input_dir=input_dir,
            region_index=int(region_index),
            region=str(region_name),
            mode=self.mode,

            czi=czi,                  # kept open; closed in close_region()
            czi_dims=dims,
            czi_filepath=filepath,

            objective_mag=mag,
            objective_mag_source="CZI metadata" if mag is not None else None,
            pixel_to_um_calc=pixel_to_um_calc,
        )
        return ctx
        
    # ------------------------------------------------------------------
    # Get valid tiles
    # ------------------------------------------------------------------
    def get_valid_tiles(self, ctx: Dict[str, Any]) -> List[int]:
        """
        Return the tile indices that are valid *mosaic tiles* (have stage coords / bbox),
        suitable for downstream processing and expected output enumeration.
        """
        # If metadata parsing ran, ctx["mosaic_tiles_iter"] is the authoritative source
        tiles_iter = ctx.get("mosaic_tiles_iter", None)
        if tiles_iter:
            return sorted({int(t[0]) for t in tiles_iter})
    
        # If metadata parsing didn't run, fall back to inferred tiles
        tiles = ctx.get("tiles", None)
        if tiles:
            return sorted({int(t) for t in tiles})
    
        return []

    # ------------------------------------------------------------------
    # Infer tiles/channels/dims (NO mosaic parsing here)
    # ------------------------------------------------------------------
    def infer_tiles_channels(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        - Determine tiles / channels / size_z / image_dimensions
        - Do NOT parse or normalize mosaic positions here
          (that belongs in build_metadata_args)
    
        Updated behavior:
        - Prefer mosaic tile indices from czi_get_mosaic_positions() when available.
          This avoids admitting non-mosaic M planes (label/overview/preview) that can be
          readable but have no mosaic bbox/stage coords.
        - Fall back to the previous "probe read_image" method if mosaic positions
          cannot be obtained.
        """
        dims: Dict[str, int] = ctx["czi_dims"]
    
        size_z = int(dims.get("Z", 1) or 1)
    
        num_channels = int(dims.get("C", 1) or 1)
        if num_channels <= 0:
            num_channels = 1
        channels = list(range(num_channels))
    
        # CZI uses M for mosaic tile index in many datasets
        n_tiles = int(dims.get("M", 1) or 1)
        if n_tiles <= 0:
            n_tiles = 1
    
        x_px = int(dims.get("X", 1) or 1)
        y_px = int(dims.get("Y", 1) or 1)
        if x_px <= 0 or y_px <= 0:
            raise RuntimeError(
                f"CZI reports non-positive image dimensions X={x_px}, Y={y_px} "
                f"for region '{ctx.get('region')}'."
            )
        image_dimensions = (x_px, y_px)
    
        czi = ctx["czi"]
        tiles: List[int] = []
    
        # ------------------------------------------------------------
        # Prefer true mosaic tile indices if available
        # ------------------------------------------------------------
        if n_tiles <= 1:
            tiles = [0]
        else:
            try:
                scene_index = int(ctx.get("region_index", 0)) if "S" in (dims or {}) else None
                block_index = 0 if "B" in (dims or {}) else None
    
         
                import contextlib
                import io
                
                with contextlib.redirect_stdout(io.StringIO()):
                    out = czi_get_mosaic_positions(
                        czi,
                        n_tiles=n_tiles,
                        scene_index=scene_index,
                        block_index=block_index,
                    )

    
                raw_tiles_iter = None
                if isinstance(out, tuple) and len(out) >= 1:
                    raw_tiles_iter = list(out[0] or [])
    
                if raw_tiles_iter:
                    # raw tuples are expected to begin with TileIndex
                    mosaic_tile_ids = sorted({int(t[0]) for t in raw_tiles_iter if t and len(t) >= 1})
                    if mosaic_tile_ids:
                        tiles = mosaic_tile_ids
            except Exception:
                # If mosaic positions aren't available, fall back to probing planes
                tiles = []
    
            # ------------------------------------------------------------
            # Fallback: probe read_image to find readable tile ids
            # ------------------------------------------------------------
            if not tiles:
                for m in range(n_tiles):
                    try:
                        img, _ = czi.read_image(M=m, C=0, Z=0)
                        if img is not None:
                            tiles.append(int(m))
                    except Exception:
                        pass
                if not tiles:
                    tiles = list(range(n_tiles))
    
        # Update ctx in-place
        ctx.update(dict(
            tiles=tiles,
            channels=channels,
            size_z=size_z,
            image_dimensions=image_dimensions,
    
            mosaic_tiles_iter=None,
            mosaic_x_raw=None,
            mosaic_y_raw=None,
            mosaic_unit_hint_raw=None,
        ))
    
        return {
            "tiles": ctx["tiles"],
            "channels": ctx["channels"],
            "size_z": ctx["size_z"],
            "image_dimensions": ctx["image_dimensions"],
        }


    # ------------------------------------------------------------------
    # Read stack
    # ------------------------------------------------------------------
    def read_stack(self, ctx: Dict[str, Any], tile: int, channel: int) -> np.ndarray:
        """
        Read a (Z, Y, X) stack for one tile+channel from a CZI.
        """
        czi = ctx["czi"]
        dims = ctx["czi_dims"]

        zmax = int(dims.get("Z", 1) or 1)
        region_index = int(ctx.get("region_index", 0))

        tile = int(tile)
        channel = int(channel)

        z_planes: List[np.ndarray] = []
        for z in range(zmax):
            # Build read_image kwargs (filter to supported dims defensively)
            kwargs = {"Z": int(z)}

            if "S" in dims:
                kwargs["S"] = region_index
            if "M" in dims:
                kwargs["M"] = tile
            if "C" in dims:
                kwargs["C"] = channel
            if "B" in dims:
                kwargs["B"] = 0

            if hasattr(czi, "dims") and isinstance(czi.dims, str):
                valid_dims = set(czi.dims)
                kwargs = {k: v for k, v in kwargs.items() if k in valid_dims}

            try:
                img, _ = czi.read_image(**kwargs)
            except Exception as e:
                # Some datasets may not support S; retry without it if the error hints that
                msg = str(e)
                if "S value" in msg or "S=" in msg:
                    kwargs.pop("S", None)
                    img, _ = czi.read_image(**kwargs)
                else:
                    raise

            arr = np.asarray(img).squeeze()
            if arr.ndim != 2 and arr.size:
                # be defensive: reduce to (Y, X)
                arr = arr.reshape(arr.shape[-2:])

            z_planes.append(arr)

        if not z_planes:
            raise RuntimeError(
                f"CZI read_stack(): no Z planes read for tile={tile}, channel={channel} "
                f"(region_index={ctx.get('region_index')}, region='{ctx.get('region')}')."
            )

        return np.stack(z_planes, axis=0).astype(np.uint16, copy=False)

    # ------------------------------------------------------------------
    # Build args for decide_and_write_tilescan() (mosaic parsing lives here)
    # ------------------------------------------------------------------
    def build_metadata_args(
        self,
        ctx: Dict[str, Any],
        *,
        pixel_to_um_manual: Optional[float],
        deconvolution_method: Optional[str],
        num_iterations: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """
        Match LIF/TIFF handler behavior:

        - MUST NOT depend on ctx["out_xml_path"] (pipeline passes out_xml_path at call site)
        - Returns kwargs for decide_and_write_tilescan()
        - Provides STRICT 5-tuples: (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
        - TileIndex MUST match ctx["tiles"] (tile ids used on disk: `_s{tile}` in filenames)

        Unit handling
        -------------
        - We avoid hardcoding units.
        - If czi_get_mosaic_positions() can provide a unit hint, we pass it through.
        - Otherwise unit_hint_raw="unknown" and decide_and_write_tilescan() will:
            * use the unit hint if plausible, or
            * run a hypothesis test if pixel size is available.
        """
        region = ctx.get("region", "unknown")
        image_dimensions: Optional[Tuple[int, int]] = ctx.get("image_dimensions", None)
        tiles: List[int] = [int(t) for t in (ctx.get("tiles", []) or [])]

        if image_dimensions is None:
            print(f"[ERROR] CZI handler: image_dimensions missing for region '{region}'. Skipping metadata.")
            return None
        if not tiles:
            print(f"{BOLD}[WARN]⚠️ {RESET} CZI handler: tiles missing/empty for region '{region}'. Skipping metadata.")
            return None

        czi = ctx.get("czi", None)
        dims: Dict[str, int] = ctx.get("czi_dims", {}) or {}

        if czi is None:
            print(f"[ERROR] CZI handler: czi handle missing for region '{region}'. Skipping metadata.")
            return None

        # n_tiles is the nominal M dimension; helper may need it
        n_tiles = int(dims.get("M", 1) or 1)
        if n_tiles <= 0:
            n_tiles = max(1, len(tiles))

        # ------------------------------------------------------------------
        # Get mosaic positions from helper
        #
        # Expected helper outputs (preferred):
        #   raw_tiles_iter, x_raw, y_raw, unit_hint_raw = czi_get_mosaic_positions(...)
        #
        # Backward compatible (older helper):
        #   raw_tiles_iter, x_raw, y_raw = czi_get_mosaic_positions(...)
        # ------------------------------------------------------------------
        raw_tiles_iter = None
        x_raw = None
        y_raw = None
        unit_hint_raw = "unknown"

        try:
            scene_index = None
            if "S" in (dims or {}):
                scene_index = int(ctx.get("region_index", 0))
            
            # block index is usually 0 if present; keep optional
            block_index = 0 if "B" in (dims or {}) else None
            
            out = czi_get_mosaic_positions(
                czi,
                n_tiles=n_tiles,
                scene_index=scene_index,
                block_index=block_index,
            )


            # Allow both return signatures
            if isinstance(out, tuple) and len(out) == 4:
                raw_tiles_iter, x_raw, y_raw, unit_hint_raw = out
            elif isinstance(out, tuple) and len(out) == 3:
                raw_tiles_iter, x_raw, y_raw = out
                unit_hint_raw = "unknown"
            else:
                raise ValueError(f"Unexpected czi_get_mosaic_positions() return signature: {type(out)} / {out!r}")

            raw_tiles_iter = list(raw_tiles_iter or [])
        except Exception as e:
            print(f"{BOLD}[WARN]⚠️ {RESET} CZI handler: no usable mosaic positions for region '{region}': {e!r}. Skipping TileScanInfo.")
            return None

        if not raw_tiles_iter:
            print(f"{BOLD}[WARN]⚠️ {RESET} CZI handler: mosaic tile list empty for region '{region}'. Skipping TileScanInfo.")
            return None

        # ------------------------------------------------------------------
        # Normalize to STRICT 5-tuples
        # Contract required by decide_and_write_tilescan():
        #   (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
        # ------------------------------------------------------------------
        try:
            t0 = tuple(raw_tiles_iter[0])
            if len(t0) != 5:
                raise ValueError(
                    f"Expected CZI mosaic tuples of len=5 (TileIndex, FieldX, FieldY, PosX, PosY); "
                    f"got len={len(t0)}: {t0}"
                )

            tiles_iter_5 = [
                (int(t[0]), int(t[1]), int(t[2]), float(t[3]), float(t[4]))
                for t in raw_tiles_iter
            ]

            # Ensure x_raw/y_raw are arrays if provided by helper; otherwise derive from tuples
            if x_raw is None or y_raw is None:
                x_raw = np.asarray([t[3] for t in tiles_iter_5], dtype=float)
                y_raw = np.asarray([t[4] for t in tiles_iter_5], dtype=float)
            else:
                x_raw = np.asarray(x_raw, dtype=float)
                y_raw = np.asarray(y_raw, dtype=float)

        except Exception as e:
            print(f"[ERROR] CZI handler: failed to normalize mosaic positions for '{region}': {e!r}. Skipping.")
            return None

        if x_raw.size == 0 or y_raw.size == 0:
            print(f"[ERROR] CZI handler: parsed mosaic arrays empty for '{region}'. Skipping TileScanInfo.")
            return None

        # ------------------------------------------------------------------
        # Subset-safe filtering + deterministic ordering
        #
        # Match TIFF/LIF policy:
        # - Only keep tiles whose TileIndex appears in ctx["tiles"] (tiles we actually process/write)
        # - Keep ordering consistent with ctx["tiles"] so downstream per-tile outputs are aligned
        #
        # IMPORTANT CZI EDGE CASE
        # -----------------------
        # infer_tiles_channels() tests tiles via read_image(...), but stage coords come from
        # get_mosaic_tile_bounding_box(...). These can disagree:
        #   - tile readable but no bounding box  -> processed tile has NO coords
        #   - bounding box exists but tile unreadable -> coords exist for tile we won't process
        #
        # We warn if overlap is partial to avoid silently writing incomplete metadata.
        # ------------------------------------------------------------------
        tiles_set = set(int(t) for t in tiles)

        # Keep only entries for tiles we actually process
        tiles_iter_5 = [t for t in tiles_iter_5 if int(t[0]) in tiles_set]
        if not tiles_iter_5:
            print(
                f"{BOLD}[WARN]⚠️ {RESET} CZI handler: mosaic positions do not overlap processed tiles for '{region}'. "
                "Skipping TileScanInfo."
            )
            return None

        # ---- report partial overlap (processed tiles missing coords) ----
        present = set(int(t[0]) for t in tiles_iter_5)
        missing_tiles = sorted(tiles_set - present)
        extra_tiles = sorted(present - tiles_set)  # usually empty after filtering, but keep for sanity

        if missing_tiles:
            preview = missing_tiles[:20]
            print(
                f"{BOLD}[WARN]⚠️ {RESET} CZI handler: {len(missing_tiles)} processed tile(s) have no mosaic stage coords "
                f"and will be omitted from TileScanInfo for '{region}'. "
                f"Missing TileIndex: {preview}{'...' if len(missing_tiles) > 20 else ''}"
            )

        if extra_tiles:
            # This should not happen after filtering, but keep it defensive.
            preview = extra_tiles[:20]
            print(
                f"{BOLD}[WARN]⚠️ {RESET} CZI handler: {len(extra_tiles)} tile(s) have coords but are not in ctx['tiles'] "
                f"for '{region}'. Extra TileIndex: {preview}{'...' if len(extra_tiles) > 20 else ''}"
            )

        # Keep ordering consistent with ctx["tiles"] (not numeric sort of TileIndex)
        rank = {int(t): i for i, t in enumerate(tiles)}
        tiles_iter_5 = sorted(tiles_iter_5, key=lambda t: rank.get(int(t[0]), 10**12))

        # Re-derive x_raw/y_raw in the filtered/reordered sequence to guarantee alignment
        x_raw = np.asarray([t[3] for t in tiles_iter_5], dtype=float)
        y_raw = np.asarray([t[4] for t in tiles_iter_5], dtype=float)

        # Record in ctx (optional; mirrors LIF/TIFF storing "mosaic" state)
        ctx["mosaic_tiles_iter"] = tiles_iter_5
        ctx["mosaic_x_raw"] = x_raw
        ctx["mosaic_y_raw"] = y_raw
        ctx["mosaic_unit_hint_raw"] = unit_hint_raw

        return dict(
            x_raw=x_raw,
            y_raw=y_raw,
            image_dimensions=image_dimensions,
            pixel_to_um_manual=pixel_to_um_manual,
            pixel_to_um_calc=ctx.get("pixel_to_um_calc", None),

            unit_hint_raw=str(unit_hint_raw or "unknown"),
            off_tol=0.25,

            tiles_iter=tiles_iter_5,
            app_name="Zeiss CZI",

            # NOTE: out_xml_path intentionally NOT included (pipeline passes it)
            deconvolution_method=deconvolution_method,
            deconvolution_iterations=num_iterations,
            objective_mag=ctx.get("objective_mag", None),
            objective_mag_source=ctx.get("objective_mag_source", None),
        )

    # ------------------------------------------------------------------
    # Close region
    # ------------------------------------------------------------------
    def close_region(self, ctx: Dict[str, Any]) -> None:
        czi = ctx.get("czi", None)
        try:
            if czi is not None:
                czi.close()
        except Exception:
            pass
        finally:
            ctx["czi"] = None
            ctx["czi_dims"] = None
            ctx["czi_filepath"] = None


# ----------------------------
# ND2 handler
# ----------------------------
class Nd2Handler(BaseHandler):
    mode = "nd2"

    @staticmethod
    def _nd2():
        """
        Lazy import to avoid killing the Jupyter kernel at module import time.
        Only imports nd2 when ND2 mode is actually used.
        """
        try:
            import nd2
        except Exception as e:
            raise ImportError(
                "ND2 mode requires the 'nd2' package. Install it to use mode='nd2'."
            ) from e
        return nd2

    # -----------------------------
    # Region discovery
    # -----------------------------
    def discover_regions(self, input_dir: Path) -> List[str]:
        nd2_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".nd2"])
        if not nd2_files:
            return []
        # Policy: ND2 regions are FILES (one region per .nd2), never the internal P dimension.
        return [f"Region_{i+1}" for i in range(len(nd2_files))]

    # -----------------------------
    # Open region (one ND2 file)
    # -----------------------------
    def open_region(self, *, input_dir: Path, region_index: int, region_name: str) -> Dict[str, Any]:
        nd2 = self._nd2()

        nd2_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".nd2"])
        if not nd2_files:
            raise ValueError("No .nd2 files found")

        if region_index < 0 or region_index >= len(nd2_files):
            raise IndexError(f"region_index {region_index} out of range for {len(nd2_files)} nd2 file(s)")

        filepath = nd2_files[int(region_index)]

        f = nd2.ND2File(str(filepath))
        try:
            darr = f.to_dask(copy=False)  # lazy
            sizes = dict(f.sizes)
        except Exception:
            try:
                f.close()
            except Exception:
                pass
            raise

        ctx: Dict[str, Any] = dict(
            input_dir=input_dir,
            region_index=int(region_index),
            region=str(region_name),
            mode=self.mode,

            nd2_file=f,          # kept open; closed in close_region()
            nd2_darr=darr,
            nd2_sizes=sizes,
            nd2_filepath=filepath,
        )

        # Best-effort objective magnification + pixel size (µm/px) from ND2 metadata (NO PRINTS here).
        try:
            mag, pixel_to_um_calc = nd2_get_mag_and_pixel_to_um(f)   
        except Exception:
            mag, pixel_to_um_calc = None, None
        

        ctx["objective_mag"] = mag
        ctx["objective_mag_source"] = "Nikon ND2 metadata" if mag is not None else None
        ctx["pixel_to_um_calc"] = pixel_to_um_calc
        ctx["nd2_pixel_to_um_calc"] = pixel_to_um_calc  # backwards-compat alias

        return ctx

    # -----------------------------
    # Infer tiles/channels/dims + stage positions
    # -----------------------------
    def infer_tiles_channels(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        sizes: Dict[str, int] = ctx["nd2_sizes"]

        size_z = int(sizes.get("Z", 1) or 1)

        num_channels = int(sizes.get("C", 1) or 1)
        if num_channels <= 0:
            num_channels = 1
        channels = list(range(num_channels))

        x_px = int(sizes.get("X", 1) or 1)
        y_px = int(sizes.get("Y", 1) or 1)
        if x_px <= 0 or y_px <= 0:
            raise RuntimeError(
                f"ND2 reports non-positive image dimensions X={x_px}, Y={y_px} for region '{ctx.get('region')}'."
            )
        image_dimensions = (x_px, y_px)

        # Tiles: prefer P (position loop) else M else single
        p_n = int(sizes.get("P", 1) or 1)
        m_n = int(sizes.get("M", 1) or 1)

        if p_n > 1:
            tiles = list(range(p_n))
            tile_dim = "P"
        elif m_n > 1:
            tiles = list(range(m_n))
            tile_dim = "M"
        else:
            tiles = [0]
            tile_dim = None

        # ---- stage coords extraction (for TileScanInfo writer) ----
        # ND2 stage positions from nd2 are typically already in microns (stagePositionUm).
        #
        # IMPORTANT DESIGN RULE
        # ---------------------
        # We keep ALL the extraction logic in one place (nd2_get_stage_positions_um),
        # which is a NO-PRINT helper by design. This prevents drift between:
        #   - infer_tiles_channels() structure inference, and
        #   - build_metadata_args() / TileScanInfo writing.
        #
        # infer_tiles_channels() is allowed to emit ONE warning if extraction fails,
        # but should not re-implement the metadata parsing itself.
        f = ctx["nd2_file"]
        coords: Optional[List[Tuple[float, float]]] = None

        try:
            coords = nd2_get_stage_positions_um(f)  # NO-PRINT helper
        except Exception:
            coords = None

        if coords is None:
            print(
                f"{BOLD}[WARN]⚠️ {RESET} ND2 handler: no stage coordinates available for region "
                f"'{ctx.get('region')}'. TileScanInfo will be skipped."
            )

        # ------------------------------------------------------------------
        # Validate stage coordinate count vs tile count (P-dimension only)
        #
        # ND2 assumptions:
        # - XYPosLoop points correspond 1:1 with P tiles
        # - coords[i] corresponds to tile i (TileIndex / filename `_s{i}`)
        #
        # If this assumption breaks, we WARN (once) and allow the pipeline
        # to continue; metadata writing will decide how to handle it.
        # ------------------------------------------------------------------
        if tile_dim == "P" and coords is not None:
            n_coords = len(coords)
            n_tiles = len(tiles)

            if n_coords != n_tiles:
                preview_coords = coords[:5]
                preview_tiles = tiles[:5]

                print(
                    f"{BOLD}[WARN]⚠️ {RESET} ND2 handler: stage coordinate count ({n_coords}) does not match "
                    f"tile count ({n_tiles}) for region '{ctx.get('region')}'. "
                    f"TileScanInfo may be incomplete or misaligned.\n"
                    f"        tiles preview: {preview_tiles}\n"
                    f"        coords preview: {preview_coords}"
                )

        ctx.update(dict(
            tiles=tiles,
            channels=channels,
            size_z=size_z,
            image_dimensions=image_dimensions,

            nd2_coords=coords,      # list of (x_um, y_um) or None
            nd2_tile_dim=tile_dim,  # "P" | "M" | None
        ))
        
        return {
            "tiles": ctx["tiles"],
            "channels": ctx["channels"],
            "size_z": ctx["size_z"],
            "image_dimensions": ctx["image_dimensions"],
        }


    # -----------------------------
    # Read one tile+channel stack (Z,Y,X)
    # -----------------------------
    def read_stack(self, ctx: Dict[str, Any], tile: int, channel: int) -> np.ndarray:
        """
        Read a (Z, Y, X) stack for one tile+channel from ND2.
    
        Supports tile dimension P or M if present; otherwise uses single tile.
    
        WHY THIS FUNCTION EXISTS (ND2 gotcha)
        ------------------------------------
        `nd2.ND2File.to_dask()` does NOT always return an xarray.DataArray.
        In some files it returns a ResourceBackedDaskArray (or similar) that:
          - has no .isel()
          - has no .dims()
    
        If we naively do `np.asarray(darr)` in that situation, we materialize the ENTIRE
        dataset (P×Z×C×Y×X) and appear to "hang" during reading.
    
        This function therefore:
          1) Tries named-dimension selection via .isel() when available (xarray path).
          2) Otherwise slices FIRST using positional indexing on the dask-like array,
             THEN materializes only the requested slab (fallback path).
          3) Normalizes output to exactly (Z, Y, X) and uint16.
    
        INFO PRINT POLICY
        -----------------
        - Emit concise [INFO] lines only when we take the fallback (non-xarray) path.
          That’s where "silent hangs" used to happen and where provenance matters.
        - Do not spam per-plane debug; just one or two lines per tile/channel read.
        """
        darr = ctx["nd2_darr"]
        sizes: Dict[str, int] = ctx["nd2_sizes"]
        tile_dim = ctx.get("nd2_tile_dim", None)
    
        tile = int(tile)
        channel = int(channel)
    
        # Z length (best-effort; ND2 may omit Z in some cases)
        size_z = int(sizes.get("Z", 1) or 1)
    
        # Helper: safe isel on DataArray
        def _isel(obj, **kwargs):
            if hasattr(obj, "isel"):
                return obj.isel(**kwargs)
            raise AttributeError("Object has no .isel()")
    
        try:
            # =====================================================================
            # PATH A: xarray.DataArray selection (named dims)
            # =====================================================================
            # This is the "ideal" case: named-dim indexing is robust to axis ordering.
            sel = {"C": channel}
    
            # Include full Z stack if present
            if "Z" in sizes and size_z > 1:
                sel["Z"] = slice(0, size_z)
    
            # Include tile index if ND2 exposes a tile dimension (P or M)
            if tile_dim in ("P", "M") and tile_dim in sizes:
                sel[tile_dim] = tile
    
            sub = _isel(darr, **sel)
    
            # NOTE: We keep this behavior unchanged: np.asarray(sub) may trigger upstream
            # computation depending on backend, but for xarray this is typically fine.
            arr = np.asarray(sub)
    
        except Exception:
            # =====================================================================
            # PATH B: positional indexing fallback (ResourceBackedDaskArray, etc.)
            # =====================================================================
            # This is the critical fix vs your previous implementation:
            # - Do NOT do np.asarray(darr) (that materializes everything).
            # - Slice FIRST to (Z,Y,X) for the requested tile+channel, then compute.
            shape = tuple(getattr(darr, "shape", ()))
            if len(shape) < 3:
                raise RuntimeError(f"ND2 read_stack(): unexpected array shape from to_dask(); shape={shape}")
    
            # Pull out reported sizes (may be 1 even if the dim isn't present)
            p_n = int(sizes.get("P", 1) or 1)
            m_n = int(sizes.get("M", 1) or 1)
            z_n = int(sizes.get("Z", 1) or 1)
            c_n = int(sizes.get("C", 1) or 1)
            y_n = int(sizes.get("Y", 1) or 1)
            x_n = int(sizes.get("X", 1) or 1)
    
            # Emit ONE helpful info line: what we're reading and from what backing type/shape.
            print(
                f"[INFO] ND2 read_stack(): using fallback positional slicing "
                f"(darr_type={type(darr).__name__}, shape={shape}) "
                f"for tile={tile}, channel={channel}"
            )
    
            # -----------------------------
            # Strong-match fast path
            # -----------------------------
            # Common ND2 mosaic layout observed in your files:
            #   (P, Z, C, Y, X)
            # If it matches exactly, slice deterministically (fast + correct).
            if len(shape) == 5 and shape == (p_n, z_n, c_n, y_n, x_n):
                # Select just the requested tile+channel stack: -> (Z, Y, X)
                darr_sel = darr[tile, slice(0, z_n), channel, :, :]
    
                # Compute only this slab (never the entire dataset).
                if hasattr(darr_sel, "compute"):
                    arr = np.asarray(darr_sel.compute())
                else:
                    arr = np.asarray(darr_sel)
    
            else:
                # -----------------------------
                # Generic conservative fallback
                # -----------------------------
                # When axis order differs, do best-effort axis identification by matching sizes:
                # - Prefer Y/X as the last two axes if they match reported Y/X
                # - Find C and Z axes by matching c_n and z_n (avoiding Y/X)
                # - Find tile axis by matching P or M size (depending on tile_dim)
                sl = [slice(None)] * len(shape)
    
                # Prefer Y/X at the end when they match (common ND2)
                ax_y = ax_x = None
                if len(shape) >= 2 and shape[-2] == y_n and shape[-1] == x_n:
                    ax_y, ax_x = len(shape) - 2, len(shape) - 1
    
                # Find channel axis by matching c_n (avoid Y/X)
                ax_c = None
                if c_n > 1:
                    for ax in range(len(shape)):
                        if ax in (ax_y, ax_x):
                            continue
                        if shape[ax] == c_n:
                            ax_c = ax
                            break
    
                # Find Z axis by matching z_n (avoid Y/X and C)
                ax_z = None
                if z_n > 1:
                    for ax in range(len(shape)):
                        if ax in (ax_y, ax_x, ax_c):
                            continue
                        if shape[ax] == z_n:
                            ax_z = ax
                            break
    
                # Find tile axis by matching P or M
                ax_tile = None
                if tile_dim == "P" and p_n > 1:
                    for ax in range(len(shape)):
                        if ax in (ax_y, ax_x, ax_c, ax_z):
                            continue
                        if shape[ax] == p_n:
                            ax_tile = ax
                            break
                elif tile_dim == "M" and m_n > 1:
                    for ax in range(len(shape)):
                        if ax in (ax_y, ax_x, ax_c, ax_z):
                            continue
                        if shape[ax] == m_n:
                            ax_tile = ax
                            break
    
                # Apply slice selections we managed to identify
                if ax_tile is not None:
                    sl[ax_tile] = tile
                if ax_c is not None:
                    sl[ax_c] = channel
                if ax_z is not None:
                    sl[ax_z] = slice(0, z_n)
    
                darr_sel = darr[tuple(sl)]
    
                # Compute only the selected slab.
                if hasattr(darr_sel, "compute"):
                    arr = np.asarray(darr_sel.compute())
                else:
                    arr = np.asarray(darr_sel)
    
        # =====================================================================
        # Normalize output to (Z, Y, X)
        # =====================================================================
        arr = np.asarray(arr)
    
        # After slicing, we expect:
        # - (Z, Y, X) normally
        # - (Y, X) if Z==1 (then add a singleton Z axis)
        # - potentially extra singleton axes depending on backend (squeeze them)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim > 3:
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim != 3:
                raise RuntimeError(f"ND2 read_stack(): could not normalize to (Z,Y,X); got shape {arr.shape}")
    
        return arr.astype(np.uint16, copy=False)

    
    # -----------------------------
    # Build args for decide_and_write_tilescan()
    # -----------------------------
    def build_metadata_args(
        self,
        ctx: Dict[str, Any],
        *,
        pixel_to_um_manual: Optional[float],
        deconvolution_method: Optional[str],
        num_iterations: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """
        Return kwargs for decide_and_write_tilescan().

        Matches LIF/TIFF handler policy:
        - MUST NOT require ctx["out_xml_path"] (pipeline passes out_xml_path explicitly)
        - tiles_iter MUST be STRICT 5-tuples: (TileIndex, FieldX, FieldY, PosX_raw, PosY_raw)
        - TileIndex MUST match on-disk tile ids used in filenames (`_s{tile}`)
        """
        region = ctx.get("region", "unknown")
        image_dimensions: Optional[Tuple[int, int]] = ctx.get("image_dimensions", None)
        if image_dimensions is None:
            print(f"[ERROR] ND2 handler: image_dimensions missing for region '{region}'. Skipping metadata.")
            return None

        coords = ctx.get("nd2_coords", None)
        tiles = list(ctx.get("tiles", []) or [])

        if not tiles:
            print(f"{BOLD}[WARN]⚠️ {RESET} ND2 handler: no tiles inferred for region '{region}'. Skipping TileScanInfo.")
            return None

        if coords is None:
            print(f"{BOLD}[WARN]⚠️ {RESET} ND2 handler: no stage coordinates available for region '{region}'. Skipping TileScanInfo.")
            return None

        if not isinstance(coords, (list, tuple)) or len(coords) == 0:
            print(f"{BOLD}[WARN]⚠️ {RESET} ND2 handler: empty stage coordinate list for region '{region}'. Skipping TileScanInfo.")
            return None

        # ND2 coords are stagePositionUm → microns.
        unit_hint_raw = "microns"

        # Build STRICT tile records aligned to tile ids.
        #
        # NOTE ABOUT FieldX / FieldY FOR ND2
        # ---------------------------------
        # We do not rely on FieldX/FieldY anywhere downstream (OME uses TileIndex + PosX/PosY).
        # For now we set:
        #   FieldX = TileIndex
        #   FieldY = 0
        # If ND2 grid indices are needed later, compute them explicitly from coords
        # (or remove FieldX/FieldY from the schema/pipeline to avoid false assumptions).
        n = min(len(tiles), len(coords))
        if len(coords) != len(tiles):
            print(
                f"{BOLD}[WARN]⚠️ {RESET} ND2 handler: coords count ({len(coords)}) != tiles count ({len(tiles)}); "
                f"using first {n} entries."
            )

        tiles_iter: List[Tuple[int, int, int, float, float]] = []
        x_list: List[float] = []
        y_list: List[float] = []

        for i in range(n):
            tid = int(tiles[i])  # must match on-disk tile id
            try:
                x_um = float(coords[i][0])
                y_um = float(coords[i][1])
            except Exception:
                continue

            tiles_iter.append((tid, tid, 0, x_um, y_um))
            x_list.append(x_um)
            y_list.append(y_um)

        if not tiles_iter:
            print(f"{BOLD}[WARN]⚠️ {RESET} ND2 handler: could not build any tile position records for '{region}'. Skipping.")
            return None

        x_raw = np.asarray(x_list, dtype=float)
        y_raw = np.asarray(y_list, dtype=float)
        if x_raw.size == 0 or y_raw.size == 0:
            print(f"{BOLD}[WARN]⚠️ {RESET} ND2 handler: empty x/y arrays after parsing for '{region}'. Skipping.")
            return None

        return dict(
            x_raw=x_raw,
            y_raw=y_raw,
            image_dimensions=image_dimensions,
            pixel_to_um_manual=pixel_to_um_manual,
            pixel_to_um_calc=ctx.get("pixel_to_um_calc", None),
            unit_hint_raw=unit_hint_raw,
            off_tol=0.25,
            tiles_iter=tiles_iter,
            app_name="NIS-Elements",  # triggers FlipX logic in writer if you keep that behavior
            # NOTE: out_xml_path intentionally NOT included (pipeline passes it)
            deconvolution_method=deconvolution_method,
            deconvolution_iterations=num_iterations,
            objective_mag=ctx.get("objective_mag", None),
            objective_mag_source=ctx.get("objective_mag_source", None),
        )

    # -----------------------------
    # Close region (release ND2 handle)
    # -----------------------------
    def close_region(self, ctx: Dict[str, Any]) -> None:
        f = ctx.get("nd2_file", None)
        try:
            if f is not None:
                f.close()
        except Exception:
            pass
        finally:
            ctx["nd2_file"] = None
            ctx["nd2_darr"] = None
            ctx["nd2_sizes"] = None
            ctx["nd2_filepath"] = None
            ctx["nd2_coords"] = None
            ctx["nd2_tile_dim"] = None

# Get handler 

def get_handler(mode: str) -> BaseHandler:
    """
    Factory for dataset handlers.

    Parameters
    ----------
    mode : str
        Dataset mode / format identifier, e.g.:
          - "tif_autosaved"
          - "tif_exported"
          - "lif"
          - "czi"
          - "nd2"

    Returns
    -------
    BaseHandler
        An initialized handler instance appropriate for the dataset type.
    """
    m = str(mode).strip().lower()

    if m in ("tif_autosaved", "tif_exported"):
        # TIFF handler needs to know WHICH Leica naming convention is used
        return TiffHandler(mode=m)

    if m == "lif":
        return LifHandler()

    if m == "czi":
        return CziHandler()

    if m == "nd2":
        return Nd2Handler()

    raise ValueError(f"Unsupported mode: {mode!r}")


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
                            num_iterations = 25,
                            PSF_metadata=None, 
                            align_channel=4, 
                            mip=True,
                            tile_dimension=6000, 
                            pixel_to_um = None,
                            chunk_size=None):
    
    """
    Main preprocessing pipeline for microscopy image data.

    This function processes microscopy image datasets stored in various formats/modes
    (autosaved/exported TIFF, LIF, ND2, CZI) and performs:

      1) Deconvolution (optional) and Maximum Intensity Projection (MIP) or stack export
      2) Writing TileScanInfo metadata XML (per region/cycle) when available
      3) Conversion of per-tile MIPs to per-cycle OME-TIFFs
      4) Cross-cycle alignment + stitching (Ashlar)
      5) Retiling stitched mosaics into fixed-size tiles

    Parameters
    ----------
    input_dirs : Sequence[Union[str, Path]]
        One input directory per cycle, containing the raw files for that cycle.
        Must be the same length/order as `cycles`.

    cycles : Sequence[Union[int, str]]
        Cycle identifiers corresponding to `input_dirs` (e.g., [1, 2, 3] or ["1","2","3"]).
        These are used in output folder/file naming (e.g., Cycle{cycle}).

    output_dir_prefix : Union[str, Path]
        Base output directory. Region folders are created under this prefix (e.g., R1, R2, ...).

    mode : str
        Input format/mode. Supported values:
          - 'tif_autosaved' : TIFF files autosaved by Leica software
          - 'tif_exported'  : TIFF files exported manually
          - 'lif'           : Leica Image File (LIF)
          - 'nd2'           : Nikon ND2
          - 'czi'           : Zeiss CZI

    n_total_cycles : int
        Total number of cycles expected for the experiment.
        Used as a safety check before running alignment/stitching to ensure all cycles exist.

    regions_to_process : Optional[Sequence[int]]
        Optional subset of regions to process, using 1-based region numbers (user-facing).
        Example: [1, 3, 4] processes R1, R3, R4.

    deconvolution_method : Optional[str]
        Deconvolution algorithm to use:
          - 'deconwolf'
          - 'redlionfish'
          - None (skip deconvolution)

    num_iterations : int, optional
        Number of iterations for the selected deconvolution method. Default is 25.

    PSF_metadata : Optional[dict]
        Metadata required to generate the PSF for deconvolution.
        Required if `deconvolution_method` is not None.

    align_channel : int, optional
        Channel index used for alignment in Ashlar. Default is 4.

    mip : bool, optional
        If True, save Maximum Intensity Projections (2D) per tile/channel.
        If False, save full Z-stacks per tile/channel. Default is True.

    tile_dimension : int, optional
        Tile size (pixels) used when retiling stitched mosaics. Default is 6000.

    pixel_to_um : Optional[float]
        Pixel size in microns per pixel (µm/px). If provided, it is used where metadata-derived
        pixel size is unavailable (and may also be written into metadata outputs depending on handlers).

    chunk_size : Optional[int]
        Chunk/tile size used for chunked processing in deconvolution backends that support it
        (e.g., Deconwolf tiling). Default is None.

    Raises
    ------
    ValueError
        If `mode` is unsupported, `deconvolution_method` is invalid, or PSF metadata is missing
        when deconvolution is requested.

    Notes
    -----
    - This function orchestrates the pipeline and delegates work to:
        `deconvolve_and_mip()`, `mipped_to_OME_tiffs()`, `align_and_stitch()`,
        and `retile_stitched_images()`.
    - Outputs are organized as:
        {output_dir_prefix}/R{region_number}/preprocessing/Cycle{cycle}/...
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
    num_iterations = 25,
    PSF_metadata: Optional[dict] = None, 
    mip: bool = True,
    pixel_to_um = None,
    chunk_size: Optional[int] = None
) -> list:
    """
    Deconvolve (optional) and write MIP/stack outputs per region and cycle.

    This function loops over (cycle, input_dir) pairs and, for each selected region:
      - Opens the dataset via the mode-specific handler
      - Infers tiles/channels and image dimensions
      - Extracts + writes TileScanInfo metadata XML when available
      - Processes only missing tile×channel outputs:
          * reads the Z-stack
          * optionally deconvolves (redlionfish / deconwolf)
          * writes either a Maximum Intensity Projection (MIP) or the full stack

    Parameters
    ----------
    input_dirs : Sequence[Union[str, Path]]
        One input directory per cycle, containing the raw files for that cycle.
        Must be the same length/order as `cycles`.
    cycles : Sequence[Union[int, str]]
        Cycle identifiers corresponding to `input_dirs` (e.g., [1, 2, 3]).
    output_dir_prefix : Union[str, Path]
        Base output directory. Region folders are created under this prefix (e.g., R1, R2, ...).
    mode : str
        Input format/mode. Supported values:
          - 'tif_autosaved' : Leica autosaved TIFF
          - 'tif_exported'  : Leica exported TIFF
          - 'lif'           : Leica Image File (LIF)
          - 'nd2'           : Nikon ND2
          - 'czi'           : Zeiss CZI
    regions_to_process : Optional[Sequence[int]]
        Optional subset of regions to process, using 1-based region numbers (user-facing).
    deconvolution_method : Optional[str]
        Deconvolution algorithm to use:
          - 'deconwolf'
          - 'redlionfish'
          - None (skip deconvolution)
    num_iterations : int, optional
        Number of iterations for the selected deconvolution method. Default is 25.
    PSF_metadata : Optional[dict]
        Metadata required to generate the PSF for deconvolution.
        Required if `deconvolution_method` is not None.
    mip : bool, optional
        If True, save Maximum Intensity Projections (2D) per tile/channel.
        If False, save full Z-stacks per tile/channel. Default is True.
    pixel_to_um : Optional[float]
        Manual pixel size in microns per pixel (µm/px). Used when metadata-derived pixel size
        is unavailable (and may also be written into TileScanInfo depending on the handler).
    chunk_size : Optional[int]
        Chunk/tile size used for chunked processing in deconvolution backends that support it
        (e.g., Deconwolf tiling). Default is None.

    Returns
    -------
    List[str]
        List of region directory paths as strings (e.g., [".../R1", ".../R2", ...]).
        These region directories are used downstream by OME-TIFF conversion, alignment, and retiling.
    """
    print(f"\033[1;96mDeconvolution and mipping\033[0m")
    
    valid_modes = {'tif_autosaved', 'tif_exported', 'lif', 'nd2', 'czi'}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}. Choose from {valid_modes}.")

    valid_methods = {'deconwolf', 'redlionfish', None}
    if deconvolution_method not in valid_methods:
        raise ValueError(f"Unsupported deconvolution method: {deconvolution_method}. Choose from {valid_methods - {None}} or None.")

    # ======================================================================
    # (0) build handler ONCE
    # ======================================================================
    handler = get_handler(mode)

    if len(cycles) != len(input_dirs):
        raise ValueError(f"len(cycles)={len(cycles)} must match len(input_dirs)={len(input_dirs)}")

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
    
        # STEP 1: detect regions via handler 
        
        all_regions = handler.discover_regions(input_dir)   # full list, in dataset order
        num_regions = len(all_regions)
        
        # Build (dataset_index, region_name, region_number)
        # region_number is what the USER sees (1-based)
        region_items = [
            (idx, name, idx + 1)
            for idx, name in enumerate(all_regions)
        ]

        # Apply user selection (regions_to_process is 1-based, user-facing)
        if regions_to_process is not None:
            ordered = []
            for rnum in regions_to_process:
                rnum = int(rnum)
                if 1 <= rnum <= num_regions:
                    idx = rnum - 1
                    ordered.append((idx, all_regions[idx], rnum))
                else:
                    print(f"{BOLD}[WARN]⚠️ {RESET} regions_to_process contains out-of-range region {rnum}; skipping.")
            region_items = ordered

        if not region_items:
            print(f"{BOLD}[WARN]⚠️ {RESET} No regions selected; skipping this cycle.")
            continue

        
        print("Regions to be processed:", [name for _, name, _ in region_items])
        print("=" * width + "\033[0m")
        
        region_directories = []
        
        # Process each region
        for region_index, region_name, region_number in region_items:
            print(f"\033[1;90mProcessing R{region_number}\033[0m")
        
            # IMPORTANT:
            # - region_index → used for opening data
            # - region_number → used for output naming
        
            region_directory = output_dir_prefix / f"R{region_number}"
            region_directories.append(str(region_directory))
            safe_mkdir(region_directory)
        
            cycle_directory = region_directory / "preprocessing" / f"Cycle{cycle}"
            safe_mkdir(cycle_directory)
        
            mipped_directory = cycle_directory / "1_mipped"
            safe_mkdir(mipped_directory)
        
            stacked_directory = cycle_directory / "1_stacked"
            metadata_directory = cycle_directory / "MetaData"
            safe_mkdir(metadata_directory)

    
            # ----- STEP 2: WRITE TileScanInfo via handler -----
            print("\033[96mExtracting metadata\033[0m")

            ctx = None
            # Open region using TRUE dataset index
            ctx = handler.open_region(
                input_dir=input_dir,
                region_index=region_index,   # ← original index
                region_name=region_name,
            )

            try: 
                inf = handler.infer_tiles_channels(ctx)
                tiles = inf["tiles"]
                channels = inf["channels"]
                size_z = inf["size_z"]
                image_dimensions = inf["image_dimensions"]
    
                # Build metadata args
                metadata_args = handler.build_metadata_args(
                    ctx,
                    pixel_to_um_manual=pixel_to_um,
                    deconvolution_method=deconvolution_method,
                    num_iterations=num_iterations,
                )
    
                
                if metadata_args:
                    decide_and_write_tilescan(
                        **metadata_args,
                        out_xml_path=metadata_directory / f"R{region_number}.xml",
                    )

                # ----- Determine valid mosaic tiles (format-agnostic) -----
                tiles_inferred = sorted({int(t) for t in tiles})
                
                # Preferred: handler provides valid tiles (e.g., CZI bbox-aware)
                valid_tiles = None
                if hasattr(handler, "get_valid_tiles"):
                    try:
                        valid_tiles = handler.get_valid_tiles(ctx)
                    except Exception:
                        valid_tiles = None
                
                # Fallback: assume all inferred tiles are valid
                if not valid_tiles:
                    valid_tiles = tiles_inferred
                else:
                    valid_tiles = sorted({int(t) for t in valid_tiles})
                
                # Sanity: never allow empty valid_tiles
                if not valid_tiles:
                    print(
                        f"{BOLD}[WARN]⚠️ {RESET} No valid mosaic tiles found; "
                        f"falling back to inferred tiles list."
                    )
                    valid_tiles = tiles_inferred
                
                # Skipped = inferred tiles that are not valid
                skipped = sorted(set(tiles_inferred) - set(valid_tiles))
                
                # Total tiles: use CZI-provided M only if present; otherwise inferred count
                total_tiles = len(tiles_inferred)
                if isinstance(ctx, dict):
                    czi_dims = ctx.get("czi_dims")
                    if isinstance(czi_dims, dict):
                        m = czi_dims.get("M")
                        if m is not None:
                            try:
                                m_int = int(m)
                                if m_int > 0:
                                    total_tiles = m_int
                            except Exception:
                                pass
                
                n_valid = len(valid_tiles)
                n_skipped = len(skipped)
                
                min_tile = valid_tiles[0] if valid_tiles else None
                max_tile = valid_tiles[-1] if valid_tiles else None
                
                print(
                    f"[R{region_number}] Tiles: {n_valid} valid / {total_tiles} total "
                    f"(min={min_tile}, max={max_tile})"
                    + (f" ({n_skipped} skipped: no bbox)" if n_skipped else "")
                )
                
                if skipped:
                    print(
                        f"{BOLD}[WARN]⚠️ {RESET} Skipping {len(skipped)} non-mosaic tile(s) "
                        f"(no bbox/stage coords): {skipped[:20]}{'...' if len(skipped) > 20 else ''}"
                    )


    
                # ----- STEP 3: SKIP EXISTING FILES -----
                print("\033[96mProcessing files\033[0m")
                
                out_dir = (mipped_directory if mip else stacked_directory)
                
                # Count only VALID outputs (avoid getting stuck because a corrupted/empty file exists)
                valid_existing = set()
                for f in out_dir.glob(f"Cycle{cycle}_s*_ch*.tif"):
                    m = re.search(r"_s0*(\d+)_ch0*(\d+)", f.name, re.IGNORECASE)
                    if m and file_exists_and_valid(f, min_size=1024):
                        valid_existing.add((int(m.group(1)), int(m.group(2))))
                
                # IMPORTANT:
                # tiles_all represents the FULL expected tile set for this region.
                # Do NOT reuse this variable for "tiles to process" later.
                # tiles_all should be the FULL expected *VALID MOSAIC* tile set for this region.
                tiles_all = list(valid_tiles)

                
                total_expected = len(tiles_all) * len(channels)
                total_valid = len(valid_existing)
                total_remaining = total_expected - total_valid
                
                print(
                    f"[INFO] Expected outputs this region/cycle: {total_expected} "
                    f"(tiles={len(tiles_all)} × channels={len(channels)}). "
                    f"Valid already present: {total_valid}. Remaining: {total_remaining}."
                )
                
                # Determine missing channels per tile (based on VALID outputs only)
                missing_channels_by_tile = {}
                for tile in tiles_all:
                    t = int(tile)
                    miss = [int(ch) for ch in channels if (t, int(ch)) not in valid_existing]
                    if miss:
                        missing_channels_by_tile[t] = miss
                
                if not missing_channels_by_tile:
                    print(
                        f"All expected files for Cycle {cycle} already exist "
                        f"(and look valid) in {out_dir}. Skipping processing."
                    )
                    continue
                
                tiles = sorted(missing_channels_by_tile.keys())
                print(f"{len(tiles)} tile(s) have missing outputs. Proceeding with missing tile-channel combos only.")


    
                # ----- STEP 4: GENERATE PSFS FOR ALL CHANNELS -----
                print("Calculating the PSF")
                
                psf_dict = {}
                
                if deconvolution_method is None:
                    print("Skipping PSF generation — deconvolution method is None.")
                    psf_dict = {}  # keep for downstream compatibility
                
                elif deconvolution_method == "redlionfish":

                    # select gpu and import redlionfish only when needed
                    gpu_id = choose_gpu_for_rl()

                    import RedLionfishDeconv as rl

                    
                    if PSF_metadata is None:
                        raise ValueError("PSF_metadata is required for redlionfish deconvolution.")
                
                    # For now: generate a tile-sized PSF (same XY as the image tile).
                    # NOTE: this can be heavy; later you can replace this with a smaller psf_xy (e.g. 256).
                    psf_x = int(image_dimensions[0])
                    psf_y = int(image_dimensions[1])
                    print(f"[INFO] Generating RL PSF at tile size: {psf_x}×{psf_y} px")
                
                    # Optional: warn if PSF magnification doesn't match metadata magnification
                    # (only if you have objective_mag in ctx and PSF_metadata has 'm')
                    try:
                        meta_mag = ctx.get("objective_mag", None)
                        psf_mag = float(PSF_metadata.get("m")) if PSF_metadata.get("m") is not None else None
                        if meta_mag is not None and psf_mag is not None and not np.isclose(meta_mag, psf_mag, rtol=0.02):
                            print(
                                f"{BOLD}[WARN]⚠️ {RESET} PSF magnification ({psf_mag:g}x) differs from metadata objective ({meta_mag:g}x) - check input parameter PSF_metadata!"
                            )
                    except Exception:
                        pass
                
                    for channel, info in PSF_metadata["channels"].items():
                        print(f"Generating PSF for channel {channel}")
                        psf_volume = fd_psf.GibsonLanni(
                            na=float(PSF_metadata["na"]),
                            m=float(PSF_metadata["m"]),
                            ni0=float(PSF_metadata["ni0"]),
                            res_lateral=float(PSF_metadata["res_lateral"]),
                            res_axial=float(PSF_metadata["res_axial"]),
                            wavelength=float(info["wavelength"]),
                            size_x=psf_x,
                            size_y=psf_y,
                            size_z=size_z,
                        ).generate()
                        psf_dict[str(channel)] = psf_volume
                
                elif deconvolution_method == "deconwolf":
                    if PSF_metadata is None:
                        raise ValueError("PSF_metadata is required for deconwolf deconvolution.")
                
                    psf_dir = cycle_directory / "PSF"
                    safe_mkdir(psf_dir)
                
                    for channel, info in PSF_metadata["channels"].items():
                        print(f"Generating PSF for channel {channel}")
                        wavelength_nm = float(info["wavelength"]) * 1000  # µm -> nm
                        psf_filename = psf_dir / f"PSF_channel_{channel}.tif"
                
                        generate_psf(
                            psf_output=psf_filename,
                            resxy=float(PSF_metadata["res_lateral"]) * 1000,
                            resz=float(PSF_metadata["res_axial"]) * 1000,
                            wavelength=wavelength_nm,
                            NA=float(PSF_metadata["na"]),
                            ni=float(PSF_metadata["ni0"]),
                        )
                        psf_dict[str(channel)] = psf_filename
                     
                # ----- STEP 5: DECONVOLVE EACH TILE AND CHANNEL -----
                # If you rebuilt `tiles = sorted(missing_channels_by_tile.keys())` earlier, update n_tiles:
                n_tiles = len(tiles)
    
                print("Single tile imaging." if n_tiles == 1 else f"Number of tiles to process: {n_tiles}")
        
                # Prepare directory to save stacked images
                safe_mkdir(stacked_directory)
        
                # Loop over each tile (spatial subdivision of the image)
                for tile in tqdm(tiles, desc="Processing tiles", leave=False):
                    tile = int(tile)
                
                    for channel in missing_channels_by_tile.get(tile, []):  # only missing channels (safe)
                        channel = int(channel)
                
                        print(f"\033[90m[\033[96mCycle {cycle}\033[90m] Tile {tile}, Channel {channel}...\033[0m")
                        tile_channel_start = time.time()
                
                        output_file_path = (mipped_directory if mip else stacked_directory) / f"Cycle{cycle}_s{tile}_ch{channel}.tif"
            
                        # Skip processing if output file already exists
                        if file_exists_and_valid(output_file_path, min_size=1024):
                            # (optional) keep your message if you want
                            # print(f"Valid output exists: {output_file_path}. Skipping.")
                            continue


                        try:
                            stacked_images = handler.read_stack(ctx, tile=tile, channel=channel)
                        except Exception as e:
                            msg = str(e)
                            if ("PixelType( Unknown type )" in msg) or ("PylibCZI_PixelTypeException" in msg):
                                print(f"{BOLD}[WARN]⚠️ {RESET} Skipping tile={tile}, ch={channel}: unsupported CZI pixel type.")
                                continue
                            raise

    
                        if stacked_images.ndim != 3:
                            raise ValueError(
                                f"Expected (Z,Y,X) stack, got shape {stacked_images.shape} "
                                f"for mode={mode}, tile={tile}, channel={channel}"
                            )
                        if stacked_images.dtype != np.uint16:
                            stacked_images = stacked_images.astype(np.uint16, copy=False)
    
            
                        # Deconvolution with RedLionFish method
                        if deconvolution_method == "redlionfish":
    
                            deconvolved_images = rl.doRLDeconvolutionFromNpArrays(stacked_images, psf_dict[str(channel)], niter=num_iterations)
                            # Save max projection if MIP requested, otherwise full stack
                            if mip:
                                processed_img = to_uint16_safe(
                                    np.max(deconvolved_images, axis=0),
                                    context=f"tile={tile} ch={channel}",
                                )
                            else:
                                processed_img = to_uint16_safe(
                                    deconvolved_images,
                                    context=f"tile={tile} ch={channel}",
                                )
                            tifffile.imwrite(output_file_path, processed_img)
                            print(f"{'Mipped' if mip else 'Stacked'} images saved in directory: {mipped_directory if mip else stacked_directory}")
                            
            
                        # Deconvolution with Deconwolf method
                        elif deconvolution_method == 'deconwolf':
                            # Create temporary directory for Deconwolf input
                            dw_input_directory = cycle_directory / 'deconwolf_input_tmp'
                            safe_mkdir(dw_input_directory)
                            
                            dw_input_path = dw_input_directory / f'Cycle{cycle}_s{tile}_ch{channel}.tif'
                            tifffile.imwrite(dw_input_path, stacked_images)    # Write input stack for Deconwolf
                            
                            dw_output_path = stacked_directory / f'Cycle{cycle}_s{tile}_ch{channel}.tif'
            
                            # Run Deconwolf deconvolution externally
                            deconvolve_image(
                                input_image=dw_input_path,
                                psf_image=psf_dict[str(channel)],
                                output_image=dw_output_path,
                                iterations=int(num_iterations),
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

            finally:
                # cleanup
                if ctx is not None:
                    handler.close_region(ctx)
    # NOTE: region_directories are identical across cycles (R1, R2, ...),
    print("\n🔹 region_directories:\n", region_directories)
    
    return region_directories




# ======================================================================================
# OME-TIFF conversion (reads TileScanInfo written above; assumes positions already µm)
# ======================================================================================

def mipped_to_OME_tiffs(
    region_directories: List[str],
    cycles: List[int],
    pixel_to_um: Optional[float] = None,
) -> None:
    """
    Convert per-tile MIPs to OME-TIFF with spatial Plane PositionX/PositionY (µm).

    Key behaviors
    -------------
    - Reads TileScanInfo XML written earlier (positions already in µm).
    - Writes one OME-TIFF per (region, cycle): Cycle{cycle}.ome.tiff
    - **No guessing** tile ordering:
        1) Prefer TileIndex mapping (best; deterministic)
        2) Else fall back to FieldX mapping ONLY if it matches on-disk tile ids
        3) Else skip (to avoid wrong order)
      Positions are normalized so min(X)=min(Y)=0.
    - ND2 fix: if TileScanInfo/Application indicates NIS-Elements (ND2 path), we flip X
      (PositionX -> -PositionX) to match the microscope coordinate handedness you observed.
      This is controlled by FlipX/FlipY attributes if present; otherwise inferred from Application.
    - Pixel size:
        Prefer PixelSizeUm stored in XML, else fall back to pixel_to_um argument.
    - Writes an optional coords CSV for debugging.
    """
    print("\033[1;96mConverting to OME-TIFFs\033[0m")

    # ==================================================================================
    # Helper: read PixelSizeUm from TileScanInfo if present
    # ==================================================================================
    def _get_pixel_size_um_from_xml(att: Optional[ET.Element]) -> Optional[float]:
        if att is None:
            return None
        raw_px = (att.attrib.get("PixelSizeUm") or "").strip()
        if not raw_px:
            return None
        try:
            return float(re.sub(r"[^\d.,+\-eE]", "", raw_px).replace(",", "."))
        except Exception:
            return None

    # ==================================================================================
    # Helper: find Tile nodes robustly under TileScanInfo
    # ==================================================================================
    def _find_tilescan_tiles(root: ET.Element) -> List[ET.Element]:
        tiles = root.findall(".//Attachment[@Name='TileScanInfo']//Tile")
        if tiles:
            return tiles
        return root.findall(".//Tile")

    # ==================================================================================
    # Helper: determine if we should flip coordinates (ND2 / NIS-Elements handedness)
    # ==================================================================================
    def _get_flip_flags(att: Optional[ET.Element]) -> Tuple[bool, bool]:
        """
        Returns (flip_x, flip_y).
        Priority:
          1) explicit FlipX/FlipY attributes on TileScanInfo
          2) infer from Application == NIS-Elements
        """
        flip_x = False
        flip_y = False

        if att is not None:
            fx = (att.attrib.get("FlipX", "") or "").strip()
            fy = (att.attrib.get("FlipY", "") or "").strip()

            if fx in ("1", "true", "True", "YES", "yes"):
                flip_x = True
            if fy in ("1", "true", "True", "YES", "yes"):
                flip_y = True

            if fx == "" and fy == "":
                app = (att.attrib.get("Application", "") or "").strip().lower()
                if app == "nis-elements":
                    # Your plots show (-x, y) is the correct orientation for ND2.
                    flip_x = True

        return flip_x, flip_y

    # ==================================================================================
    # Helper: read TileScanInfo positions keyed by tile id (TileIndex)
    # ==================================================================================
    def _positions_um_by_tile_from_tilescaninfo(
        tile_nodes: List[ET.Element],
        tiles_from_files: List[int],
    ) -> Tuple[Dict[int, Tuple[float, float]], str]:
        """
        Returns (pos_by_tile, source) using ONLY TileIndex.
    
        If TileIndex doesn't overlap with on-disk tile ids, returns ({}, "none").
        """
        tiles_set = set(tiles_from_files)
    
        pos_by_index: Dict[int, Tuple[float, float]] = {}
        saw_any = False
        for n in tile_nodes:
            ti = n.attrib.get("TileIndex", None)
            if ti is None:
                continue
            saw_any = True
            try:
                tid = int(ti)
                pos_by_index[tid] = (float(n.attrib["PosX"]), float(n.attrib["PosY"]))
            except Exception:
                continue
    
        if saw_any:
            overlap = tiles_set.intersection(pos_by_index.keys())
            if overlap:
                return pos_by_index, "TileIndex"
    
        return {}, "none"


    # ==================================================================================
    # Main loop
    # ==================================================================================
    for cycle in cycles:
        print(f"\033[1;90mProcessing Cycle {cycle}\033[0m")

        for region_directory in region_directories:
            region_directory = Path(region_directory)
            cycle_directory = region_directory / "preprocessing" / f"Cycle{cycle}"

            # Input MIPs
            mipped_directory = cycle_directory / "1_mipped"

            # Output OME-TIFF
            ome_tiff_directory = safe_mkdir(cycle_directory / "2_ome_tiffs")
            ome_tiff_path = ome_tiff_directory / f"Cycle{cycle}.ome.tiff"

            # XML metadata folder
            metadata_directory = cycle_directory / "MetaData"

            # ----------------------------------------------------------------------
            # STEP 0 — Skip if already converted
            # ----------------------------------------------------------------------
            if ome_tiff_path.exists():
                print(f"OME-TIFF exists: {ome_tiff_path}. Skipping.")
                continue

            # ----------------------------------------------------------------------
            # STEP 1 — Gather per-tile MIP files
            # ----------------------------------------------------------------------
            tif_files = natsorted(list(mipped_directory.glob("*.tif")))
            if not tif_files:
                print(f"No .tif files in {mipped_directory}. Skipping.")
                continue

            # Build file index: tile -> channel -> path
            file_index: Dict[int, Dict[int, Path]] = defaultdict(dict)
            for f in tif_files:
                m = re.search(r"_s0*(\d+)_ch0*(\d+)", f.name, re.IGNORECASE)
                if m:
                    tile, channel = map(int, m.groups())
                    file_index[tile][channel] = f

            tiles_from_files = sorted(file_index.keys())
            channels = sorted({ch for d in file_index.values() for ch in d.keys()})
            if not channels or not tiles_from_files:
                print("{BOLD}[WARN]⚠️ {RESET} Could not infer tiles/channels from filenames. Skipping.")
                continue

            print(
                f"[INFO] Found {len(tiles_from_files)} tile(s) on disk "
                f"for Cycle {cycle}, region '{region_directory.name}'."
            )


            # ----------------------------------------------------------------------
            # STEP 2 — Load TileScanInfo XML (prefer region_id.xml)
            # ----------------------------------------------------------------------
            region_id = region_directory.name
            preferred = metadata_directory / f"{region_id}.xml"
            md_candidates = sorted(metadata_directory.glob("*.xml"))

            if preferred.exists():
                md_file = preferred
            elif md_candidates:
                md_file = md_candidates[0]
                print(f"{BOLD}[WARN]⚠️ {RESET} Preferred XML missing; using {md_file.name}")
            else:
                print(f"No XML metadata in {metadata_directory}. Skipping.")
                continue

            try:
                root = ET.parse(md_file).getroot()
            except Exception as e:
                print(f"{BOLD}[WARN]⚠️ {RESET} Failed to parse {md_file}: {e}. Skipping.")
                continue

            att = root.find(".//Attachment[@Name='TileScanInfo']")
            tile_nodes = _find_tilescan_tiles(root)
            if not tile_nodes:
                print(f"{BOLD}[WARN]⚠️ {RESET} No <Tile> positions in {md_file.name}. Skipping.")
                continue

            flip_x, flip_y = _get_flip_flags(att)
            if flip_x or flip_y:
                print(f"[META] Applying coordinate flip(s): FlipX={int(flip_x)} FlipY={int(flip_y)}")

            # ----------------------------------------------------------------------
            # STEP 3 — Decide effective pixel size (µm/px): XML > argument
            # ----------------------------------------------------------------------
            px_from_xml = _get_pixel_size_um_from_xml(att)
            effective_px = px_from_xml if px_from_xml is not None else (
                float(pixel_to_um) if pixel_to_um is not None else None
            )

            if effective_px is None:
                print(
                    "{BOLD}[WARN]⚠️ {RESET} No pixel size available (neither XML PixelSizeUm nor pixel_to_um arg). "
                    "OME will be written without PhysicalSizeX/Y."
                )
            else:
                print(
                    f"[META] Using pixel size: {effective_px:.10f} µm/px "
                    f"({'XML' if px_from_xml is not None else 'argument'})"
                )

            # ----------------------------------------------------------------------
            # STEP 4 — Map XML positions to filename tile ids (NO sorting fallback)
            # ----------------------------------------------------------------------
            # TileIndex is an *identity key*, not an ordering.
            #
            # We ONLY join by:
            #   filename tile id (from CycleX_s{tile}_ch{ch}.tif)
            #        <->  XML TileIndex
            #        <->  (PosX, PosY) in microns
            #
            # Spatial layout is determined by (PosX, PosY) — NOT by TileIndex order and
            # NOT by FieldX/FieldY. This avoids “guessing” and stays correct for snake,
            # non-raster, or irregular acquisitions.
            #
            # If the TileIndex mapping is missing/inconsistent, we fail fast rather than
            # silently writing a wrong mosaic.
            # ----------------------------------------------------------------------
            
            pos_um_by_tile, pos_source = _positions_um_by_tile_from_tilescaninfo(tile_nodes, tiles_from_files)
            if not pos_um_by_tile:
                print(
                    "[ERROR] TileScanInfo has no usable TileIndex mapping to file tile ids. "
                    "Skipping OME-TIFF to avoid wrong tile placement."
                )
                continue
            
            print(f"[META] Position mapping source: {pos_source}")
            
            # Report tile counts: on-disk vs XML-declared
            tiles_from_xml = sorted(pos_um_by_tile.keys())
            tiles_on_disk = sorted(tiles_from_files)
            
            print(f"[INFO] Tiles on disk: {len(tiles_on_disk)}")
            print(f"[INFO] Tiles declared in XML: {len(tiles_from_xml)}")
            
            # XML tiles missing on disk → will show as black/empty areas in mosaic viewers
            missing_on_disk = [t for t in tiles_from_xml if t not in set(tiles_on_disk)]
            if missing_on_disk:
                print(
                    f"{BOLD}[WARN]⚠️ {RESET} XML declares {len(missing_on_disk)} tile(s) that do not exist on disk. "
                    f"Viewers may show these as black/empty regions. "
                    f"Example missing: {missing_on_disk[:10]}{'...' if len(missing_on_disk) > 10 else ''}"
                )
            
            # On-disk tiles missing in XML → cannot place them, so we skip them
            missing_in_xml = [t for t in tiles_on_disk if t not in pos_um_by_tile]
            if missing_in_xml:
                print(
                    f"{BOLD}[WARN]⚠️ {RESET} XML missing positions for {len(missing_in_xml)} on-disk tile(s); "
                    f"these will be skipped. "
                    f"Example missing: {missing_in_xml[:10]}{'...' if len(missing_in_xml) > 10 else ''}"
                )
            
            # Keep on-disk tile id ordering, but only for tiles we can position
            tiles_aligned = [t for t in tiles_on_disk if t in pos_um_by_tile]
            if not tiles_aligned:
                print("{BOLD}[WARN]⚠️ {RESET} No overlapping tiles between files and XML positions. Skipping.")
                continue
            
            print(
                f"[INFO] Will write {len(tiles_aligned)} tile(s) to OME-TIFF "
                f"(intersection of disk ∩ XML)."
            )
            
            x_um_raw = np.array([pos_um_by_tile[t][0] for t in tiles_aligned], dtype=float)
            y_um_raw = np.array([pos_um_by_tile[t][1] for t in tiles_aligned], dtype=float)

            # Apply ND2 flips BEFORE normalization (equivalent after, but clearer)
            if flip_x:
                x_um_raw = -x_um_raw
            if flip_y:
                y_um_raw = -y_um_raw

            # Normalize origin AFTER alignment (so x/y correspond to same tile ids)
            x_um = x_um_raw - float(np.min(x_um_raw))
            y_um = y_um_raw - float(np.min(y_um_raw))

            # Optional coord CSV for debugging
            if effective_px is not None:
                x_px = x_um / float(effective_px)
                y_px = y_um / float(effective_px)
                pd.DataFrame(
                    {"tile": tiles_aligned, "x_um": x_um, "y_um": y_um, "x_px": x_px, "y_px": y_px}
                ).to_csv(
                    ome_tiff_directory / f"Cycle{cycle}_coords.csv",
                    index=False,
                )

            # ----------------------------------------------------------------------
            # STEP 5 — Read one tile to determine image dimensions
            # ----------------------------------------------------------------------
            first_tile = tiles_aligned[0]
            first_ch = channels[0]
            first_img_path = file_index[first_tile][first_ch]
            try:
                height_px, width_px = tifffile.imread(first_img_path).shape
            except Exception as e:
                print(f"{BOLD}[WARN]⚠️ {RESET} Could not read {first_img_path} for dims: {e}. Skipping.")
                continue

            # ----------------------------------------------------------------------
            # STEP 6 — Write OME-TIFF (T tiles, C channels per tile)
            # ----------------------------------------------------------------------
            with tifffile.TiffWriter(ome_tiff_path, bigtiff=True, ome=True) as tif:
                for i, tile_id in enumerate(tiles_aligned):
                    image_stack = np.empty((len(channels), height_px, width_px), dtype=np.uint16)

                    for ci, ch in enumerate(channels):
                        try:
                            image_stack[ci] = tifffile.imread(file_index[tile_id][ch]).astype(np.uint16)
                        except Exception:
                            image_stack[ci] = np.zeros((height_px, width_px), dtype=np.uint16)

                    posX_um = float(x_um[i])
                    posY_um = float(y_um[i])

                    # Pixels metadata (physical size)
                    pixels_md = {}
                    if effective_px is not None:
                        pixels_md = {
                            "PhysicalSizeX": float(effective_px), "PhysicalSizeXUnit": "µm",
                            "PhysicalSizeY": float(effective_px), "PhysicalSizeYUnit": "µm",
                        }

                    # Plane metadata (position per channel plane)
                    plane_md = {
                        "PositionX": [posX_um] * len(channels),
                        "PositionY": [posY_um] * len(channels),
                    }

                    metadata = {"Pixels": pixels_md, "Plane": plane_md} if pixels_md else {"Plane": plane_md}
                    tif.write(image_stack, metadata=metadata)

            print(f"[DONE] Wrote OME-TIFF: {ome_tiff_path}")


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
    
    import ashlar.scripts.ashlar as ashlar
    ashlar.configure_terminal()

    maximum_shift=200
    filter_sigma=3
    
    for region_directory in region_directories:

        region_directory = Path(region_directory)
        region_suffix = region_directory.name  # e.g. "R1", "R10"
        
        if re.match(r"^R\d+$", region_suffix):
            print(f"\033[1mProcessing {region_suffix}\033[0m")            

        # --- STEP 1: Make directories for each cycle ---
        for cycle in cycles:
            cycle_directory = region_directory / 'preprocessing' / f'Cycle{cycle}'
            ome_tiff_directory = cycle_directory / '2_ome_tiffs'
            stitched_directory = cycle_directory / '3_stitched'
            safe_mkdir(stitched_directory)

        # --- STEP 2: Build OME-TIFF inputs in the EXACT requested cycles order ---
        ome_tiffs: List[Path] = []
        missing = []
        
        for cyc in cycles:
            ome_path = (
                region_directory
                / "preprocessing"
                / f"Cycle{cyc}"
                / "2_ome_tiffs"
                / f"Cycle{cyc}.ome.tiff"
            )
            if not ome_path.exists():
                missing.append(str(ome_path))
            else:
                ome_tiffs.append(ome_path)
        
        if missing:
            raise RuntimeError(
                "Missing OME-TIFF(s) for requested cycles. Expected these files:\n  - "
                + "\n  - ".join(missing)
            )
        
        print(f"Using OME-TIFF inputs (in cycles order): {cycles}")
        # NOTE: This order defines Ashlar output indices Cycle0, Cycle1, ... (temporary outputs)

        # --- PRECHECK 0: stitching must run on ALL cycles (no subsets) ---
        if len(cycles) != int(n_total_cycles):
            raise RuntimeError(
                f"[ERROR] Refusing to run stitching on a subset: "
                f"len(cycles)={len(cycles)} but n_total_cycles={n_total_cycles}. "
                f"Provide all cycles for stitching. cycles={cycles}"
    )

        # --- PRECHECK: enforce full experiment cycle count before running Ashlar ---
        # We intentionally do NOT allow Ashlar to run unless the dataset on disk contains
        # the expected total number of cycles for this experiment (n_total_cycles).
        # Earlier pipeline steps can run on subsets; alignment/stitching must not.
        all_ome_tiffs = natsorted([
            f for f in (region_directory / "preprocessing").rglob("2_ome_tiffs/*.ome.tiff")
        ])
        
        found_cycles = sorted(set(
            int(m.group(1))
            for f in all_ome_tiffs
            for m in [re.search(r"Cycle(\d+)", f.name)]
            if m
        ))
        
        if len(found_cycles) != int(n_total_cycles):
            raise RuntimeError(
                f"[ERROR] Refusing to run Ashlar: expected n_total_cycles={n_total_cycles} "
                f"but found {len(found_cycles)} cycle(s) on disk for {region_directory.name}: {found_cycles}. "
                f"Generate OME-TIFFs for all cycles (or fix n_total_cycles) before stitching."
            )



        # --- STEP 3: Determine n_channels from the first input (safe: guaranteed exists now) ---
        with tifffile.TiffFile(str(ome_tiffs[0])) as tif:
            # axes like 'TCZYX' or similar; robustly find 'C'
            axes = tif.series[0].axes
            if "C" not in axes:
                raise RuntimeError(f"OME-TIFF missing channel axis 'C': {ome_tiffs[0]}")
            n_channels = int(tif.series[0].shape[axes.index("C")])

        # Validate align_channel
        if not (0 <= int(align_channel) < int(n_channels)):
            raise ValueError(
                f"align_channel={align_channel} out of range for n_channels={n_channels} "
                f"(valid: 0..{n_channels-1})"
            )

        # --- STEP 3.1: Define expected stitched outputs for requested cycles ---
        expected_outputs = [
            Path(region_directory) / "preprocessing" / f"Cycle{cyc}" / "3_stitched" / f"Cycle{cyc}_ch{ch}.tif"
            for cyc in cycles
            for ch in range(n_channels)
        ]

        # Skip only if *all* expected outputs exist AND are non-trivially sized (valid).
        # This prevents skipping when a previous run left tiny/corrupted files behind.
        if all(file_exists_and_valid(p, min_size=1024) for p in expected_outputs):
            print(
                f"Stitched images already exist (and look valid) for all requested cycles "
                f"(cycles={len(cycles)}, channels={n_channels}). Skipping."
            )
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
        safe_mkdir(Path(tmp_pattern).parent)
    
        # --- STEP 5: Run Ashlar with inputs in the correct order ---
        ome_tiff_files = [str(p) for p in ome_tiffs]
        
        tmp_path = region_directory / "preprocessing" / "ashlar_tmp"
        safe_mkdir(tmp_path)
        
        tmp_pattern = str(tmp_path / "Cycle{cycle}_ch{channel}.tif")
        
        try:
            if plates:
                rc = ashlar.process_plates(
                    ome_tiff_files,
                    None,
                    tmp_pattern,
                    flip_x, flip_y,
                    ffp_paths, dfp_paths,
                    0.0,
                    aligner_args, mosaic_args,
                    pyramid, quiet
                )
            else:
                rc = ashlar.process_single(
                    ome_tiff_files,
                    tmp_pattern,
                    flip_x, flip_y,
                    ffp_paths, dfp_paths,
                    0.0,
                    aligner_args, mosaic_args,
                    pyramid, quiet
                )
        
            if rc not in (None, 0):
                raise RuntimeError(f"Ashlar returned non-zero status code: {rc}")
        
        except ashlar.ProcessingError as e:
            ashlar.print_error(str(e))
            continue
        
        except Exception as e:
            ashlar.print_error(f"Unexpected error during Ashlar run: {e}")
            continue  
            
        # --- STEP 6: Remap Ashlar temp cycle indices back onto REAL cycle numbers ---
        #
        # IMPORTANT ASSUMPTION (now enforced explicitly):
        # Ashlar writes:
        #   Cycle0_ch*, Cycle1_ch*, ...
        # where the numeric index corresponds EXACTLY to the order of ome_tiff_files.
        #
        # In this pipeline:
        #   ome_tiff_files are ordered to match `cycles`
        # Therefore:
        #   tmp Cycle{i}  →  real Cycle{cycles[i]}
        #

        tmp_path = Path(tmp_pattern).parent

        # Sanity check: ensure all expected temp files exist BEFORE moving anything
        expected_tmp = [
            tmp_path / f"Cycle{cyc_idx}_ch{ch}.tif"
            for cyc_idx in range(len(cycles))
            for ch in range(n_channels)
        ]

        missing_tmp = [p for p in expected_tmp if not p.exists()]
        if missing_tmp:
            raise FileNotFoundError(
                f"Ashlar temp output incomplete: missing {len(missing_tmp)} file(s). "
                f"Example missing: {missing_tmp[:5]}"
            )

        # Now remap safely
        for cyc_idx, cyc in enumerate(cycles):
            stitched_dir = region_directory / "preprocessing" / f"Cycle{cyc}" / "3_stitched"
            safe_mkdir(stitched_dir)

            for ch in range(n_channels):
                tmp_file = tmp_path / f"Cycle{cyc_idx}_ch{ch}.tif"
                final_file = stitched_dir / f"Cycle{cyc}_ch{ch}.tif"

                # Overwrite protection: warn but replace
                if final_file.exists():
                    print(f"{BOLD}[WARN]⚠️ {RESET} Overwriting existing stitched file: {final_file}")

                tmp_file.replace(final_file)

        print(
            f"[INFO] Remapped Ashlar outputs: "
            f"{len(cycles)} cycles × {n_channels} channels "
            f"from {tmp_path} → region Cycle folders"
        )

        # Clean up temporary Ashlar directory
        shutil.rmtree(tmp_path, ignore_errors=True)
   
    
def retile_stitched_images(
    region_directories,
    cycles,
    tile_dimension=6000
):
    """
    Tiles stitched .tif images from each region/cycle and saves them with a naming convention:
        Cycle{cycle}_s{tile_index}_ch{channel}.tif

    Also writes a CSV of tile upper-left pixel coordinates (x,y) for the tiling grid.

    Notes / assumptions
    -------------------
    - Input stitched images are 2D per channel (Y, X).
    - All stitched channel images in a given (region, cycle) are the same shape.
    - Output tile indices (s*) are per-channel (i.e. each channel gets s0..sN-1).
    """
    def _import_cv2():
        import cv2
        return cv2
        
    cv2 = _import_cv2()

    print(f"\033[1;96mRetiling stitched images\033[0m")

    for cycle in cycles:
        print(f"\033[1;90mProcessing Cycle {cycle}\033[0m")

        for region_directory in region_directories:
            region_directory = Path(region_directory)

            region_name = region_directory.name
            if re.match(r"^R\d+$", region_name):
                print(f"\033[1mProcessing {region_name}\033[0m")

            cycle_directory = region_directory / "preprocessing" / f"Cycle{cycle}"
            stitched_directory = cycle_directory / "3_stitched"
            retiled_directory = cycle_directory / "4_retiled"
            safe_mkdir(retiled_directory)

            tif_files = sorted([p for p in stitched_directory.iterdir() if p.is_file() and p.suffix.lower() == ".tif"])
            if not tif_files:
                print(f"No stitched TIFFs found for cycle {cycle} in {stitched_directory}")
                continue

            # ------------------------------------------------------------------
            # Pre-check: determine expected tiling grid from FIRST stitched file
            # (avoid loading full image just to get shape)
            # ------------------------------------------------------------------
            try:
                with tifffile.TiffFile(str(tif_files[0])) as tf:
                    shape = tf.series[0].shape
            except Exception as e:
                print(f"[ERROR] Could not read shape from {tif_files[0].name}: {e}")
                continue

            if len(shape) != 2:
                print(f"[ERROR] Expected 2D stitched image (Y,X) but got shape={shape} in {tif_files[0].name}")
                continue

            img_h, img_w = int(shape[0]), int(shape[1])

            pad_h = (math.ceil(img_h / tile_dimension) * tile_dimension) - img_h
            pad_w = (math.ceil(img_w / tile_dimension) * tile_dimension) - img_w
            padded_h = img_h + pad_h
            padded_w = img_w + pad_w

            nrows = padded_h // tile_dimension
            ncols = padded_w // tile_dimension
            expected_tiles_per_img = nrows * ncols
            expected_total_tiles = expected_tiles_per_img * len(tif_files)

            # Compute tile grid coords ONCE (same for all channels/images)
            x_positions = [col * tile_dimension for row in range(nrows) for col in range(ncols)]
            y_positions = [row * tile_dimension for row in range(nrows) for col in range(ncols)]

            existing_tiles = list(retiled_directory.glob(f"Cycle{cycle}_s*_ch*.tif"))

            # If the number of tiles matches, sample-check one tile size
            if len(existing_tiles) == expected_total_tiles and existing_tiles:
                try:
                    sample_tile = tifffile.imread(existing_tiles[0])
                except Exception as e:
                    print(f"{BOLD}[WARN]⚠️ {RESET} Could not read sample existing tile: {existing_tiles[0].name}: {e}")
                    sample_tile = None

                if sample_tile is None or sample_tile.shape != (tile_dimension, tile_dimension):
                    print(
                        f"{BOLD}[WARN]⚠️ {RESET} Existing tiles look wrong (expected {tile_dimension}×{tile_dimension}). "
                        f"Reprocessing all tiles in {retiled_directory}."
                    )
                    for p in existing_tiles:
                        p.unlink()
                else:
                    print(
                        f"All expected tiles found ({expected_total_tiles}) and sample tile shape matches "
                        f"({tile_dimension}×{tile_dimension}). Skipping."
                    )
                    # Still ensure coords CSV exists (optional but useful)
                    coords_csv_path = retiled_directory / f"Cycle{cycle}_retiled_coords.csv"
                    if not coords_csv_path.exists():
                        pd.DataFrame({"x": x_positions, "y": y_positions}).to_csv(coords_csv_path, header=False, index=False)
                    continue
            else:
                print(f"Missing/extra tiles (expected {expected_total_tiles}, found {len(existing_tiles)}). Reprocessing all.")
                for p in existing_tiles:
                    p.unlink()

            # ------------------------------------------------------------------
            # Begin tiling (per stitched channel image)
            # ------------------------------------------------------------------
            for tif_path in tif_files:
                try:
                    image = tifffile.imread(str(tif_path))
                    if image.ndim != 2:
                        raise ValueError(f"Expected 2D image, got shape={image.shape}")

                    print(f"Tiling: {tif_path.name}")

                    # Pad to multiple of tile_dimension (pad is on bottom/right only)
                    pad_h = (math.ceil(image.shape[0] / tile_dimension) * tile_dimension) - image.shape[0]
                    pad_w = (math.ceil(image.shape[1] / tile_dimension) * tile_dimension) - image.shape[1]

                    image_padded = cv2.copyMakeBorder(
                        image,
                        top=0, bottom=pad_h,
                        left=0, right=pad_w,
                        borderType=cv2.BORDER_CONSTANT,
                        value=0,  # explicit black padding
                    )

                    img_height, img_width = image_padded.shape
                    nrows = img_height // tile_dimension
                    ncols = img_width // tile_dimension

                    # (nrows, tile, ncols, tile) -> (nrows, ncols, tile, tile)
                    tiled_array = image_padded.reshape(nrows, tile_dimension, ncols, tile_dimension).swapaxes(1, 2)

                    # Channel from filename "..._chX.tif"
                    m = re.search(r"ch(\d+)", tif_path.stem, re.IGNORECASE)
                    channel = int(m.group(1)) if m else 0

                    tile_count = 0
                    for row in range(nrows):
                        for col in range(ncols):
                            tile_img = tiled_array[row, col]
                            tile_filename = retiled_directory / f"Cycle{cycle}_s{tile_count}_ch{channel}.tif"
                            tifffile.imwrite(str(tile_filename), tile_img)
                            tile_count += 1

                except Exception as e:
                    print(f"[ERROR] Processing {tif_path.name}: {e}")
                    continue

            # Write coords CSV once (grid coords)
            coords_csv_path = retiled_directory / f"Cycle{cycle}_retiled_coords.csv"
            pd.DataFrame({"x": x_positions, "y": y_positions}).to_csv(coords_csv_path, header=False, index=False)
            print(f"Tiling complete. Positions saved to {coords_csv_path}")





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





# -------------------------------------------------------------------------------------
# MAIN FUNCTION FOR PREPROCESSING
# -------------------------------------------------------------------------------------
def preprocessing_main(input_dirs,
                            cycles,
                            output_dir_prefix,
                            mode,
                            deconvolution_method=None,
                            PSF_metadata=None, 
                            align_channel=4, 
                            n_total_cycles=5,
                            mip=True,
                            tile_dimension=6000,  
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
                            deconvolution_method=deconvolution_method,
                            PSF_metadata=PSF_metadata, 
                            mip=mip,
                            chunk_size=chunk_size)

    # OME TIFFS
    mipped_to_OME_tiffs(
        region_directories=region_directories, 
        cycles=cycles)

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
    deconvolution_method: Optional[str] = None,
    PSF_metadata: Optional[dict] = None, 
    mip: bool = True,
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
        print(f"Deconvolution method: {deconvolution_method}".ljust(width))
    
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
            dims = czi.dims
            print("CZI dims:", dims)
        
            # --- Regions (Scenes = S dimension) ---
            num_regions = dims.get("S", 1)
            regions = [f"Region_{i+1}" for i in range(num_regions)]
        
            print(f"[CZI MODE] Detected {num_regions} region(s): {regions}")


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
        
            # ----- Step 1: Prepare file lists based on mode -----
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
                            pattern = f"--C{str(channel).zfill(2)}"
                        else:
                            pattern = f"_ch{channel}"
                        channel_files = [f for f in files_in_tile if pattern in f.name]
                        tile_channel_files[(tile, channel)] = channel_files
        
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
                mosaic = image_dict.get('mosaic_position', None) # Get mosaic positions
                num_channels = image_dict['channels']
                channels = list(range(num_channels))  # [0, 1, 2, 3, 4, 5]

            elif mode == 'czi':
                # For each region, get tile, channel, and Z info
                size_z = dims.get("Z", 1)
                num_channels = dims.get("C", 1)
                n_tiles = dims.get("M", 1)
                image_dimensions = (dims.get("X"), dims.get("Y"))
                channels = list(range(num_channels))
                tiles = list(range(n_tiles))
            
                print(
                    f"[CZI MODE] {region}: "
                    f"{n_tiles} tiles, {size_z} Z-slices, {num_channels} channels, "
                    f"image size {image_dimensions[0]} × {image_dimensions[1]}"
                )
            
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
    
            
        
            # ----- Step 2: Copy metadata if available -----
            print('Extracting metadata')
            
            if mode in ('tif_autosaved', 'tif_exported'):
                # Look for Metadata subdirectory inside the input directory
                input_metadata_dir = input_dir / 'Metadata'
                
                if input_metadata_dir.exists():
                    # Find metadata files matching the current region
                    metadata_files = [f for f in input_metadata_dir.iterdir() if region in f.name]
                    
                    # Select the first metadata file that is NOT a 'properties' file
                    metadata_file = next((f for f in metadata_files if 'properties' not in f.name), None)
                    
                    # Copy the selected metadata file to the metadata output directory
                    if metadata_file:
                        custom_copy(metadata_file, metadata_directory)
            
            elif mode == 'lif' and mosaic is not None:
                # If in LIF mode and mosaic info is available, generate XML metadata
                
                # Build XML structure: <Data> -> <Image> -> <Attachment> -> multiple <Tile> elements
                data = ET.Element("Data")
                image_elem = ET.SubElement(data, "Image", TextDescription="")
                
                attachment = ET.SubElement(
                    image_elem, 
                    "Attachment", 
                    Name="TileScanInfo", 
                    Application="LAS AF", 
                    FlipX="0", 
                    FlipY="0", 
                    SwapXY="0"
                )
            
                # Add a <Tile> element for each mosaic tile with positional info
                for x, y, pos_x, pos_y in mosaic:
                    ET.SubElement(
                        attachment, 
                        "Tile", 
                        FieldX=str(x), 
                        FieldY=str(y),
                        PosX=f"{pos_x:.10f}", 
                        PosY=f"{pos_y:.10f}"
                    )
            
                # Write the XML tree to a file in the metadata output directory
                tree = ET.ElementTree(data)
                tree.write(metadata_directory / f"{image_name}.xml", encoding="utf-8", xml_declaration=True)

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
            

            # ----- Step 3: Generate PSFs for all channels -----
            print('Calculating the PSF')
        
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
        
            # ----- Step 4: Deconvolve each tile and channel -----
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

                    elif mode == 'nd2':
                        # ND2 array has shape (M, Z, C, Y, X)
                        # Select one tile (M), all Z, one channel (C), full XY
                        stacked_images = arr[int(tile), :, int(channel), :, :]
                        print(f"ND2 stacked_images shape (tile {tile}, ch {channel}): {stacked_images.shape}")

        
                    # Deconvolution with RedLionFish method
                    if deconvolution_method == 'redlionfish':
                        deconvolved_images = rl.doRLDeconvolutionFromNpArrays(stacked_images, psf_dict[str(channel)], niter=50)
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
                    print(f"\033[1;37m[Timing] Full deconvolution cycle for Tile {tile}, Channel {channel} took {tile_channel_end - tile_channel_start:.2f} seconds\033[0m")
            
            # After all tiles and channels are done
            if mip and stacked_directory.exists():
                shutil.rmtree(stacked_directory)
                print(f"Deleted stacked directory: {stacked_directory}")
    
    return region_directories


def mipped_to_OME_tiffs(region_directories, cycles):
    """
    Convert Leica TIFF tiles into an OME-TIFF with spatial metadata.

    Args:
        region_directories (list of Path or str): Directories for different regions.
        cycle (int): Cycle number (used in output filename).

    Returns:
        None. Outputs OME-TIFF and CSV with tile positions.
    """

    print(f"\033[1;96mConverting to OME-TIFFs\033[0m")

    for cycle in cycles:
        print(f"\033[1;90mProcessing Cycle {cycle}\033[0m")
    
        for region_directory in region_directories:
            region_suffix = region_directory[-2:]
            if re.match(r"R\d+", region_suffix):
                print(f"\033[1mProcessing {region_suffix}\033[0m")
            
            region_directory = Path(region_directory)
            cycle_directory = region_directory / 'preprocessing' / f'Cycle{cycle}'            
            mipped_directory = cycle_directory / '1_mipped'
            ome_tiff_directory = cycle_directory / '2_ome_tiffs'
            ome_tiff_path = ome_tiff_directory / f'Cycle{cycle}.ome.tiff'
            metadata_directory = cycle_directory / 'MetaData'
        
            ome_tiff_directory.mkdir(parents=True, exist_ok=True)
        
            if ome_tiff_path.exists():
                print(f"OME-TIFF already exists: {ome_tiff_path}. Skipping.")
                continue
        
            tif_files = list(mipped_directory.glob('*.tif'))
            if not tif_files:
                print(f"No TIFF files found in {mipped_directory}. Skipping.")
                continue
        
            # Build file index: tile → channel → filepath
            file_index = defaultdict(dict)
            for f in tif_files:
                match = re.search(r'_s(\d+)_ch(\d+)', f.name)
                if match:
                    tile, channel = match.groups()
                    file_index[tile][channel] = f
        
            tiles = sorted(file_index.keys(), key=int)
            channels = sorted({ch for chs in file_index.values() for ch in chs}, key=int)
        
            if not channels:
                print(f"[WARN] No channels found in {mipped_directory}. Check filename pattern.")
                continue
        
            # Load tile positions from XML
            metadata_files = list(metadata_directory.glob('*.xml')) + list(metadata_directory.glob('*.xlif'))
            if not metadata_files:
                print(f"No metadata found in {metadata_directory}. Skipping.")
                continue
        
            root = ET.parse(metadata_files[0]).getroot()
            tile_elements = root.findall(".//Tile")
            x_coords = np.array([float(t.attrib['PosX']) for t in tile_elements])
            y_coords = np.array([float(t.attrib['PosY']) for t in tile_elements])
        
            pixel_size_um = 0.1625
            scale = 3.21e-7  # Leica-provided scaling
        
            x_scaled = ((x_coords - x_coords.min()) / scale + 1).astype(int)
            y_scaled = ((y_coords - y_coords.min()) / scale + 1).astype(int)
            positions = np.column_stack((x_scaled, y_scaled))
           
            pd.DataFrame({'x': x_scaled, 'y': y_scaled}).to_csv(
                ome_tiff_directory / f'Cycle{cycle}_coords.csv', index=False)
        
            # Get image shape
            first_tile = next(iter(file_index.values()))
            first_channel = next(iter(first_tile.values()))
            height, width = tifffile.imread(first_channel).shape
            
            # Write tile-by-tile into OME-TIFF file
            with tifffile.TiffWriter(ome_tiff_path, bigtiff=True, ome=True) as tif:
                for tile_index, tile in enumerate(tiles):
                    position = positions[tile_index]
                    image_stack = np.empty((len(channels), height, width), dtype=np.uint16)
        
                    for channel_index, channel in enumerate(channels):
                        try:
                            image_stack[channel_index] = tifffile.imread(file_index[tile][channel]).astype(np.uint16)
                        except Exception as e:
                            print(f"[WARN] Tile {tile}, Channel {channel} missing or unreadable: {e}")
                            image_stack[channel_index] = np.zeros((height, width), dtype=np.uint16)
        
                    metadata = {
                        'Pixels': {
                            'PhysicalSizeX': pixel_size_um,
                            'PhysicalSizeXUnit': 'µm',
                            'PhysicalSizeY': pixel_size_um,
                            'PhysicalSizeYUnit': 'µm'
                        },
                        'Plane': {
                            'PositionX': [position[0] * pixel_size_um] * len(channels),
                            'PositionY': [position[1] * pixel_size_um] * len(channels)
                        }
                    }
    
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
    filter_sigma=5.0, 
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
                
        # Define the expected full range of cycles
        expected_cycles = list(range(1, n_total_cycles + 1))
        
        # Sanity check: must match exactly
        if found_cycles != expected_cycles:
            missing = [c for c in expected_cycles if c not in found_cycles]
            extra = [c for c in found_cycles if c not in expected_cycles]
        
            print(f"Expected cycles {expected_cycles}, but found {found_cycles}. OME tiffs for all expected cycles need to be available before stitching and aligning.")
            if missing:
                print(f"   Missing cycles: {missing}")
            if extra:
                print(f"   Unexpected cycles: {extra}")
        
            raise RuntimeError(
                f"Cycle mismatch. Expected {expected_cycles}, but found {found_cycles}."
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
            for cyc in expected_cycles
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
         
            if plates:
                ashlar.process_plates(
                    ome_tiff_files,
                    None,                      # don’t give a base output dir
                    tmp_pattern,   # full pattern string
                    flip_x, flip_y, ffp_paths, dfp_paths,
                    0.0,                       # barrel_correction
                    aligner_args, mosaic_args,
                    pyramid, quiet
                )
            else:
                ashlar.process_single(
                    ome_tiff_files,
                    tmp_pattern,   # same pattern string
                    flip_x, flip_y, ffp_paths, dfp_paths,
                    0.0,                       # barrel_correction
                    aligner_args, mosaic_args,
                    pyramid, quiet
                )

            
        except ashlar.ProcessingError as e:
            ashlar.print_error(str(e))
            continue

        # --- STEP 6: Remap Cycle0..N-1 → Cycle1..N and move files ---
        tmp_dir = Path(tmp_pattern).parent
        for cyc_idx, cyc in enumerate(expected_cycles):
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
                    channel_num = int(channel_match.group(1)) if channel_match else 0
    
                    tile_count = 0
                    for row in range(nrows):
                        for col in range(ncols):
                            x_positions.append(col * tile_dimension)
                            y_positions.append(row * tile_dimension)
    
                            tile_img = tiled_array[row, col]
                            tile_filename = retiled_directory / f'Cycle{cycle}_s{tile_count}_ch{channel_num}.tif'
                            tifffile.imwrite(tile_filename, tile_img)
                            tile_count += 1
    
                except Exception as e:
                    print(f"[ERROR] Processing {tif_path.name}: {e}")
                    continue
    
            # Save tile positions
            tile_positions_df = pd.DataFrame({'x': x_positions, 'y': y_positions})
            coords_csv_path = retiled_directory / f'Cycle{cycle}_retiled_coords.csv'
            tile_positions_df.to_csv(coords_csv_path, header=False, index=False)
    
            print(f"Tiling complete. Positions saved to {coords_csv_path}")
    






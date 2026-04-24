import os
import re
import numpy as np
import tifffile
from pathlib import Path
from typing import Mapping, Tuple, Union
from skimage.io import imread
from slicedimage import ImageFormat
from starfish import Codebook
from starfish.types import Axes, Coordinates, Features, Number
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
import xml.etree.ElementTree as ET  
from datetime import datetime  # 

def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

# -----------------------------------------------------------------------------
# XML provenance policy (SpaceTX stage)
#
# 1) Only write an XML if this run actually generates SpaceTX outputs for the region
#    (i.e. NOT skipped by the early-exit that detects existing experiment.json + codebook.json).
#
# 2) Never overwrite existing XMLs for partial runs: each productive run writes a
#    uniquely named XML file using a filesystem-safe UTC timestamp run_id.
#
# 3) Runs that skip because outputs already exist produce NO XML.
# -----------------------------------------------------------------------------

# Represents one image tile with coordinate mapping and pixel data
class ISSTile2D(FetchedTile):
    def __init__(self, file_path, tile, tilexy, tile_dims, pixel_to_um):
        self.file_path = file_path
        self.tile = tile
        self.tilexy = tilexy
        self.tile_dims = tile_dims
        self.pixel_to_um = pixel_to_um

    # Returns the shape of the tile (height, width)
    @property
    def shape(self) -> Mapping[Axes, int]:
        return {
            Axes.Y: self.tile_dims[0],
            Axes.X: self.tile_dims[1],
        }

    # Returns physical coordinates for the tile
    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return {
            Coordinates.X: (self.tilexy[self.tile, 0] * self.pixel_to_um,
                            (self.tilexy[self.tile, 0] + self.tile_dims[1]) * self.pixel_to_um),
            Coordinates.Y: (self.tilexy[self.tile, 1] * self.pixel_to_um,
                            (self.tilexy[self.tile, 1] + self.tile_dims[0]) * self.pixel_to_um),
            Coordinates.Z: (0.0, 0.0),
        }

    # Loads the pixel data from file
    def tile_data(self) -> np.ndarray:
        #print(f"Loading tile data from {self.file_path}")
        return imread(self.file_path)


# Fetcher for primary image tiles
class ISS2DPrimaryTileFetcher(TileFetcher):
    def __init__(self, region_directory, cycle_names, channel_order, tilexy, tile_dims, pixel_to_um, tiles_subdir):
        self.region_directory = region_directory
        self.cycle_names = cycle_names
        self.channel_order = channel_order
        self.tilexy = tilexy
        self.tile_dims = tile_dims
        self.pixel_to_um = pixel_to_um
        self.tiles_subdir = tiles_subdir  # <-- ADDED

    # Returns an ISSTile2D object for a given tile, cycle, channel, and z-plane
    def get_tile(self, tile: int, c: int, ch: int, z: int) -> FetchedTile:
        cycle = self.cycle_names[c]
        path = self.region_directory / "preprocessing" / cycle / self.tiles_subdir  # <-- CHANGED
        file_path = path / f"{cycle}_s{tile}_ch{self.channel_order[ch]}.tif"
        #print(f"Fetching primary tile: {file_path}")
        return ISSTile2D(file_path, tile, self.tilexy[cycle], self.tile_dims, self.pixel_to_um)


# Fetcher for auxiliary tiles 
class ISS2DAuxTileFetcher(TileFetcher):
    def __init__(self, region_directory, cycle_names, nuclei_channel, tilexy, tile_dims, pixel_to_um, tiles_subdir):
        self.region_directory = region_directory
        self.cycle_names = cycle_names
        self.nuclei_channel = nuclei_channel
        self.tilexy = tilexy
        self.tile_dims = tile_dims
        self.pixel_to_um = pixel_to_um
        self.tiles_subdir = tiles_subdir  # <-- ADDED

    # Returns an ISSTile2D object for a given tile, cycle, and z-plane (fixed nuclei channel)
    def get_tile(self, tile: int, c: int, ch: int, z: int) -> FetchedTile:
        nuclei_ch = self.nuclei_channel -1 # change to 0-indexed
        cycle = self.cycle_names[c]
        path = self.region_directory / "preprocessing" / cycle / self.tiles_subdir  # <-- CHANGED
        file_path = path / f"{cycle}_s{tile}_ch{nuclei_ch}.tif"
        #print(f"Fetching aux tile: {file_path}")
        return ISSTile2D(file_path, tile, self.tilexy[cycle], self.tile_dims, self.pixel_to_um)


# Main function to create SpaceTx-compatible experiment and codebook files for a region
def make_spacetx_format(input_dir, 
                        codebook_csv,
                        regions_to_process=None,
                        output_dir_prefix=None,
                        pixel_to_um=1,
                        channels=["DAPI", "Cy3", "Cy5", "AF750", "AF488"],
                        DO_decorators=["AF750", "Cy5", "Cy3", "AF488"],
                        nuclei_channel="DAPI",
                        CARE = False):
    """
    Create the experiment directory, tile fetchers, and codebook JSON files used for downstream decoding.

    This function discovers regions and cycles produced by the preprocessing module, then writes:
      - experiment.json describing where image tiles live on disk
      - codebook.json describing the gene-by-round/channel encoding

    CARE integration:
      - If CARE=False (default), tiles are read from: preprocessing/<CycleX>/4_retiled/
      - If CARE=True, tiles are read from:  preprocessing/<CycleX>/4_retiled/CARE/
        (i.e. the denoised outputs written by the ISS_CARE step)

    Args:
        input_dir (str or Path): Top-level experiment directory containing region folders (e.g., R1, R2).
        codebook_csv (str or Path): Path to the codebook CSV.
        regions_to_process (list[int] | None): Optional list of 1-based region indices to process.
        output_dir_prefix (str | Path | None): Optional base directory where outputs should be written.
        pixel_to_um (float): Physical pixel size in microns (µm per pixel). This value sets the units of spatial 
        coordinates (e.g. xc, yc) reported in experiment metadata and decoded outputs. Use 1.0 for pixel-based 
        coordinates, or your microscope-specific value for microns.
        channels (list[str]): List of all channel names (must include "DAPI").
        DO_decorators (list[str]): RNA channel names used for decoding (subset of `channels`).
        nuclei_channel (str): Name of the nuclei channel (must be present in `channels`).
        CARE (bool): If True, use denoised tiles from 4_retiled/CARE. If False, use 4_retiled.

    Behavior:
        - Discovers region folders matching R\\d+ under input_dir.
        - For each region, discovers cycles named Cycle\\d+ under <region>/preprocessing/.
        - Reads tile position CSVs from the selected tile directory for each cycle.
        - Writes experiment.json and codebook.json into decoding/1_SpaceTX_format (mirrored under output_dir_prefix if set).
        - Backs up JSON files under decoding/1_SpaceTX_format/original_jsons.
    """


    input_dir = Path(input_dir)
    print(f"[INFO] Processing directory: {input_dir}")

    # Generate one run_id per function invocation.
    # Each region that actually gets written will produce one XML with this run_id.
    run_id = timestamp_for_filename()

    # Select which retiled directory to read tiles from.
    # CARE=False -> preprocessing/<cycle>/4_retiled
    # CARE=True  -> preprocessing/<cycle>/4_retiled/CARE
    tiles_subdir = "4_retiled/CARE" if CARE else "4_retiled"

    # Single, explicit log line for reproducibility in pipeline logs.
    print(
        f"[INFO] Using {'CARE-denoised' if CARE else ''} retiled images "
        f"from: preprocessing/<cycle>/{tiles_subdir}"
    )
    


    # --- Output directory prefix handling ---
    if output_dir_prefix is not None:
        output_dir_prefix = Path(output_dir_prefix)
        output_dir_prefix.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Using output_dir_prefix: {output_dir_prefix.resolve()}")
    else:
        print("[INFO] Using default output location under each region directory")
    

    codebook_csv = Path(codebook_csv)
    print('[INFO] Codebook: ', codebook_csv) 

    if pixel_to_um == 1:
        print(f"[INFO] Spatial units: coordinates will be reported in pixels; pixel_to_um = {pixel_to_um}")
    else:
        print(f"[INFO] Spatial units: coordinates will be reported in microns (µm); pixel_to_um = {pixel_to_um}")


    nuclei_channel = channels.index("DAPI") + 1 # 1-indexed

    # --- Step 1: Find/select region directories matching R\d+ ---
    region_pattern = re.compile(r"^R(\d+)$")
    
    regions_found = []
    for r in input_dir.iterdir():
        if not r.is_dir():
            continue
        m = region_pattern.match(r.name)
        if m:
            regions_found.append((int(m.group(1)), r))
    
    regions_found.sort(key=lambda t: t[0])  # R1, R2, R10 correctly
    
    if not regions_found:
        raise RuntimeError(f"No regions found in {input_dir} (expected folders like R1, R2, ...)")
    
    available_numbers = [n for n, _ in regions_found]
    available_map = {n: p for n, p in regions_found}
    
    all_regions = [f"R{n}" for n in available_numbers]
    print(f"[INFO] Regions found on disk ({len(all_regions)}): {all_regions}")
    
    # --- Select regions to process ---
    if regions_to_process is None:
        region_numbers = available_numbers
    else:
        if not isinstance(regions_to_process, (list, tuple)):
            raise TypeError("regions_to_process must be a list of 1-based ints, e.g. [1, 2].")
    
        region_numbers = [int(x) for x in regions_to_process]
        if any(x < 1 for x in region_numbers):
            raise ValueError(f"regions_to_process contains invalid region numbers: {regions_to_process}")
    
        missing = [n for n in region_numbers if n not in available_map]
        if missing:
            raise FileNotFoundError(
                f"Requested region(s) not found: {[f'R{n}' for n in missing]}. "
                f"Available regions: {all_regions}"
            )
    
    # keep the user’s requested order
    region_directories = [available_map[n] for n in region_numbers]
    
    selected_regions = [f"R{n}" for n in region_numbers]
    skipped_regions = [r for r in all_regions if r not in selected_regions]
    
    print(f"[INFO] Regions selected ({len(selected_regions)}): {selected_regions}")
    if skipped_regions:
        print(f"[INFO] Regions skipped ({len(skipped_regions)}): {skipped_regions}")
    
    print("[INFO] Regions to be processed:", selected_regions)

    
    for region_directory in region_directories:
        region_name = region_directory.name

        # Create output directory for this region
        if output_dir_prefix is None:
            # Original behavior: write under each region folder
            SpaceTX_dir = region_directory / "decoding" / "1_SpaceTX_format"
        else:
            # New behavior: write under output_dir_prefix, mirroring region name
            SpaceTX_dir = Path(output_dir_prefix) / region_directory.name / "decoding" / "1_SpaceTX_format"
        
        SpaceTX_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"\033[1mProcessing region {region_name}\033[0m")

        # ===== EARLY EXIT / MINIMAL WORK BRANCHING =====
        experiment_json_path = SpaceTX_dir / "experiment.json"
        codebook_json_path = SpaceTX_dir / "codebook.json"
        
        # If both key outputs already exist, skip and write NO XML for this region.
        if experiment_json_path.exists() and codebook_json_path.exists():
            print(f"[{region_name}] Skipping: experiment.json and codebook.json already exist.")
            print(f"  ✔ {experiment_json_path}")
            print(f"  ✔ {codebook_json_path}")
            continue
                
        # Find all cycles for this region (Cycle1, Cycle2, ...)
        cycle_names = sorted([
            c.name for c in (region_directory / "preprocessing").glob("Cycle*")
            if c.is_dir() and re.fullmatch(r"Cycle\d+", c.name)
        ])

        if not cycle_names:
            print(f"Warning: No cycles found in {region_directory}, skipping...")
            continue
        
        width = 80
        print("=" * width + "\033[0m")
        print(f"\033[1;90mProcessing region: {region_name}\033[0m")
        print(f"Output SpaceTX directory: {SpaceTX_dir}")
        print(f"Number of cycles to be processed: {len(cycle_names)}")
        
        retiled_files = []
        tile_counts = {}
        tilexy_per_cycle = {}
        tile_dims = None
        
        # --- Step 2: Gather tile information and positions for all cycles ---
        for cycle_index, cycle in enumerate(cycle_names):
            cycle_dir = region_directory / "preprocessing" / cycle / tiles_subdir

            # Find all tile files for this cycle
            tif_files = sorted(cycle_dir.glob("*_ch0.tif"))
            retiled_files.extend(tif_files)
            tile_counts[cycle] = len(tif_files)
            #print(f"Number of tiles to be processed: {len(tif_files)}")
        
            # Find CSV with tile positions
            csv_files = list(cycle_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No tile position CSV found in {cycle_dir}")
            elif len(csv_files) > 1:
                raise ValueError(f"Multiple CSV files found in {cycle_dir}, but only one expected.")
            
            tilexy = np.loadtxt(csv_files[0], delimiter=",")
            tilexy_per_cycle[cycle] = tilexy
        
            # Determine tile dimensions from the first tile file
            if tile_dims is None and tif_files:
                tile_dims = tifffile.imread(tif_files[0]).shape
                print(f"Tile dimension: {tile_dims}")
        
        # Map channel names to channel indices, and determine order for DO_decorators
        channel_map = {name: idx for idx, name in enumerate(channels)}
        channel_order = [channel_map[ch] for ch in DO_decorators]
        print(f"Channel order: {channel_order}")
        
        # Create primary and auxiliary tile fetchers
        primary_fetcher = ISS2DPrimaryTileFetcher(
            region_directory, cycle_names, channel_order, tilexy_per_cycle, tile_dims, pixel_to_um, tiles_subdir
        )
        aux_fetcher = ISS2DAuxTileFetcher(
            region_directory, cycle_names, nuclei_channel, tilexy_per_cycle, tile_dims, pixel_to_um, tiles_subdir
        )
        
        # Hook to add codebook reference to experiment.json
        def _add_codebook_hook(json_doc):
            json_doc['codebook'] = "codebook.json"
            return json_doc
              
        # --- Step 3: Write experiment.json and other files ---
        write_experiment_json(
            path=SpaceTX_dir,
            fov_count=next(iter(tile_counts.values())),
            tile_format=ImageFormat.TIFF,
            primary_image_dimensions={
                Axes.ROUND: len(cycle_names),
                Axes.CH: len(DO_decorators),
                Axes.ZPLANE: 1,
            },
            aux_name_to_dimensions={
                'nuclei': {
                    Axes.ROUND: len(cycle_names),
                    Axes.CH: 1,
                    Axes.ZPLANE: 1,
                },
            },
            primary_tile_fetcher=primary_fetcher,
            aux_tile_fetcher={'nuclei': aux_fetcher},
            postprocess_func=_add_codebook_hook,
            default_shape={Axes.Y: tile_dims[0], Axes.X: tile_dims[1]}
        )

        # --- Step 4: Build and write the real codebook (overwriting any dummy) ---
        codebook_array = []
        with open(codebook_csv, "r") as f:
            for line in f:
                line = line.rstrip('\n').split(',')
                # Build codeword for each target: one (r, c, value) per round
                codewords = [
                    {Axes.ROUND.value: r, Axes.CH.value: int(code) - 1, Features.CODE_VALUE: 1}
                    for r, code in enumerate(line[1:])
                ]
                codebook_array.append({Features.CODEWORD: codewords, Features.TARGET: line[0]})
        codebook = Codebook.from_code_array(codebook_array)
        codebook_json_path = SpaceTX_dir / "codebook.json"
        codebook.to_json(codebook_json_path)  # This overwrites dummy codebook with the real one
        #print("Codebook shape:", codebook.shape)

        # --- ADDED: Write an XML manifest in the SpaceTX folder (per region) ---
        #
        # Provenance policy requested:
        #   - Only write XML if this run produced outputs for this region (true here because we
        #     passed the early-exit and have written experiment.json/codebook.json).
        #   - Never overwrite: use unique filename with run_id.
        xml_path = SpaceTX_dir / f"spacetx_run_{run_id}.xml"
        root = ET.Element("SpaceTXExperiment", attrib={"region": str(region_name)})

        # Optional: store run_id inside the XML for traceability if filenames are moved.
        root.set("run_id", str(run_id))

        paths_el = ET.SubElement(root, "Paths")
        ET.SubElement(paths_el, "SpaceTXDir").text = str(SpaceTX_dir)
        ET.SubElement(paths_el, "ExperimentJSON").text = "experiment.json"
        ET.SubElement(paths_el, "CodebookJSON").text = "codebook.json"

        meta_el = ET.SubElement(root, "Metadata")
        ET.SubElement(meta_el, "pixel_to_um").text = str(pixel_to_um)
        ET.SubElement(meta_el, "units").text = "pixels" if pixel_to_um == 1 else "microns"
        ET.SubElement(meta_el, "tile_height").text = str(tile_dims[0])
        ET.SubElement(meta_el, "tile_width").text = str(tile_dims[1])
        ET.SubElement(meta_el, "tiles_subdir").text = str(tiles_subdir)

        cycles_el = ET.SubElement(root, "Cycles", attrib={"count": str(len(cycle_names))})
        for c in cycle_names:
            ET.SubElement(cycles_el, "Cycle", attrib={"name": str(c)})

        ch_el = ET.SubElement(root, "Channels")
        ET.SubElement(ch_el, "All").text = ",".join([str(x) for x in channels])
        ET.SubElement(ch_el, "Decoding").text = ",".join([str(x) for x in DO_decorators])
        ET.SubElement(ch_el, "Nuclei").text = "DAPI"

        tiles_el = ET.SubElement(root, "Tiles")
        ET.SubElement(tiles_el, "fov_count").text = str(next(iter(tile_counts.values())))

        tree = ET.ElementTree(root)
        try:
            ET.indent(tree, space="  ", level=0)  # python>=3.9
        except Exception:
            pass
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        print(f" SpaceTX XML written to: {xml_path}")

        # --- END ADDED XML ---

        print(f" SpaceTx formatted files written to: {SpaceTX_dir}")

        # --- Step 5: Backup original JSON files ---
        orig_json_dir = SpaceTX_dir / "original_jsons"
        orig_json_dir.mkdir(exist_ok=True)
        
        for file in SpaceTX_dir.glob("*.json"):
            orig = orig_json_dir / file.name
            # Copy original JSON, stripping path (if needed)
            orig.write_text(file.read_text().replace(str(SpaceTX_dir).replace('\\', '\\\\') + '\\\\', ''))

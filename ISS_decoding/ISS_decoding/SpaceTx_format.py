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

# Represents one image tile with coordinate mapping and pixel data
class ISSTile2D(FetchedTile):
    def __init__(self, file_path, tile, tilexy, tile_dims, pixelscale):
        self.file_path = file_path
        self.tile = tile
        self.tilexy = tilexy
        self.tile_dims = tile_dims
        self.pixelscale = pixelscale

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
            Coordinates.X: (self.tilexy[self.tile, 0] * self.pixelscale,
                            (self.tilexy[self.tile, 0] + self.tile_dims[1]) * self.pixelscale),
            Coordinates.Y: (self.tilexy[self.tile, 1] * self.pixelscale,
                            (self.tilexy[self.tile, 1] + self.tile_dims[0]) * self.pixelscale),
            Coordinates.Z: (0.0, 0.0),
        }

    # Loads the pixel data from file
    def tile_data(self) -> np.ndarray:
        #print(f"Loading tile data from {self.file_path}")
        return imread(self.file_path)


# Fetcher for primary image tiles
class ISS2DPrimaryTileFetcher(TileFetcher):
    def __init__(self, region_directory, cycle_names, channel_order, tilexy, tile_dims, pixelscale):
        self.region_directory = region_directory
        self.cycle_names = cycle_names
        self.channel_order = channel_order
        self.tilexy = tilexy
        self.tile_dims = tile_dims
        self.pixelscale = pixelscale

    # Returns an ISSTile2D object for a given tile, cycle, channel, and z-plane
    def get_tile(self, tile: int, c: int, ch: int, z: int) -> FetchedTile:
        cycle = self.cycle_names[c]
        path = self.region_directory / "preprocessing" / cycle / "4_retiled"
        file_path = path / f"{cycle}_s{tile}_ch{self.channel_order[ch]}.tif"
        #print(f"Fetching primary tile: {file_path}")
        return ISSTile2D(file_path, tile, self.tilexy[cycle], self.tile_dims, self.pixelscale)


# Fetcher for auxiliary tiles 
class ISS2DAuxTileFetcher(TileFetcher):
    def __init__(self, region_directory, cycle_names, nuclei_channel, tilexy, tile_dims, pixelscale):
        self.region_directory = region_directory
        self.cycle_names = cycle_names
        self.nuclei_channel = nuclei_channel
        self.tilexy = tilexy
        self.tile_dims = tile_dims
        self.pixelscale = pixelscale

    # Returns an ISSTile2D object for a given tile, cycle, and z-plane (fixed nuclei channel)
    def get_tile(self, tile: int, c: int, ch: int, z: int) -> FetchedTile:
        nuclei_ch = self.nuclei_channel -1 # change to 0-indexed
        cycle = self.cycle_names[c]
        path = self.region_directory / "preprocessing" / cycle / "4_retiled"
        file_path = path / f"{cycle}_s{tile}_ch{nuclei_ch}.tif"
        #print(f"Fetching aux tile: {file_path}")
        return ISSTile2D(file_path, tile, self.tilexy[cycle], self.tile_dims, self.pixelscale)


# Main function to create SpaceTx-compatible experiment and codebook files for a region
def make_spacetx_format(input_dir, 
                        codebook_csv,
                        pixelscale=0.1625,
                        channels=["DAPI", "Cy3", "Cy5", "AF750", "AF488"],
                        DO_decorators=["AF750", "Cy5", "Cy3", "AF488"],
                        nuclei_channel="DAPI"):
    """
    Prepare a SpaceTx-format experiment directory, tile fetchers, and codebook for ISS/starfish analysis.
    
    Args:
        input_dir (str or Path): Top-level experiment directory containing region folders (e.g., R1, R2).
        codebook_csv (str or Path): Path to the codebook CSV.
        pixelscale (float): Microns per pixel.
        channels (list): List of all channel names.
        DO_decorators (list): List of RNA channels (used for DO calculation).
        nuclei_channel (int): Index of the nuclei channel.
    
    Behavior:
        - Discovers regions and cycles, sets up all tile fetchers.
        - Writes experiment.json and codebook.json for SpaceTx/starfish.
        - Backs up original JSONs.
    """


    input_dir = Path(input_dir)
    print(f"Processing directory: {input_dir}")

    codebook_csv = Path(codebook_csv)
    print('Codebook: ', codebook_csv) 

    nuclei_channel = channels.index("DAPI") + 1 # 1-indexed

    # --- Step 1: Find all region directories matching R\d+ ---
    region_pattern = re.compile(r'^R\d+$')
    region_directories = [r for r in input_dir.iterdir() if r.is_dir() and region_pattern.match(r.name)]
    region_names = [r.name for r in region_directories]
    
    for region_directory, region_name in zip(region_directories, region_names):

        # Create SpaceTx output directory for this region
        SpaceTX_dir = region_directory / "decoding" / "1_SpaceTX_format"
        SpaceTX_dir.mkdir(parents=True, exist_ok=True)

        # ===== EARLY EXIT / MINIMAL WORK BRANCHING =====
        experiment_json_path = SpaceTX_dir / "experiment.json"
        codebook_json_path = SpaceTX_dir / "codebook.json"
        
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
            cycle_dir = region_directory / "preprocessing" / cycle / "4_retiled"

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
            region_directory, cycle_names, channel_order, tilexy_per_cycle, tile_dims, pixelscale
        )
        aux_fetcher = ISS2DAuxTileFetcher(
            region_directory, cycle_names, nuclei_channel, tilexy_per_cycle, tile_dims, pixelscale
        )
        
        # Hook to add codebook reference to experiment.json
        def _add_codebook_hook(json_doc):
            json_doc['codebook'] = "codebook.json"
            return json_doc
              
        # --- Step 3: Write experiment.json and other SpaceTx files ---
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
 
        print(f" SpaceTx formatted files written to: {SpaceTX_dir}")

        # --- Step 5: Backup original JSON files ---
        orig_json_dir = SpaceTX_dir / "original_jsons"
        orig_json_dir.mkdir(exist_ok=True)
        
        for file in SpaceTX_dir.glob("*.json"):
            orig = orig_json_dir / file.name
            # Copy original JSON, stripping path (if needed)
            orig.write_text(file.read_text().replace(str(SpaceTX_dir).replace('\\', '\\\\') + '\\\\', ''))

    
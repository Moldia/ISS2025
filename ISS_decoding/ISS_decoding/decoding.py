# Standard library
import math
import re
from pathlib import Path
from typing import Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET
from datetime import datetime

# NOTE (provenance policy):
#   - We ONLY write a decoding XML if this run actually writes the region-level CSV.
#   - We NEVER overwrite existing XMLs: each productive run writes a uniquely-named XML.
#   - Runs that skip because outputs already exist produce NO XML.

# Third-party
import numpy as np
import pandas as pd

# Starfish
from starfish import Codebook, Experiment, FieldOfView
from starfish.image import ApplyTransform, Filter, LearnTransform
from starfish.spots import DecodeSpots, FindSpots
from starfish.types import Axes, TraceBuildingStrategies, Levels
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match

def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M")
    
def read_spacetx_coordinate_metadata(SpaceTX_dir):
    """
    Read coordinate unit metadata written during SpaceTX generation.

    Returns
    -------
    tuple[str, float | None]
        coordinate_units, coordinate_pixel_to_um
    """
    SpaceTX_dir = Path(SpaceTX_dir)

    xml_files = sorted(SpaceTX_dir.glob("spacetx_run_*.xml"))
    if not xml_files:
        print(
            f"[WARN] No SpaceTX XML metadata found in {SpaceTX_dir}. "
            "Coordinate units unknown."
        )
        return "unknown", None

    xml_path = xml_files[-1]

    try:
        root = ET.parse(xml_path).getroot()
        meta_el = root.find("Metadata")

        if meta_el is None:
            print(f"[WARN] No <Metadata> section found in {xml_path.name}.")
            return "unknown", None

        pixel_to_um_text = meta_el.findtext("pixel_to_um")
        units_text = meta_el.findtext("units")

        coordinate_pixel_to_um = None
        if pixel_to_um_text not in (None, "", "None"):
            coordinate_pixel_to_um = float(pixel_to_um_text)

        coordinate_units = units_text or (
            "pixels" if coordinate_pixel_to_um == 1 else "microns"
        )

        print(
            f"[INFO] Decoding coordinate units from SpaceTX metadata: "
            f"{coordinate_units} "
            f"(pixel_to_um={coordinate_pixel_to_um}, source={xml_path.name})"
        )

        return coordinate_units, coordinate_pixel_to_um

    except Exception as e:
        print(f"[WARN] Could not read SpaceTX coordinate metadata from {xml_path}: {e}")
        return "unknown", None


def QC_score_calc(decoded):
    """
    Compute per-spot quality and second-peak metrics from a DecodedIntensityTable.
    """
    arr = np.array(decoded.values, dtype=float, copy=True)

    if arr.ndim != 3:
        raise ValueError(f"decoded.values must be 3D (spots, rounds, channels), got shape {arr.shape}")

    n_spots, n_rounds, n_channels = arr.shape

    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    totals = arr.sum(axis=2)
    maxes = arr.max(axis=2)

    quality = np.zeros_like(maxes, dtype=float)
    np.divide(maxes, totals, out=quality, where=totals > 0)

    spr = np.zeros_like(maxes, dtype=float)
    if n_channels >= 2:
        part = np.partition(arr, kth=-2, axis=2)
        second = part[:, :, -2]
        top = part[:, :, -1]
        np.divide(second, top, out=spr, where=top > 0)
    else:
        spr.fill(0.0)

    df = decoded.to_features_dataframe()
    df["quality_minimum"] = quality.min(axis=1)
    df["quality_mean"] = quality.mean(axis=1)
    df["quality_all_bases"] = quality.tolist()
    df["second_peak_ratio_min"] = spr.min(axis=1)
    df["second_peak_ratio_mean"] = spr.mean(axis=1)
    df["second_peak_ratio_all_bases"] = spr.tolist()

    return df


def ISS_pipeline(
    tile,
    codebook,
    dense=False,
    register=True,
    register_dapi=True,
    masking_radius=15,
    int_threshold=0.002,
    sigma_vals=(1, 10, 30),
    decode_mode="PRMC",
    channel_normalization="MH",
    spot_detection_mode="starfish",
    spotiflow_model=None,
    prob_threshold=None,
):
    print("Loading image planes")
    primary_image = tile.get_image(FieldOfView.PRIMARY_IMAGES)
    nuclei = tile.get_image("nuclei")

    dots = primary_image.reduce({Axes.CH, Axes.ROUND}, func="max")

    if register:
        if register_dapi:
            ref_round = 1 if dense else 0
            nuclei_ref = nuclei.sel({Axes.ROUND: ref_round, Axes.CH: 0, Axes.ZPLANE: 0})
            print(f"Registering images based on nuclei stain (ROUND={ref_round})")
            learn_translation = LearnTransform.Translation(
                reference_stack=nuclei_ref,
                axes=Axes.ROUND,
                upsampling=1000,
            )
            transforms_list = learn_translation.run(nuclei)
        else:
            print("Creating reference images")
            ref_for_reg = dots if dense else primary_image.reduce({Axes.CH, Axes.ZPLANE}, func="max")
            print("Registering images")
            learn_translation = LearnTransform.Translation(
                reference_stack=ref_for_reg,
                axes=Axes.ROUND,
                upsampling=100,
            )
            run_stack = primary_image.reduce({Axes.CH, Axes.ZPLANE}, func="max")
            transforms_list = learn_translation.run(run_stack)

        warp = ApplyTransform.Warp()
        registered = warp.run(primary_image, transforms_list=transforms_list, in_place=False, verbose=True)

        filt = Filter.WhiteTophat(masking_radius, is_volume=False)
        filtered = filt.run(registered, verbose=True, in_place=False)
    else:
        print("Not registering images, applying filter to raw data")
        filt = Filter.WhiteTophat(masking_radius, is_volume=False)
        filtered = filt.run(primary_image, verbose=True, in_place=False)

    print("Normalizing channel intensities")
    if channel_normalization == "MH":
        sbp = Filter.MatchHistograms({Axes.CH, Axes.ROUND})
    else:
        sbp = Filter.ClipPercentileToZero(
            p_min=80,
            p_max=99.999,
            level_method=Levels.SCALE_BY_CHUNK,
        )

    scaled = sbp.run(filtered, n_processes=1, in_place=False)

    if spot_detection_mode == "starfish":
        min_sigma, max_sigma, num_sigma = sigma_vals
        detector = FindSpots.BlobDetector(
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=int_threshold,
            measurement_type="mean",
        )

        def find_spots(reference_image):
            return detector.run(reference_image=reference_image, image_stack=scaled)

    elif spot_detection_mode == "spotiflow":
        if dense:
            raise ValueError("spot_detection_mode='spotiflow' is not supported with dense=True")
        from spotiflow.starfish import SpotiflowDetector

        detector_kwargs = {"model": spotiflow_model or "general"}
        if prob_threshold is not None:
            detector_kwargs["probability_threshold"] = float(prob_threshold)
        detector = SpotiflowDetector(**detector_kwargs)

        def find_spots(reference_image):
            del reference_image
            return detector.run(scaled, n_processes=1)

    else:
        raise ValueError(
            "spot_detection_mode must be 'starfish' or 'spotiflow', got "
            f"{spot_detection_mode!r}"
        )

    def make_decoder():
        if decode_mode == "PRMC":
            return DecodeSpots.PerRoundMaxChannel(codebook=codebook)
        elif decode_mode == "MD":
            return DecodeSpots.MetricDistance(
                codebook=codebook,
                max_distance=1,
                min_intensity=1,
                metric="euclidean",
                norm_order=2,
                trace_building_strategy=TraceBuildingStrategies.EXACT_MATCH,
            )
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")

    if dense:
        channels = list(primary_image.xarray.coords[Axes.CH].values)
        decoder = make_decoder()
        per_channel_qc = []

        for ch in channels:
            channel_ref = primary_image.sel({Axes.ROUND: 0, Axes.CH: ch, Axes.ZPLANE: 0})
            print(f"Locating spots for channel {ch}")
            spots = find_spots(channel_ref)

            decoded = decoder.run(spots=spots)
            df_qc = QC_score_calc(decoded)
            df_qc["channel"] = ch
            per_channel_qc.append(df_qc)

            _ = build_spot_traces_exact_match(spots)

        return pd.concat(per_channel_qc, ignore_index=True) if per_channel_qc else pd.DataFrame()

    dots = primary_image.reduce({Axes.CH, Axes.ROUND}, func="max")
    dots_max = dots.reduce({Axes.ZPLANE}, func="max")

    print("Locating spots in reference image")
    spots = find_spots(dots_max)

    decoder = make_decoder()
    print(f"Decoding with {decode_mode}")
    decoded = decoder.run(spots=spots)

    _ = build_spot_traces_exact_match(spots)

    return QC_score_calc(decoded)


def process_experiment(
    input_dir=None,
    regions_to_process=None,
    output_dir_prefix=None,
    register=False,
    register_dapi=False,
    masking_radius=15,
    normalization_method="MH",
    decode_mode="PRMC",
    dense=False,
    spot_detection_mode="starfish",
    int_threshold=0.002,
    sigma_vals=[1, 10, 30],
    prob_threshold=None,
    spacetx_dirs: Mapping[str, str | Path] | None = None,
    decoded_dirs: Mapping[str, str | Path] | None = None,
    final_csv_files: Mapping[str, str | Path] | None = None,
    region_names: Sequence[str] | None = None,
    overwrite: bool = False,
):
    """
    Run spot finding and decoding on all tiles/FOVs in each region of an ISS/SpaceTx experiment.

    Coordinate units are read from the SpaceTX XML metadata written during
    SpaceTX generation. They are recorded in the decoded CSV and decoding XML.

    ``spacetx_dirs``, ``decoded_dirs``, and ``final_csv_files`` are per-region
    independent overrides. Explicit paths are used exactly as supplied. With
    all required overrides, ``input_dir`` may be ``None``. Existing per-FOV and
    final outputs are reused by default; ``overwrite=True`` recomputes every
    FOV and replaces the requested final CSV.
    """

    input_dir = Path(input_dir) if input_dir is not None else None
    print(f"Processing directory: {input_dir if input_dir is not None else 'explicit paths'}")

    run_id = timestamp_for_filename()

    if output_dir_prefix is not None:
        output_dir_prefix = Path(output_dir_prefix)
    if decoded_dirs is not None or final_csv_files is not None:
        print("[INFO] Using explicit decoded output mappings")
        if output_dir_prefix is not None:
            print(f"[INFO] Unmapped outputs fall back to: {output_dir_prefix.resolve()}")
    elif output_dir_prefix is not None:
        print(f"[INFO] Using output_dir_prefix: {output_dir_prefix.resolve()}")
    else:
        print("[INFO] Using default output location under each region directory")

    if dense:
        print("DENSE (per-channel) decoding")

    if dense and decode_mode != "PRMC":
        print(f"dense=True → overriding decode_mode='{decode_mode}' to 'PRMC'")
        decode_mode = "PRMC"

    region_pattern = re.compile(r"^R(\d+)$")
    available_paths: dict[str, Path] = {}
    if input_dir is not None:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        discovered = []
        for path in input_dir.iterdir():
            if path.is_dir() and (match := region_pattern.match(path.name)):
                discovered.append((int(match.group(1)), path))
        for _, path in sorted(discovered):
            available_paths[path.name] = path

    explicit_names = []
    for mapping in (spacetx_dirs, decoded_dirs, final_csv_files):
        if mapping:
            explicit_names.extend(str(name) for name in mapping)
    if region_names is not None:
        available_names = [str(name) for name in region_names]
    else:
        available_names = list(available_paths)
        available_names.extend(name for name in explicit_names if name not in available_names)

    if not available_names:
        raise RuntimeError(
            "No regions found. Pass input_dir with R* folders or explicit "
            "region_names/path mappings."
        )

    if regions_to_process is not None:
        if not isinstance(regions_to_process, (list, tuple)):
            raise TypeError("regions_to_process must be a list of 1-based ints, e.g. [1, 2].")
        wanted_numbers = [int(value) for value in regions_to_process]
        if any(value < 1 for value in wanted_numbers):
            raise ValueError(
                f"regions_to_process contains invalid region numbers: {regions_to_process}"
            )
        wanted_names = [f"R{value}" for value in wanted_numbers]
        missing = [name for name in wanted_names if name not in available_names]
        if missing:
            raise FileNotFoundError(
                f"Requested region(s) not found: {missing}. Available regions: {available_names}"
            )
        selected_regions = wanted_names
    else:
        selected_regions = available_names

    all_regions = available_names
    skipped_regions = [r for r in all_regions if r not in selected_regions]

    print(f"[INFO] Regions found on disk ({len(all_regions)}): {all_regions}")
    print(f"[INFO] Regions selected for decoding ({len(selected_regions)}): {selected_regions}")
    if skipped_regions:
        print(f"[INFO] Regions skipped ({len(skipped_regions)}): {skipped_regions}")

    SPOTIFLOW_MODEL = None
    if spot_detection_mode == "spotiflow":
        from spotiflow.model import Spotiflow

        SPOTIFLOW_MODEL = Spotiflow.from_pretrained("general")
    elif spot_detection_mode != "starfish":
        raise ValueError(
            "spot_detection_mode must be 'starfish' or 'spotiflow', got "
            f"{spot_detection_mode!r}"
        )

    for region_name in selected_regions:
        region_directory = available_paths.get(region_name)

        if decoded_dirs is not None and region_name in decoded_dirs:
            decoded_dir = Path(decoded_dirs[region_name])
        elif final_csv_files is not None and region_name in final_csv_files:
            decoded_dir = Path(final_csv_files[region_name]).parent
        elif output_dir_prefix is None:
            if region_directory is None:
                raise ValueError(
                    f"No decoded output path supplied for {region_name}; pass "
                    "decoded_dirs, final_csv_files, output_dir_prefix, or input_dir"
                )
            decoded_dir = region_directory / "decoding" / ("2_decoded_dense" if dense else "2_decoded")
        else:
            decoded_dir = output_dir_prefix / region_name / "decoding" / (
                "2_decoded_dense" if dense else "2_decoded"
            )

        if final_csv_files is not None and region_name in final_csv_files:
            final_csv = Path(final_csv_files[region_name])
        else:
            final_csv = decoded_dir / f"{region_name}_decoded.csv"

        if spacetx_dirs is not None and region_name in spacetx_dirs:
            SpaceTX_dir = Path(spacetx_dirs[region_name])
        elif output_dir_prefix is None:
            if region_directory is None:
                raise ValueError(
                    f"No SpaceTX input supplied for {region_name}; pass spacetx_dirs or input_dir"
                )
            SpaceTX_dir = region_directory / "decoding" / "1_SpaceTX_format"
        else:
            SpaceTX_dir = output_dir_prefix / region_name / "decoding" / "1_SpaceTX_format"
        SpaceTX_dir = SpaceTX_dir.resolve()
        experiment_json = SpaceTX_dir / "experiment.json"
        if not experiment_json.is_file():
            source = "Explicit SpaceTX" if spacetx_dirs and region_name in spacetx_dirs else "SpaceTX"
            raise FileNotFoundError(f"{source} experiment.json not found: {experiment_json}")

        print("=" * 60)
        print(f"\033[1mProcessing region {region_name}\033[0m")
        print(f"Output decoded directory: {decoded_dir}")

        if final_csv.exists() and not overwrite:
            print(f"[{region_name}] Skipping: {final_csv.name} already exists.")
            print(f"  ✔ {final_csv}")
            continue

        decoded_dir.mkdir(parents=True, exist_ok=True)
        final_csv.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and final_csv.exists():
            final_csv.unlink()

        coordinate_units, coordinate_pixel_to_um = read_spacetx_coordinate_metadata(SpaceTX_dir)

        experiment = Experiment.from_json(str(experiment_json))
        tiles = list(experiment.keys())

        csv_files = sorted(decoded_dir.glob("fov_*.csv"))
        print(f"Found {len(csv_files)} output files")

        tiles_done = [f.stem for f in csv_files]
        not_done = sorted(tiles if overwrite else set(tiles) - set(tiles_done))
        print(f"-> {len(not_done)} tiles left to process")

        for tile_id in not_done:
            tile = experiment[tile_id]
            tile_csv = decoded_dir / f"{tile_id}.csv"
            if overwrite and tile_csv.exists():
                tile_csv.unlink()
            print(f"\033[1;90mProcessing tile {tile_id[-3:]} \033[0m")

            df = ISS_pipeline(
                tile,
                experiment.codebook,
                dense=dense,
                register=register,
                register_dapi=register_dapi,
                masking_radius=masking_radius,
                int_threshold=int_threshold,
                sigma_vals=sigma_vals,
                decode_mode=decode_mode,
                channel_normalization=normalization_method,
                spot_detection_mode=spot_detection_mode,
                spotiflow_model=SPOTIFLOW_MODEL,
                prob_threshold=prob_threshold,
            )

            if df is None or df.empty:
                continue

            df["tile"] = tile_id
            df["coordinate_units"] = coordinate_units
            df["coordinate_pixel_to_um"] = coordinate_pixel_to_um

            print(f"Saving per-tile CSV: {tile_csv}")
            df.to_csv(tile_csv, index=False)

        print(f"\nWriting region-level concatenated CSV for {region_name!r} ...")

        wrote_region_csv = False
        csv_paths = sorted(decoded_dir.glob("fov_*.csv"))

        if not csv_paths:
            print(f" No tile CSVs matching 'fov*.csv' found in {decoded_dir}; nothing to concatenate.")
        else:
            dfs = []
            for p in csv_paths:
                df = pd.read_csv(p)
                dfs.append(df)
                print(f"  • Loaded {p.name} ({len(df)} rows)")

            concat = pd.concat(dfs, ignore_index=True)
            concat.insert(0, "cont. spot ids", np.arange(len(concat), dtype=int))

            concat.to_csv(final_csv, index=False)
            wrote_region_csv = True
            print(f" → Wrote {len(concat)} total rows to {final_csv}")

        if wrote_region_csv:
            xml_path = decoded_dir / f"decoding_run_{run_id}.xml"
            root = ET.Element("ISSDecodingRun", attrib={"region": str(region_name)})
            root.set("run_id", str(run_id))

            paths_el = ET.SubElement(root, "Paths")
            ET.SubElement(paths_el, "DecodedDir").text = str(decoded_dir)
            ET.SubElement(paths_el, "SpaceTXDir").text = str(SpaceTX_dir)
            ET.SubElement(paths_el, "ExperimentJSON").text = str(SpaceTX_dir / "experiment.json")
            ET.SubElement(paths_el, "FinalCSV").text = str(final_csv)

            params_el = ET.SubElement(root, "Parameters")
            ET.SubElement(params_el, "dense").text = str(dense)
            ET.SubElement(params_el, "register").text = str(register)
            ET.SubElement(params_el, "register_dapi").text = str(register_dapi)
            ET.SubElement(params_el, "masking_radius").text = str(masking_radius)
            ET.SubElement(params_el, "normalization_method").text = str(normalization_method)
            ET.SubElement(params_el, "decode_mode").text = str(decode_mode)
            ET.SubElement(params_el, "spot_detection_mode").text = str(spot_detection_mode)
            ET.SubElement(params_el, "int_threshold").text = str(int_threshold)
            ET.SubElement(params_el, "sigma_vals").text = ",".join([str(x) for x in sigma_vals])
            ET.SubElement(params_el, "prob_threshold").text = (
                "None" if prob_threshold is None else str(prob_threshold)
            )
            ET.SubElement(params_el, "coordinate_units").text = str(coordinate_units)
            ET.SubElement(params_el, "coordinate_pixel_to_um").text = (
                "None" if coordinate_pixel_to_um is None else str(coordinate_pixel_to_um)
            )

            tiles_el = ET.SubElement(root, "Tiles")
            ET.SubElement(tiles_el, "tiles_total").text = str(len(tiles))
            ET.SubElement(tiles_el, "tiles_done").text = str(len(tiles_done))
            ET.SubElement(tiles_el, "tiles_remaining").text = str(len(not_done))

            tree = ET.ElementTree(root)
            try:
                ET.indent(tree, space="  ", level=0)
            except Exception:
                pass

            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            print(f"[{region_name}] Decoding XML written to: {xml_path}")

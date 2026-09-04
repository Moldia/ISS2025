# Standard library
import json
import re
from pathlib import Path
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
from starfish import Experiment, FieldOfView
from starfish.image import ApplyTransform, Filter, LearnTransform
from starfish.spots import DecodeSpots, FindSpots
from starfish.types import Axes, Features, TraceBuildingStrategies, Levels

from .postcode_adapter import (
    prepare_postcode_inputs,
    postcode_output_to_decoded_table,
    summarize_postcode_output,
)


POSTCODE_COMMIT = "4db68cc5cc398128bcfd97a764bef3c98ee3c583"
POSTCODE_OUTPUT_SCHEMA_VERSION = "1.0"
POSTCODE_DEFAULT_KWARGS = {
    "num_iter": 60,
    "batch_size": 15000,
    "up_prc_to_remove": 99.95,
    "modify_bkg_prior": True,
    "estimate_bkg": True,
    "estimate_additional_barcodes": None,
    "add_remaining_barcodes_prior": 0.05,
    "print_training_progress": True,
    "set_seed": 1,
    "device": "auto",
}


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def effective_postcode_kwargs(overrides=None):
    """Return the complete pinned-decoder settings after applying user overrides."""
    settings = dict(POSTCODE_DEFAULT_KWARGS)
    settings.update(overrides or {})
    return settings


def add_spot_identity(dataframe, region_name, tile_id):
    """Add deterministic region/tile spot identifiers and a stable column order."""
    dataframe = dataframe.copy()
    if Features.SPOT_ID not in dataframe.columns:
        dataframe[Features.SPOT_ID] = np.arange(len(dataframe), dtype=int)

    spot_ids = dataframe[Features.SPOT_ID].copy()
    missing_spot_id = spot_ids.isna()
    if missing_spot_id.any():
        spot_ids = spot_ids.astype(object)
        spot_ids.loc[missing_spot_id] = np.flatnonzero(missing_spot_id)
        dataframe[Features.SPOT_ID] = spot_ids

    def stable_id_part(value):
        if isinstance(value, (float, np.floating)) and value.is_integer():
            return str(int(value))
        return str(value)

    id_parts = spot_ids.map(stable_id_part)
    occurrence = id_parts.groupby(id_parts, sort=False).cumcount()
    duplicated = id_parts.duplicated(keep=False)
    if duplicated.any():
        id_parts.loc[duplicated] = (
            id_parts.loc[duplicated].astype(str)
            + ":"
            + occurrence.loc[duplicated].astype(str)
        )

    dataframe["spot_uid"] = (
        str(region_name) + ":" + str(tile_id) + ":" + id_parts.astype(str)
    )
    dataframe["region"] = str(region_name)
    dataframe["tile"] = str(tile_id)

    leading_columns = [
        "spot_uid",
        "region",
        "tile",
        Features.SPOT_ID,
        Axes.X.value,
        Axes.Y.value,
        Axes.ZPLANE.value,
        "xc",
        "yc",
        "zc",
        "coordinate_units",
        "coordinate_pixel_to_um",
        Features.TARGET,
        "candidate_target",
        "assignment_class",
        Features.PASSES_THRESHOLDS,
        "assignment_probability",
        "best_gene_probability",
        "background_probability",
        "infeasible_probability",
        "nan_probability",
        "second_gene",
        "second_gene_probability",
        "gene_probability_margin",
        "decoder",
    ]
    leading_columns = [column for column in leading_columns if column in dataframe]
    remaining_columns = [
        column for column in dataframe.columns if column not in leading_columns
    ]
    return dataframe.loc[:, leading_columns + remaining_columns]


def _numpy_value(value):
    """Convert NumPy/torch-compatible values into arrays safe for ``np.savez``."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype == object:
        array = array.astype(str)
    return array


def save_postcode_decoder_artifacts(
    output,
    decoded_dir,
    tile_id,
    spot_uids,
    postcode_kwargs=None,
):
    """Save full posterior and fitted PoSTcode state for optional re-analysis."""
    if output is None:
        return None, None

    decoded_dir = Path(decoded_dir)
    posterior_dir = decoded_dir / "posteriors"
    model_dir = decoded_dir / "models"
    posterior_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    posterior_path = posterior_dir / f"{tile_id}.npz"
    posterior_payload = {
        "class_probs": _numpy_value(output["class_probs"]),
        "spot_uid": np.asarray(spot_uids, dtype=str),
        "target_names": np.asarray(output["target_names"], dtype=str),
    }
    for class_name, indices in output["class_ind"].items():
        posterior_payload[f"class_indices_{class_name}"] = np.atleast_1d(
            indices
        ).astype(int)
    np.savez_compressed(posterior_path, **posterior_payload)

    model_path = model_dir / f"{tile_id}.npz"
    model_payload = {
        "postcode_commit": np.asarray(POSTCODE_COMMIT),
        "output_schema_version": np.asarray(POSTCODE_OUTPUT_SCHEMA_VERSION),
        "postcode_kwargs_json": np.asarray(
            json.dumps(
                output.get(
                    "postcode_kwargs",
                    effective_postcode_kwargs(postcode_kwargs),
                ),
                sort_keys=True,
                default=str,
            )
        ),
        "target_names": np.asarray(output["target_names"], dtype=str),
    }
    if "barcodes" in output:
        model_payload["barcodes"] = _numpy_value(output["barcodes"])
    for group_name in ("params", "norm_const"):
        for name, value in output.get(group_name, {}).items():
            model_payload[f"{group_name}_{name}"] = _numpy_value(value)
    np.savez_compressed(model_path, **model_payload)
    return posterior_path, model_path


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


def decode_spots_with_postcode(
    spots,
    codebook,
    prob_threshold=None,
    postcode_kwargs=None,
    return_raw=False,
):
    """Decode Starfish spot traces with PoSTcode while retaining Starfish coordinates and QC."""
    try:
        from postcode.decoding_functions import decoding_function
    except ImportError as exc:
        raise ImportError(
            "PoSTcode is not installed. Install ISS_decoding with the 'postcode' extra "
            "or create the environment from ISS_decoding.yml."
        ) from exc

    inputs = prepare_postcode_inputs(spots, codebook)
    if inputs.spot_intensities.shape[0] == 0:
        empty = inputs.intensity_table.to_features_dataframe()
        empty_columns = {
            Features.TARGET: "object",
            "candidate_target": "object",
            "assignment_class": "object",
            Features.PASSES_THRESHOLDS: "bool",
            "assignment_probability": "float64",
            "best_gene_probability": "float64",
            "background_probability": "float64",
            "infeasible_probability": "float64",
            "nan_probability": "float64",
            "second_gene": "object",
            "second_gene_probability": "float64",
            "gene_probability_margin": "float64",
            "postcode_probability": "float64",
            "postcode_class": "object",
            "decoder": "object",
            "quality_minimum": "float64",
            "quality_mean": "float64",
            "quality_all_bases": "object",
            "second_peak_ratio_min": "float64",
            "second_peak_ratio_mean": "float64",
            "second_peak_ratio_all_bases": "object",
        }
        for column, dtype in empty_columns.items():
            if column not in empty:
                empty[column] = pd.Series(index=empty.index, dtype=dtype)
        return (empty, None) if return_raw else empty

    kwargs = effective_postcode_kwargs(postcode_kwargs)
    output = decoding_function(
        inputs.spot_intensities,
        inputs.barcodes,
        **kwargs,
    )
    output = dict(output)
    output["target_names"] = inputs.target_names
    output["barcodes"] = inputs.barcodes
    output["postcode_kwargs"] = kwargs
    decoded, probabilities, class_names = postcode_output_to_decoded_table(
        output,
        inputs.intensity_table,
        inputs.target_names,
        probability_threshold=prob_threshold,
    )
    assignments = summarize_postcode_output(
        output,
        inputs.target_names,
        probability_threshold=prob_threshold,
    )
    dataframe = QC_score_calc(decoded)
    for column in assignments.columns:
        dataframe[column] = assignments[column].to_numpy()

    # Compatibility aliases retained for callers of the first PoSTcode integration.
    dataframe["postcode_probability"] = probabilities
    dataframe["postcode_class"] = class_names
    dataframe["decoder"] = "postcode"
    return (dataframe, output) if return_raw else dataframe


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
    postcode_kwargs=None,
    return_postcode_raw=False,
):
    decode_mode = decode_mode.upper()
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

    min_sigma, max_sigma, num_sigma = sigma_vals
    bd = FindSpots.BlobDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=int_threshold,
        measurement_type="mean",
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

    def decode_detected_spots(spots):
        if decode_mode == "POSTCODE":
            print("Decoding with PoSTcode")
            return decode_spots_with_postcode(
                spots,
                codebook,
                prob_threshold=prob_threshold,
                postcode_kwargs=postcode_kwargs,
                return_raw=return_postcode_raw,
            )
        decoder = make_decoder()
        print(f"Decoding with {decode_mode}")
        decoded = decoder.run(spots=spots)
        return QC_score_calc(decoded)

    if dense:
        channels = list(primary_image.xarray.coords[Axes.CH].values)
        per_channel_qc = []

        for ch in channels:
            channel_ref = primary_image.sel({Axes.ROUND: 0, Axes.CH: ch, Axes.ZPLANE: 0})
            print(f"Locating spots for channel {ch}")
            spots = bd.run(reference_image=channel_ref, image_stack=scaled)

            df_qc = decode_detected_spots(spots)
            df_qc["channel"] = ch
            per_channel_qc.append(df_qc)

        return pd.concat(per_channel_qc, ignore_index=True) if per_channel_qc else pd.DataFrame()

    dots = primary_image.reduce({Axes.CH, Axes.ROUND}, func="max")
    dots_max = dots.reduce({Axes.ZPLANE}, func="max")

    print("Locating spots in reference image")
    spots = bd.run(reference_image=dots_max, image_stack=scaled)

    return decode_detected_spots(spots)


def process_experiment(
    input_dir,
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
    sigma_vals=(1, 10, 30),
    prob_threshold=None,
    postcode_kwargs=None,
    save_postcode_artifacts=False,
):
    """
    Run spot finding and decoding on all tiles/FOVs in each region of an ISS/SpaceTx experiment.

    Coordinate units are read from the SpaceTX XML metadata written during SpaceTX generation.
    They are recorded in the decoded CSV and decoding XML.
    """

    input_dir = Path(input_dir)
    decode_mode = decode_mode.upper()
    print(f"Processing directory: {input_dir}")

    run_id = timestamp_for_filename()

    if output_dir_prefix is not None:
        output_dir_prefix = Path(output_dir_prefix)
        output_dir_prefix.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Using output_dir_prefix: {output_dir_prefix.resolve()}")
    else:
        print("[INFO] Using default output location under each region directory")

    if dense:
        print("DENSE (per-channel) decoding")

    if dense and decode_mode != "PRMC":
        print(f"dense=True → overriding decode_mode='{decode_mode}' to 'PRMC'")
        decode_mode = "PRMC"

    if save_postcode_artifacts and decode_mode != "POSTCODE":
        raise ValueError(
            "save_postcode_artifacts=True is only valid with decode_mode='POSTCODE'."
        )

    region_pattern = re.compile(r"^R(\d+)$")

    regions_found = []
    for r in input_dir.iterdir():
        if not r.is_dir():
            continue
        m = region_pattern.match(r.name)
        if m:
            regions_found.append((int(m.group(1)), r))

    regions_found.sort(key=lambda t: t[0])

    if not regions_found:
        raise RuntimeError(f"No regions found in {input_dir} (expected folders like R1, R2, ...)")

    available_numbers = [n for n, _ in regions_found]
    available_map = {n: p for n, p in regions_found}

    if regions_to_process is not None:
        if not isinstance(regions_to_process, (list, tuple)):
            raise TypeError("regions_to_process must be a list of 1-based ints, e.g. [1, 2].")

        wanted = [int(x) for x in regions_to_process]
        if any(x < 1 for x in wanted):
            raise ValueError(f"regions_to_process contains invalid region numbers: {regions_to_process}")

        missing = [n for n in wanted if n not in available_map]
        if missing:
            raise FileNotFoundError(
                f"Requested region(s) not found: {[f'R{n}' for n in missing]}. "
                f"Available regions: {[f'R{n}' for n in available_numbers]}"
            )

        region_numbers = wanted
        region_directories = [available_map[n] for n in wanted]
    else:
        region_numbers = available_numbers
        region_directories = [available_map[n] for n in available_numbers]

    all_regions = [f"R{n}" for n in available_numbers]
    selected_regions = [f"R{n}" for n in region_numbers]
    skipped_regions = [r for r in all_regions if r not in selected_regions]

    print(f"[INFO] Regions found on disk ({len(all_regions)}): {all_regions}")
    print(f"[INFO] Regions selected for decoding ({len(selected_regions)}): {selected_regions}")
    if skipped_regions:
        print(f"[INFO] Regions skipped ({len(skipped_regions)}): {skipped_regions}")

    SPOTIFLOW_MODEL = None
    SpotiflowDetector = None

    if spot_detection_mode == "spotiflow":
        from spotiflow.model import Spotiflow
        from spotiflow.starfish import SpotiflowDetector

        SPOTIFLOW_MODEL = Spotiflow.from_pretrained("general")

    for region_directory in region_directories:
        region_name = region_directory.name
        is_postcode = decode_mode == "POSTCODE"
        postcode_settings = (
            effective_postcode_kwargs(postcode_kwargs) if is_postcode else {}
        )

        decoded_subdir = (
            "2_decoded_dense"
            if dense
            else "2_decoded_postcode" if is_postcode else "2_decoded"
        )
        if output_dir_prefix is None:
            decoded_dir = region_directory / "decoding" / decoded_subdir
        else:
            decoded_dir = output_dir_prefix / region_name / "decoding" / decoded_subdir
        decoded_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"\033[1mProcessing region {region_name}\033[0m")
        print(f"Output decoded directory: {decoded_dir}")

        final_stem = (
            f"{region_name}_decoded_postcode" if is_postcode else f"{region_name}_decoded"
        )
        final_csv = decoded_dir / f"{final_stem}.csv"
        final_parquet = decoded_dir / f"{final_stem}.parquet" if is_postcode else None
        if is_postcode and final_parquet.exists():
            if not final_csv.exists():
                pd.read_parquet(final_parquet).to_csv(final_csv, index=False)
                print(f"[{region_name}] Restored missing CSV compatibility copy.")
            print(f"[{region_name}] Skipping: canonical output already exists.")
            print(f"  ✔ {final_parquet}")
            continue
        if not is_postcode and final_csv.exists():
            print(f"[{region_name}] Skipping: {final_csv.name} already exists.")
            print(f"  ✔ {final_csv}")
            continue

        if output_dir_prefix is None:
            SpaceTX_dir = (region_directory / "decoding" / "1_SpaceTX_format").resolve()
        else:
            SpaceTX_dir = (
                output_dir_prefix / region_name / "decoding" / "1_SpaceTX_format"
            ).resolve()

        coordinate_units, coordinate_pixel_to_um = read_spacetx_coordinate_metadata(SpaceTX_dir)

        experiment = Experiment.from_json(str(SpaceTX_dir / "experiment.json"))
        tiles = list(experiment.keys())
        tile_output_dir = decoded_dir / "tiles" if is_postcode else decoded_dir
        tile_output_dir.mkdir(parents=True, exist_ok=True)
        if is_postcode and save_postcode_artifacts:
            (decoded_dir / "posteriors").mkdir(parents=True, exist_ok=True)
            (decoded_dir / "models").mkdir(parents=True, exist_ok=True)
        tile_suffix = ".parquet" if is_postcode else ".csv"
        tile_files = sorted(tile_output_dir.glob(f"fov_*{tile_suffix}"))
        print(f"Found {len(tile_files)} completed tile outputs")

        tiles_done = [path.stem for path in tile_files]
        not_done = sorted(set(tiles) - set(tiles_done))
        print(f"-> {len(not_done)} tiles left to process")

        for tile_id in not_done:
            tile = experiment[tile_id]
            print(f"\033[1;90mProcessing tile {tile_id[-3:]} \033[0m")

            pipeline_result = ISS_pipeline(
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
                prob_threshold=prob_threshold,
                postcode_kwargs=postcode_kwargs,
                return_postcode_raw=save_postcode_artifacts,
            )
            if is_postcode and save_postcode_artifacts:
                df, raw_postcode_output = pipeline_result
            else:
                df = pipeline_result
                raw_postcode_output = None

            if df is None or (df.empty and not is_postcode):
                continue

            df["tile"] = tile_id
            df["coordinate_units"] = coordinate_units
            df["coordinate_pixel_to_um"] = coordinate_pixel_to_um

            if is_postcode:
                df = add_spot_identity(df, region_name, tile_id)
                if df["spot_uid"].duplicated().any():
                    raise ValueError(f"Duplicate spot_uid values detected in tile {tile_id}.")
                if save_postcode_artifacts:
                    save_postcode_decoder_artifacts(
                        raw_postcode_output,
                        decoded_dir,
                        tile_id,
                        df["spot_uid"].to_numpy(),
                        postcode_kwargs=postcode_kwargs,
                    )
                tile_path = tile_output_dir / f"{tile_id}.parquet"
                print(f"Saving per-tile Parquet: {tile_path}")
                df.to_parquet(tile_path, index=False)
            else:
                tile_path = tile_output_dir / f"{tile_id}.csv"
                print(f"Saving per-tile CSV: {tile_path}")
                df.to_csv(tile_path, index=False)

        output_label = "Parquet and CSV" if is_postcode else "CSV"
        print(f"\nWriting region-level concatenated {output_label} for {region_name!r} ...")

        tile_paths = sorted(tile_output_dir.glob(f"fov*{tile_suffix}"))
        wrote_region_output = False
        if not tile_paths:
            print(f" No tile outputs found in {tile_output_dir}; nothing to concatenate.")
        else:
            dfs = []
            for path in tile_paths:
                df = pd.read_parquet(path) if is_postcode else pd.read_csv(path)
                dfs.append(df)
                print(f"  • Loaded {path.name} ({len(df)} rows)")

            concat = pd.concat(dfs, ignore_index=True)
            if is_postcode:
                if concat["spot_uid"].duplicated().any():
                    duplicates = concat.loc[
                        concat["spot_uid"].duplicated(keep=False), "spot_uid"
                    ].unique()
                    raise ValueError(
                        "Duplicate spot_uid values detected across region output: "
                        f"{duplicates[:5].tolist()}"
                    )
                concat.to_parquet(final_parquet, index=False)
                concat.to_csv(final_csv, index=False)
                print(f" → Wrote {len(concat)} rows to {final_parquet.name}")
                print(f" → Wrote CSV compatibility copy to {final_csv.name}")
            else:
                concat.insert(0, "cont. spot ids", np.arange(len(concat), dtype=int))
                concat.to_csv(final_csv, index=False)
                print(f" → Wrote {len(concat)} total rows to {final_csv.name}")
            wrote_region_output = True

        if wrote_region_output:
            completed_tile_files = sorted(tile_output_dir.glob(f"fov*{tile_suffix}"))
            completed_tile_count = len(completed_tile_files)
            remaining_tile_count = len(set(tiles) - {p.stem for p in completed_tile_files})

            xml_path = decoded_dir / f"decoding_run_{run_id}.xml"
            root = ET.Element("ISSDecodingRun", attrib={"region": str(region_name)})
            root.set("run_id", str(run_id))

            paths_el = ET.SubElement(root, "Paths")
            ET.SubElement(paths_el, "DecodedDir").text = str(decoded_dir)
            ET.SubElement(paths_el, "SpaceTXDir").text = str(SpaceTX_dir)
            ET.SubElement(paths_el, "ExperimentJSON").text = str(SpaceTX_dir / "experiment.json")
            ET.SubElement(paths_el, "FinalCSV").text = str(final_csv)
            if is_postcode:
                ET.SubElement(paths_el, "FinalParquet").text = str(final_parquet)

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
            ET.SubElement(params_el, "postcode_kwargs").text = json.dumps(
                postcode_settings, sort_keys=True, default=str
            )
            ET.SubElement(params_el, "save_postcode_artifacts").text = str(
                save_postcode_artifacts
            )
            if is_postcode:
                ET.SubElement(params_el, "postcode_commit").text = POSTCODE_COMMIT
                ET.SubElement(params_el, "output_schema_version").text = (
                    POSTCODE_OUTPUT_SCHEMA_VERSION
                )
            ET.SubElement(params_el, "coordinate_units").text = str(coordinate_units)
            ET.SubElement(params_el, "coordinate_pixel_to_um").text = (
                "None" if coordinate_pixel_to_um is None else str(coordinate_pixel_to_um)
            )

            tiles_el = ET.SubElement(root, "Tiles")
            ET.SubElement(tiles_el, "tiles_total").text = str(len(tiles))
            ET.SubElement(tiles_el, "tiles_done").text = str(completed_tile_count)
            ET.SubElement(tiles_el, "tiles_remaining").text = str(remaining_tile_count)

            tree = ET.ElementTree(root)
            try:
                ET.indent(tree, space="  ", level=0)
            except Exception:
                pass

            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            print(f"[{region_name}] Decoding XML written to: {xml_path}")

            if is_postcode:
                manifest_path = decoded_dir / f"decoding_run_{run_id}.json"
                manifest = {
                    "schema_version": POSTCODE_OUTPUT_SCHEMA_VERSION,
                    "run_id": run_id,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "region": region_name,
                    "decoder": {
                        "name": "postcode",
                        "commit": POSTCODE_COMMIT,
                    },
                    "paths": {
                        "decoded_dir": str(decoded_dir),
                        "spacetx_dir": str(SpaceTX_dir),
                        "experiment_json": str(SpaceTX_dir / "experiment.json"),
                        "final_parquet": str(final_parquet),
                        "final_csv": str(final_csv),
                        "tile_tables": str(tile_output_dir),
                        "posteriors": (
                            str(decoded_dir / "posteriors")
                            if save_postcode_artifacts
                            else None
                        ),
                        "models": (
                            str(decoded_dir / "models")
                            if save_postcode_artifacts
                            else None
                        ),
                    },
                    "parameters": {
                        "register": register,
                        "register_dapi": register_dapi,
                        "masking_radius": masking_radius,
                        "normalization_method": normalization_method,
                        "spot_detection_mode": spot_detection_mode,
                        "int_threshold": int_threshold,
                        "sigma_vals": list(sigma_vals),
                        "prob_threshold": prob_threshold,
                        "postcode_kwargs": json.loads(
                            json.dumps(postcode_settings, default=str)
                        ),
                        "save_postcode_artifacts": save_postcode_artifacts,
                        "coordinate_units": coordinate_units,
                        "coordinate_pixel_to_um": coordinate_pixel_to_um,
                    },
                    "tiles": {
                        "total": len(tiles),
                        "done": completed_tile_count,
                        "remaining": remaining_tile_count,
                    },
                    "rows": len(concat),
                }
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"[{region_name}] Decoding JSON written to: {manifest_path}")

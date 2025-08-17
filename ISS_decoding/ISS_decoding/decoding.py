# Standard library
import math
import re
from pathlib import Path
from typing import Tuple

# Third-party
import numpy as np
import pandas as pd

# Starfish
from starfish import Codebook, Experiment, FieldOfView
from starfish.image import ApplyTransform, Filter, LearnTransform
from starfish.spots import DecodeSpots, FindSpots
from starfish.types import Axes, TraceBuildingStrategies
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match

# Spotiflow
from spotiflow.model import Spotiflow
from spotiflow.starfish import SpotiflowDetector


def QC_score_calc(decoded):
    """
    Compute per-spot quality and second-peak metrics from a DecodedIntensityTable.
    - Robust: handles NaNs, infs, zero totals, and too-few channels.
    - Fast: vectorized over rounds/channels.
    
    Returns
    -------
    pandas.DataFrame
        decoded.to_features_dataframe() with added columns:
        ['quality_minimum', 'quality_mean', 'quality_all_bases',
         'second_peak_ratio_min', 'second_peak_ratio_mean', 'second_peak_ratio_all_bases']
    """
    # Expect shape (n_spots, n_rounds, n_channels)
    arr = np.array(decoded.values, dtype=float, copy=True)

    if arr.ndim != 3:
        raise ValueError(f"decoded.values must be 3D (spots, rounds, channels), got shape {arr.shape}")

    n_spots, n_rounds, n_channels = arr.shape

    # Replace non-finite with 0 (defensive like the first version)
    # This prevents NaNs from propagating into mins/means
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Quality = max(channel)/sum(channel) per (spot, round) ----
    totals = arr.sum(axis=2)                # (n_spots, n_rounds)
    maxes  = arr.max(axis=2)                # (n_spots, n_rounds)

    quality = np.zeros_like(maxes, dtype=float)
    np.divide(maxes, totals, out=quality, where=totals > 0)

    # ---- Second-peak ratio = second_largest / largest per (spot, round) ----
    spr = np.zeros_like(maxes, dtype=float)
    if n_channels >= 2:
        # np.partition is O(n); get the top two values along channels
        part = np.partition(arr, kth=-2, axis=2)  # last two are the largest
        second = part[:, :, -2]
        top    = part[:, :, -1]

        # Guard against top==0 (all zeros) -> ratio 0
        np.divide(second, top, out=spr, where=top > 0)
    else:
        # Fewer than 2 channels => by definition ratio is 0 (matches first version's intent)
        spr.fill(0.0)

    # ---- Aggregate per spot ----
    qc_min  = quality.min(axis=1)   # (n_spots,)
    qc_mean = quality.mean(axis=1)  # (n_spots,)
    qc_all  = quality.tolist()      # list of per-round qualities

    spr_min  = spr.min(axis=1)
    spr_mean = spr.mean(axis=1)
    spr_all  = spr.tolist()

    # ---- Attach to features dataframe ----
    df = decoded.to_features_dataframe()
    df['quality_minimum']             = qc_min
    df['quality_mean']                = qc_mean
    df['quality_all_bases']           = qc_all
    df['second_peak_ratio_min']       = spr_min
    df['second_peak_ratio_mean']      = spr_mean
    df['second_peak_ratio_all_bases'] = spr_all

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
    decode_mode='PRMC',
    channel_normalization='MH',
    spot_detection_mode='starfish',
    spotiflow_model=None,
    prob_threshold=None
):
     
    print('Loading image planes')
    primary_image = tile.get_image(FieldOfView.PRIMARY_IMAGES)
    nuclei = tile.get_image('nuclei')
    
    # --- STEP 1: Create registration reference ---
    dots = primary_image.reduce({Axes.CH, Axes.ROUND}, func="max")
    
    # --- STEP 2: Registration ---
    if register:
        if register_dapi:
            ref_round = 1 if dense else 0
            nuclei_ref = nuclei.sel({Axes.ROUND: ref_round, Axes.CH: 0, Axes.ZPLANE: 0})
            print(f'Registering images based on nuclei stain (ROUND={ref_round})')
            learn_translation = LearnTransform.Translation(reference_stack=nuclei_ref, axes=Axes.ROUND, upsampling=1000)
            transforms_list = learn_translation.run(nuclei)
        else:
            print('Creating reference images')
            ref_for_reg = dots if dense else primary_image.reduce({Axes.CH, Axes.ZPLANE}, func="max")
            print('Registering images')
            learn_translation = LearnTransform.Translation(reference_stack=ref_for_reg, axes=Axes.ROUND, upsampling=100)
            run_stack = primary_image.reduce({Axes.CH, Axes.ZPLANE}, func="max")
            transforms_list = learn_translation.run(run_stack)
    
        warp = ApplyTransform.Warp()
        registered = warp.run(primary_image, transforms_list=transforms_list, in_place=False, verbose=True)
    
        # --- STEP 3: Masking / Filtering ---
        filt = Filter.WhiteTophat(masking_radius, is_volume=False)
        filtered = filt.run(registered, verbose=True, in_place=False)
    else:
        print('Not registering images, applying filter to raw data')
        filt = Filter.WhiteTophat(masking_radius, is_volume=False)
        filtered = filt.run(primary_image, verbose=True, in_place=False)
    
    # --- STEP 4: Channel normalization ---
    print('Normalizing channel intensities')
    if channel_normalization == 'MH':
        sbp = Filter.MatchHistograms({Axes.CH, Axes.ROUND})
    else:
        sbp = Filter.ClipPercentileToZero(
            p_min=80,
            p_max=99.999,
            level_method=Levels.SCALE_BY_CHUNK
        )
    scaled = sbp.run(filtered, n_processes=1, in_place=False)
    
    # --- STEP 5: Spot detection ---
    min_sigma, max_sigma, num_sigma = sigma_vals
    bd = FindSpots.BlobDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=int_threshold,
        measurement_type='mean'
    )

    # --- STEP 6: Decoder factory ---
    def make_decoder():
        if decode_mode == 'PRMC':
            return DecodeSpots.PerRoundMaxChannel(codebook=codebook)
        elif decode_mode == 'MD':
            return DecodeSpots.MetricDistance(
                codebook=codebook,
                max_distance=1,
                min_intensity=1,
                metric='euclidean',
                norm_order=2,
                trace_building_strategy=TraceBuildingStrategies.EXACT_MATCH
            )
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")

    # --- STEP 7: Dense mode (per-channel decoding) ---
    if dense:
        channels = list(primary_image.xarray.coords[Axes.CH].values)
        decoder = make_decoder()
        per_channel_qc = []

        for ch in channels:
            channel_ref = primary_image.sel({Axes.ROUND: 0, Axes.CH: ch, Axes.ZPLANE: 0})
            print(f'Locating spots for channel {ch}')
            spots = bd.run(reference_image=channel_ref, image_stack=scaled)

            decoded = decoder.run(spots=spots)
            df_qc = QC_score_calc(decoded)
            df_qc['channel'] = ch
            per_channel_qc.append(df_qc)

            _ = build_spot_traces_exact_match(spots)

        return pd.concat(per_channel_qc, ignore_index=True) if per_channel_qc else pd.DataFrame()

    # --- STEP 8: Standard mode (single-pass decoding) ---
    dots = primary_image.reduce({Axes.CH, Axes.ROUND}, func="max")
    dots_max = dots.reduce({Axes.ZPLANE}, func="max")

    print('Locating spots in reference image')
    spots = bd.run(reference_image=dots_max, image_stack=scaled)

    decoder = make_decoder()
    print(f'Decoding with {decode_mode}')
    decoded = decoder.run(spots=spots)

    _ = build_spot_traces_exact_match(spots)

    return QC_score_calc(decoded)


def process_experiment(
    input_dir, 
    register=False, 
    register_dapi=False,
    masking_radius=15, 
    normalization_method='MH',  # or other method
    decode_mode='PRMC',
    dense=False,
    spot_detection_mode='starfish',
    int_threshold=0.002, # starfish threshold
    sigma_vals=[1, 10, 30],  # min, max and number for starfish
    prob_threshold=None # spotiflow threshold
    
    
):
    """
    Run spot finding and decoding on all tiles/FOVs in each region of an ISS/SpaceTx experiment.
    Skips already-processed tiles, saves results as per-tile CSVs, and concatenates region-level CSV.

    Args:
        input_dir (str or Path): Path to top-level output directory containing regions (e.g., 'R1', 'R2').
        register (bool): Whether to perform image registration (default True).
        register_dapi (bool): If True, use nuclei (DAPI) for registration; otherwise, use signal max-projection.
        masking_radius (int): Radius for WhiteTophat filtering (spot enhancement).
        threshold (float): Threshold for blob detector in spot finding.
        sigma_vals (list): [min_sigma, max_sigma, num_sigma] for spot finding.
        decode_mode (str): Decoding algorithm to use ('PRMC' or 'MD').
        normalization_method (str): Channel normalization ('MH' for match histograms or 'CPTZ').
        dense (bool): If True, use dense per-channel decoding.

    Workflow:
        - For each region directory (matching 'R\\d+'):
            - Determine which tiles/FOVs are already processed (per-tile CSV exists).
            - For each unprocessed tile:
                - Run spot finding & decoding via ISS_pipeline()
                - Save per-tile decoded CSV
            - Concatenate all per-tile CSVs to a region-level CSV file.

    Notes:
        - Robust to partial completion: skips tiles already processed.
        - Output files are written in a subfolder of each region, named 'decoded' or 'decoded_dense'.
    """
        
     # --- Step 1: Discover all region directories ---
    input_dir = Path(input_dir)
    print(f"Processing directory: {input_dir}")

    if dense:
        print('DENSE (per-channel) decoding')

    if dense and decode_mode != 'PRMC':
        print(f"dense=True → overriding decode_mode='{decode_mode}' to 'PRMC'")
        decode_mode = 'PRMC'
    
    region_pattern = re.compile(r'^R\d+$')
    region_directories = [r for r in input_dir.iterdir() if r.is_dir() and region_pattern.match(r.name)]

    # --- Step 2: Load model once (if detection mode is spotiflow) ---
    if spot_detection_mode == 'spotiflow':
        SPOTIFLOW_MODEL = Spotiflow.from_pretrained("general")
    else:
        SPOTIFLOW_MODEL = None

    # --- Step 3: Process each region directory ---
    for region_directory in region_directories:
        
        region_name = region_directory.name
        decoded_dir = region_directory / 'decoding' / ('2_decoded_dense' if dense else '2_decoded')
        decoded_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"\033[1mProcessing region {region_name}\033[0m")
        print(f"Output decoded directory: {decoded_dir}")

        # --- Step 4: Load SpaceTx experiment metadata ---
        SpaceTX_dir = (region_directory / 'decoding' / '1_SpaceTX_format').resolve()
        experiment = Experiment.from_json(str(SpaceTX_dir / 'experiment.json'))

        tiles = list(experiment.keys())

        # --- Step 5: Find processed tiles ---
        csv_files = sorted(decoded_dir.glob("fov_*.csv"))
        print(f"Found {len(csv_files)} output files")
        
        tiles_done = [f.stem for f in csv_files]
        not_done = sorted(set(tiles) - set(tiles_done))
        print(f"-> {len(not_done)} tiles left to process")

        # --- Step 6: Decode each unprocessed tile ---
        for tile_id in not_done:
            tile = experiment[tile_id]
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
                    channel_normalization=normalization_method
            )
            if df is None or df.empty:
                continue
            df['tile'] = tile_id
            # save per-tile CSV
            print(f"Saving per-tile CSV: {decoded_dir / f'{tile_id}.csv'}")
            df.to_csv(decoded_dir / f"{tile_id}.csv", index=False)
            

        # --- Step 7: Write region-level concatenated CSV by reading per-tile files named "fov*.csv" ---
        print(f"\nWriting region-level concatenated CSV for {region_name!r} ...")
        
        # find all per-tile CSVs that start with "fov" (and end in .csv)
        csv_paths = sorted(decoded_dir.glob("fov*.csv"))
        
        if not csv_paths:
            print(f" No tile CSVs matching 'fov*.csv' found in {decoded_dir}; nothing to concatenate.")
        else:
            dfs = []
            for p in csv_paths:
                df = pd.read_csv(p)
                dfs.append(df)
                print(f"  • Loaded {p.name} ({len(df)} rows)")
        
            # concatenate them
            concat = pd.concat(dfs, ignore_index=True)
            concat.insert(0, 'cont. spot ids', np.arange(len(concat), dtype=int))
        
            out_file = decoded_dir / f"{region_name}_decoded.csv"
            concat.to_csv(out_file, index=False)
            print(f" → Wrote {len(concat)} total rows to {out_file.name}")
        



def plot_starfish_output(spots_file, 
                        dpi = 500, 
                        fig_size = (15,10), 
                        conversion = 0.1625, 
                        size_of_spots = 1):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['text.color'] = 'w'
    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = fig_size
    mpl.rcParams['figure.dpi'] = dpi
    import pandas as pd
    
    df_concat = pd.read_csv(spots_file)
    spots_filt = df_concat[df_concat['target'].notna()]
    
    groups1 = spots_filt.groupby('target')
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    #io.imshow(dapi*10)
    for i, gene in enumerate(sorted(spots_filt.target.unique())):
        group1 = spots_filt[spots_filt.target == gene]
        ax.scatter(group1.xc/0.1625, group1.yc/0.1625, marker='.', linewidth=0, s=size_of_spots, label=gene)

    plt.gca().invert_yaxis()
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=10)
    scalebar = AnchoredSizeBar(ax.transData,
                               615.3846153846, '200 μm', 'lower right', 
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=5,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)
    plt.axis('scaled')
    plt.axis('off')
    plt.title('Starfish decoding' + '\n' + 'Count: ' + str(spots_filt.shape[0]), size = 10)
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def quality_per_gene(
    reads: pd.DataFrame,
    on: str = 'quality_mean',
    gene_name: str = 'target',
    max_genes: int = 50
):
    """
    Violin plot of a given quality score per gene/target,
    ordered by mean quality. Limits to top-N by count to avoid crowding.

    Parameters
    ----------
    reads : pd.DataFrame
        Decoded reads with at least [gene_name, on].
    on : str
        Quality metric to plot (default = 'quality_mean').
    gene_name : str
        Column to group by (default = 'target').
    max_genes : int
        Maximum number of categories to display (default = 50).
    """
    df = reads.copy()
    df[on] = pd.to_numeric(df[on], errors="coerce")

    # Compute means
    ordervals = df.groupby(gene_name)[on].mean().sort_values()

    # Limit to top N (by number of reads)
    top_genes = df[gene_name].value_counts().head(max_genes).index
    df = df[df[gene_name].isin(top_genes)]
    ordervals = ordervals.loc[ordervals.index.intersection(top_genes)]

    # Figure height scales with categories but capped
    height = max(4, min(20, len(ordervals) * 0.3))
    plt.figure(figsize=(6, height))

    sns.violinplot(
        y=gene_name, x=on, data=df,
        order=ordervals.index,
        density_norm="width", inner="box"
    )
    plt.title(f"{on} per {gene_name} (top {len(ordervals)})")
    plt.tight_layout()

    return ordervals

def quality_per_cycle(
    reads,
    cycles: int = 5,
    format_base_quality: bool = False
):
    """
    Violin plot of quality per sequencing cycle.
    """
    if not format_base_quality:
        reads['quality_all_bases'] = reads['quality_all_bases'].str.replace('[', '', regex=False)
        reads['quality_all_bases'] = reads['quality_all_bases'].str.replace(']', '', regex=False)
        quality_per_base = pd.DataFrame(list(reads['quality_all_bases'].str.split(','))).astype(float)
        for col in quality_per_base.columns:
            reads[f'qc_cycle{col+1}'] = quality_per_base[col]

    cycle_cols = [f'qc_cycle{i}' for i in range(1, cycles+1)]
    qualities = reads.loc[:, cycle_cols]

    output = pd.DataFrame({
        'quality': qualities.values.ravel(),
        'cycle': np.repeat(qualities.columns, len(qualities))
    })

    plt.figure(figsize=(cycles * 2, 7))
    sns.violinplot(y='quality', x='cycle', data=output)
    plt.title('Qualities for each cycle')


def compare_scores(
    reads,
    score1: str = 'quality_minimum',
    score2: str = 'quality_mean',
    kind: str = 'kde',
    color: str = '#3266a8',
    format_base_quality: bool = False,
    hue: str = None
):
    """
    Jointplot comparing two quality scores.
    kind options: 'scatter' | 'kde' | 'hist' | 'hex' | 'reg' | 'resid'
    """
    if hue == 'assigned':
        reads['assigned'] = ~reads['target'].isna()

    if not format_base_quality:
        reads['quality_all_bases'] = reads['quality_all_bases'].str.replace('[', '', regex=False)
        reads['quality_all_bases'] = reads['quality_all_bases'].str.replace(']', '', regex=False)
        quality_per_base = pd.DataFrame(list(reads['quality_all_bases'].str.split(','))).astype(float)
        for col in quality_per_base.columns:
            reads[f'qc_base{col+1}'] = quality_per_base[col]

    sns.jointplot(x=score1, y=score2, data=reads, kind=kind, color=color, hue=hue)

def plot_scores(
    reads: pd.DataFrame,
    on: str = 'quality_mean',
    hue: str = None,
    log_scale: bool = False,
    format_base_quality: bool = False,
    palette: str = 'ch:rot=-.25,hue=1,light=.75'
):
    """
    Makes two plots:
      1) Histogram (stacked by hue)
      2) KDE density plot (if hue is provided)
    """
    df = reads.copy()

    # Assign 'assigned' if needed
    if hue == 'assigned':
        df['assigned'] = ~df['target'].isna()

    # Expand base qualities if requested
    if not format_base_quality and 'quality_all_bases' in df:
        df['quality_all_bases'] = df['quality_all_bases'].str.replace('[', '', regex=False)
        df['quality_all_bases'] = df['quality_all_bases'].str.replace(']', '', regex=False)
        quality_per_base = pd.DataFrame(list(df['quality_all_bases'].str.split(','))).astype(float)
        for col in quality_per_base.columns:
            df[f'qc_base{col+1}'] = quality_per_base[col]

    # Ensure numeric
    if on in df.columns:
        df[on] = pd.to_numeric(df[on], errors='coerce')

    # --- Plot 1: histogram ---
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=df, x=on, hue=hue,
        multiple="stack", palette=palette,
        edgecolor=".3", linewidth=.5,
        log_scale=log_scale
    )
    plt.title(f"Histogram of {on}")
    plt.tight_layout()

    # --- Plot 2: KDE density (only if hue provided) ---
    if hue is not None:
        sns.displot(
            data=df,
            x=on, hue=hue,
            kind="kde", height=6,
            multiple="fill",   # <-- change to "layers" or "stack"
            palette=palette,
            alpha=0.7
        )


def plot_frequencies(reads, on: str = 'target', max_categories: int = 50):
    """
    Bar plot of counts per category (e.g. per gene, per target, per FOV).
    Returns a DataFrame with counts.

    Parameters
    ----------
    reads : pd.DataFrame
        Input data containing the column to count.
    on : str
        Column name to count frequencies for.
    max_categories : int
        Maximum number of categories to show (most frequent).
    """
    if on not in reads.columns:
        raise KeyError(f"Column '{on}' not found in reads")

    counts = reads[on].value_counts()

    # Limit to top N categories if too many
    if len(counts) > max_categories:
        counts = counts.head(max_categories)

    plt.figure(figsize=(10, max(4, len(counts) * 0.3)))
    sns.barplot(x=counts.values, y=counts.index, color="steelblue")
    plt.xlabel('counts')
    plt.ylabel(on)
    plt.title(f'Number of each {on} (top {len(counts)})')
    plt.tight_layout()

    return counts.rename_axis(on).reset_index(name='counts')



def plot_expression(
    reads,
    key: str = 'target',
    colorcode: str = "colorblind",
    xcolumn: str = 'xc',
    ycolumn: str = 'yc',
    genes='all',
    size: int = 8,
    background: str = 'white',
    figuresize=(10, 7),
    save: str | None = None,
    fmt: str = 'pdf',
    title_color: str = 'black'
):
    """
    Scatter‐map of reads colored by `key`.
    - genes='all': plot all categories
    - genes='individual': plot each category separately
    - genes=list: plot selected categories
    """
    adataobs = reads.copy()
    sizecols = len(adataobs[key].unique())
    cls = sns.color_palette(colorcode, sizecols)
    cls2 = cls.as_hex()
    colors = dict(zip(adataobs[key].unique(), cls2))

    plt.rcParams['figure.facecolor'] = background

    if genes == 'all':
        cl = adataobs[key]
        plt.figure(figsize=figuresize)
        plt.scatter(
            x=adataobs[xcolumn], y=adataobs[ycolumn],
            c=cl.map(colors), s=size, linewidths=0
        )
        plt.axis('off')
        if save:
            plt.savefig(f"{save}/map_all_genes_{size}_{background}_{key}.{fmt}")

    elif genes == 'individual':
        for each in adataobs[key].unique():
            adatasubobs = adataobs[adataobs[key] == each]
            plt.figure(figsize=figuresize)
            plt.scatter(
                x=adataobs[xcolumn], y=adataobs[ycolumn],
                c='grey', s=size/5, linewidths=0
            )
            plt.scatter(
                x=adatasubobs[xcolumn], y=adatasubobs[ycolumn],
                c=adatasubobs[key].map(colors), s=size, linewidths=0
            )
            plt.axis('off')
            plt.title(f"{each}: {len(adatasubobs)} reads", color=title_color)
            if save:
                plt.savefig(f"{save}/map_individual_cluster_{each}_{size}{background}_{key}.{fmt}")

    else:
        adatasubobs = adataobs[adataobs[key].isin(genes)]
        plt.figure(figsize=figuresize)
        plt.scatter(
            x=adataobs[xcolumn], y=adataobs[ycolumn],
            c='grey', s=size/5, linewidths=0
        )
        plt.scatter(
            x=adatasubobs[xcolumn], y=adatasubobs[ycolumn],
            c=adatasubobs[key].map(colors), s=size, linewidths=0
        )
        plt.axis('off')
        plt.legend()
        if save:
            gene_str = ''.join(map(str, genes))
            plt.savefig(f"{save}/map_group_of_clusters_{gene_str}_{size}{background}_{key}.{fmt}")

    plt.rcParams['figure.facecolor'] = 'white'


from pathlib import Path

def filter_reads(
    reads,
    min_quality_mean=False,
    min_quality_minimum=False,
    max_distance=False,
    max_radius=False,
    min_radius=False,
    min_intensity=False,
    max_intensity=False,
    *,
    save_file: bool = False,
    source_file: str | Path | None = None,
    overwrite: bool = False
):
    """
    Filter reads and optionally save next to the original decoded CSV.

    Parameters
    ----------
    reads : pd.DataFrame
        Input reads dataframe.
    save_file : bool
        If True, save filtered file next to source_file.
    source_file : str or Path
        Path to original decoded CSV (required if save_file=True).
    overwrite : bool
        Overwrite existing file if True.

    Returns
    -------
    pd.DataFrame
        Filtered reads
    """
    readsfilt = reads.copy()

    # --- Filtering ---
    if max_distance:
        readsfilt = readsfilt[readsfilt['distance'] < max_distance]
    if min_quality_mean:
        readsfilt = readsfilt[readsfilt['quality_mean'] > min_quality_mean]
    if min_quality_minimum:
        readsfilt = readsfilt[readsfilt['quality_minimum'] > min_quality_minimum]
    if max_radius:
        readsfilt = readsfilt[readsfilt['radius'] < max_radius]
    if min_radius:
        readsfilt = readsfilt[readsfilt['radius'] > min_radius]
    if min_intensity:
        readsfilt = readsfilt[readsfilt['intensity'] > min_intensity]
    if max_intensity:
        readsfilt = readsfilt[readsfilt['intensity'] < max_intensity]

    # --- Saving ---
    if save_file:
        if source_file is None:
            raise ValueError("source_file must be provided when save_file=True")

        source_file = Path(source_file)

        # Build suffix from active filters
        parts = []
        if min_quality_mean:
            parts.append(f"minqmean{min_quality_mean}")
        if min_quality_minimum:
            parts.append(f"minqmin{min_quality_minimum}")
        if max_distance:
            parts.append(f"maxdist{max_distance}")
        if max_radius:
            parts.append(f"maxrad{max_radius}")
        if min_radius:
            parts.append(f"minrad{min_radius}")
        if min_intensity:
            parts.append(f"minint{min_intensity}")
        if max_intensity:
            parts.append(f"maxint{max_intensity}")

        suffix = "__" + "_".join(parts) if parts else "__filtered"
        out_file = source_file.with_name(f"{source_file.stem}{suffix}.csv")

        if out_file.exists() and not overwrite:
            print(f"Skipping save (exists): {out_file.name}")
        else:
            readsfilt.to_csv(out_file, index=False)
            print(f"Saved filtered reads: {out_file}")

    return readsfilt
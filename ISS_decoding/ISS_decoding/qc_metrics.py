import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def quality_per_gene(
    reads,
    on: str = 'quality_mean',
    gene_name: str = 'target',
    format_base_quality: bool = False
):
    """
    Violin plot of a given quality score per gene, ordered by mean quality.
    Returns the ordered mean-quality Series.
    """
    if not format_base_quality:
        reads['quality_all_bases'] = reads['quality_all_bases'].str.replace('[', '', regex=False)
        reads['quality_all_bases'] = reads['quality_all_bases'].str.replace(']', '', regex=False)
        quality_per_base = pd.DataFrame(list(reads['quality_all_bases'].str.split(','))).astype(float)
        for col in quality_per_base.columns:
            reads[f'qc_base{col+1}'] = quality_per_base[col]

    ordervals = reads.groupby(gene_name).mean()[on].sort_values()
    valsdict = dict(zip(ordervals.index, np.round(ordervals, 2)))
    reads['meangenequality'] = reads[gene_name].map(valsdict)
    reads = reads.sort_values(by='meangenequality')

    plt.figure(figsize=(6, len(valsdict) / 1.2))
    sns.violinplot(y=gene_name, x=on, data=reads)
    plt.title(f"{on} per {gene_name}")
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
    reads,
    on: str = 'quality_mean',
    hue: str = None,
    log_scale: bool = False,
    format_base_quality: bool = False,
    palette: str = 'ch:rot=-.25,hue=1,light=.75'
):
    """
    Histogram (stacked by hue) of a quality score.
    """
    if hue == 'assigned':
        reads['assigned'] = ~reads['target'].isna()

    if not format_base_quality:
        reads['quality_all_bases'] = reads['quality_all_bases'].str.replace('[', '', regex=False)
        reads['quality_all_bases'] = reads['quality_all_bases'].str.replace(']', '', regex=False)
        quality_per_base = pd.DataFrame(list(reads['quality_all_bases'].str.split(','))).astype(float)
        for col in quality_per_base.columns:
            reads[f'qc_base{col+1}'] = quality_per_base[col]

    sns.histplot(
        reads, x=on, hue=hue,
        multiple="stack", palette=palette,
        edgecolor=".3", linewidth=.5,
        log_scale=log_scale
    )

    if hue == 'assigned':
        sns.displot(
            data=reads,
            x=on, hue=hue,
            kind="kde", height=6,
            multiple="fill", clip=(0, None),
            palette=palette
        )


def plot_frequencies(reads, on: str = 'target'):
    """
    Bar plot of counts per category (e.g. per gene or per FOV).
    Returns a DataFrame with counts.
    """
    readssum = reads.groupby(on).count()
    readssum[on] = readssum.index
    readssum = readssum.sort_values(by='fov')

    plt.figure(figsize=(10, len(readssum) / 4))
    plt.title(f'Number of each {on}')
    ax = sns.barplot(x="fov", y=on, data=readssum)
    ax.set(xlabel='counts', ylabel=on)

    subset = readssum.iloc[:, 0:1]
    subset.columns = ['counts']
    return subset


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


def filter_reads(
    reads,
    min_quality_mean=False,
    min_quality_minimum=False,
    max_distance=False,
    max_radius=False,
    min_radius=False,
    min_intensity=False,
    max_intensity=False
):
    """
    Filter reads by thresholds on various columns.
    """
    readsfilt = reads.copy()

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

    return readsfilt

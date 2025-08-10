import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _expand_base_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'quality_all_bases' column once into numeric per-cycle columns:
    qc_cycle1, qc_cycle2, …
    """
    if any(col.startswith('qc_cycle') for col in df.columns):
        return df
    # strip brackets, split on comma, convert to float
    base_q = (
        df['quality_all_bases']
        .str.strip('[]')
        .str.split(',', expand=True)
        .astype(float)
    )
    base_q.columns = [f'qc_cycle{i+1}' for i in base_q.columns]
    return df.join(base_q)


def quality_per_cycle(reads: pd.DataFrame) -> None:
    """
    Violin plot of per-cycle quality.
    """
    df = _expand_base_quality(reads)
    cycle_cols = sorted(c for c in df if c.startswith('qc_cycle'))
    melted = df.melt(
        value_vars=cycle_cols,
        var_name='cycle',
        value_name='quality',
    )
    plt.figure(figsize=(len(cycle_cols)*1.5, 6))
    sns.violinplot(x='cycle', y='quality', data=melted)
    plt.title('Quality per cycle')
    plt.tight_layout()


def quality_per_gene(
    reads: pd.DataFrame,
    score: str = 'quality_mean',
    gene: str = 'target',
) -> pd.Series:
    """
    Violin plot of a given quality score per gene, 
    ordered by mean quality.
    Returns the ordered mean‐quality Series.
    """
    df = reads.copy()
    means = df.groupby(gene)[score].mean().sort_values()
    order = means.index
    plt.figure(figsize=(6, max(4, len(order)*0.2)))
    sns.violinplot(x=score, y=gene, data=df, order=order)
    plt.title(f'{score} per {gene}')
    plt.tight_layout()
    return means


def compare_scores(
    reads: pd.DataFrame,
    score1: str = 'quality_minimum',
    score2: str = 'quality_mean',
    kind: str = 'kde',
    color: str = '#3266a8',
    hue: str = None,
) -> None:
    """
    Jointplot comparing two quality scores.
    """
    df = reads.copy()
    if hue == 'assigned':
        df['assigned'] = df['target'].notna()
        hue = 'assigned'
    sns.jointplot(
        x=score1, y=score2,
        data=df, kind=kind,
        color=color, hue=hue
    )
    plt.tight_layout()


def plot_scores(
    reads: pd.DataFrame,
    score: str = 'quality_mean',
    hue: str = None,
    log_scale: bool = False,
    palette: str = 'ch:rot=-.25,hue=1,light=.75',
) -> None:
    """
    Histogram (stacked by hue) of a quality score.
    """
    df = reads.copy()
    if hue == 'assigned':
        df['assigned'] = df['target'].notna()
        hue = 'assigned'
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=df,
        x=score, hue=hue,
        multiple='stack',
        palette=palette,
        edgecolor='.3',
        linewidth=.5,
        log_scale=log_scale
    )
    plt.tight_layout()


def plot_frequencies(reads: pd.DataFrame, by: str = 'target') -> pd.DataFrame:
    """
    Bar plot of counts per category (e.g. per gene or per FOV).
    Returns a DataFrame with 'counts' and the index = categories.
    """
    counts = reads[by].value_counts().sort_values()
    plt.figure(figsize=(10, max(4, len(counts)*0.2)))
    sns.barplot(x=counts.values, y=counts.index, palette='deep')
    plt.xlabel('counts')
    plt.ylabel(by)
    plt.title(f'Number of each {by}')
    plt.tight_layout()
    return counts.rename_axis(by).reset_index(name='counts')


def plot_expression(
    reads: pd.DataFrame,
    key: str = 'target',
    x: str = 'xc',
    y: str = 'yc',
    genes: list[str] | None = None,
    size: float = 8,
    palette: str = 'colorblind',
    background: str = 'white',
    save: str | None = None,
    fmt: str = 'pdf',
) -> None:
    """
    Scatter‐map of reads colored by `key`. If `genes` list is given,
    plot others in gray and those in `genes` in color.
    """
    df = reads.copy()
    plt.rcParams['figure.facecolor'] = background
    plt.figure(figsize=(10, 7))

    if genes is None:
        sns.scatterplot(
            data=df, x=x, y=y, hue=key,
            palette=palette, s=size, linewidth=0
        )
    else:
        mask = df[key].isin(genes)
        sns.scatterplot(
            data=df[~mask], x=x, y=y,
            color='lightgray', s=size/3, linewidth=0
        )
        sns.scatterplot(
            data=df[mask], x=x, y=y, hue=key,
            palette=palette, s=size, linewidth=0
        )

    plt.axis('off')
    if save:
        suffix = 'all' if genes is None else '_'.join(map(str, genes))
        plt.savefig(f'{save}/map_{suffix}_{key}.{fmt}')

    plt.rcParams['figure.facecolor'] = 'white'


def filter_reads(reads: pd.DataFrame, **criteria) -> pd.DataFrame:
    """
    Filter by named criteria, e.g. min_quality_mean=0.5, max_distance=2.
    """
    df = reads.copy()
    for name, thresh in criteria.items():
        if thresh is False or name not in df.columns:
            continue
        op, col = name.split('_', 1)
        if op == 'min':
            df = df[df[col] > thresh]
        elif op == 'max':
            df = df[df[col] < thresh]
    return df

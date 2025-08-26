# --- Clean imports for AnnData construction ---
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy import ndimage as ndi
from skimage import measure
import scanpy as sc
from typing import Iterable
from math import ceil
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import label
from skimage.segmentation import expand_labels


# -----------------------------
# Helpers
# -----------------------------
def get_object_info(segmentation_labels: np.ndarray) -> pd.DataFrame:
    """Summarize per-object geometry from a labeled segmentation image."""
    props = measure.regionprops(segmentation_labels)
    if not props:
        return pd.DataFrame(columns=["x", "y", "area"]).astype({"x": float, "y": float, "area": float})
    data = {
        "label": [r.label for r in props],
        "x":     [float(r.centroid[1]) for r in props],
        "y":     [float(r.centroid[0]) for r in props],
        "area":  [float(r.area)        for r in props],
    }
    return pd.DataFrame(data).set_index("label").sort_index()


def assign_spots_to_cells(segmentation_labels: np.ndarray, spots: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor (order=0) sampling of the labeled image at spot coordinates."""
    coords = np.vstack([spots["y"].to_numpy(), spots["x"].to_numpy()])
    cell_labels = ndi.map_coordinates(
        segmentation_labels, coords, order=0, mode="constant", cval=0.0, prefilter=False
    ).astype(np.int32)
    out = spots.copy()
    out["cell"] = cell_labels
    return out


def Diff(li1, li2):
    """Symmetric difference of two lists."""
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


# -----------------------------
# AnnData creation
# -----------------------------
    input_dir: str,
    region: str,
    segmentation_method: str,
    dense: bool = True,
    filter_data: bool = True,
    metric: str = "quality_minimum",
    write_h5ad: bool = True,
    value: float = 0.5,
    convert_coords: bool = True,
    conversion_factor: float = 0.1625,
) -> sc.AnnData:
    """Create AnnData from decoded spots and segmentation labels."""
    print(f"\n\033[1mProcessing region: {region}\033[0m")

    decoded_dir = "2_decoded_dense" if dense else "2_decoded"
    spots_path = Path(input_dir) / region / "decoding" / decoded_dir / f"{region}_decoded.csv"
    seg_path   = Path(input_dir) / region / "postprocessing" / "segmentation" / f"{region}_{segmentation_method}_expanded.npz"
    out_dir    = Path(input_dir) / region / "postprocessing" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{region}_anndata_{segmentation_method}.h5ad"

    if write_h5ad and output_file.is_file():
        print(f"AnnData file already exists, reloading: {output_file}")
        return sc.read_h5ad(output_file)

    spots = pd.read_csv(spots_path)
    if filter_data and metric in spots.columns:
        if metric in {"quality_minimum", "quality_mean"}:
            spots = spots[spots[metric] > value]
        elif metric == "distance":
            spots = spots[spots[metric] < value]

    spots = spots[["target", "xc", "yc"]].dropna().copy()
    if convert_coords:
        spots["x"] = spots["xc"] / conversion_factor
        spots["y"] = spots["yc"] / conversion_factor
    else:
        spots["x"], spots["y"] = spots["xc"], spots["yc"]
    spots = spots.rename(columns={"target": "Gene"})[["Gene", "x", "y"]]

    labels = load_npz(seg_path).toarray().astype(np.int32)

    cells_df = get_object_info(labels)
    assigned = assign_spots_to_cells(labels, spots)
    assigned = assigned[assigned["cell"] != 0]

    counts = assigned.groupby(["Gene", "cell"]).size().unstack(fill_value=0)
    counts.columns = counts.columns.astype(int, copy=False)

    present_cell_ids_int = counts.columns
    cells_df = cells_df.reindex(present_cell_ids_int).copy()

    counts_cg = counts.T
    counts_cg.index   = counts_cg.index.astype(str)
    counts_cg.columns = counts_cg.columns.astype(str)
    cells_df.index    = cells_df.index.astype(str)

    ad_sp = sc.AnnData(counts_cg)
    ad_sp.obs = ad_sp.obs.join(cells_df, how="left")
    ad_sp.obs["CellID"] = ad_sp.obs.index.astype(int)
    ad_sp.obsm["spatial"] = ad_sp.obs[["x", "y"]].to_numpy(float)

    if write_h5ad:
        print(f"Writing AnnData: {output_file}")
        ad_sp.write_h5ad(output_file)
    return ad_sp


def concat_anndata(
    regions: Iterable[str],
    input_dir: str | Path,
    segmentation_method: str = "cellpose",
):
    """Concatenate per-region AnnData files into one."""
    adatas = []
    for region in regions:
        f = Path(input_dir) / region / "postprocessing" / "segmentation" / f"{region}_anndata_{segmentation_method}.h5ad"
        if f.is_file():
            print(f"Reading: {f}")
            ad = sc.read(f)
            ad.obs["region_id"] = region
            adatas.append(ad)
    if not adatas:
        raise FileNotFoundError("No AnnData files found.")
    return sc.concat(adatas, index_unique="-", join="outer", fill_value=0)


# -----------------------------
# Clustering & plotting
# -----------------------------
def recluster_specific_cluster(anndata, to_cluster, rerun_umap=False, resolutions=[0.1,0.2,0.3,0.5]):
    """Subset by cluster label and re-run Leiden clustering."""
    sub = anndata[anndata.obs.cell_type.isin([to_cluster])]
    sc.pp.neighbors(sub, n_neighbors=30, n_pcs=30)
    if rerun_umap:
        sc.tl.umap(sub, min_dist=1)
    for r in resolutions:
        sc.tl.leiden(sub, resolution=r, key_added=f"cell_type_{r}")
        sc.pl.umap(sub, color=f"cell_type_{r}", s=30, legend_loc="on data")
    return sub


def plot_umap(anndata, color="cell_type", compute_umap=False, n_neighbors=30, n_pcs=30, min_dist=1):
    """Plot UMAP embedding with flexible recomputation."""
    if compute_umap:
        sc.pp.neighbors(anndata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(anndata, min_dist=min_dist)
    sc.pl.umap(anndata, color=color, s=20, legend_loc="on data", frameon=False)


def plot_marker_genes(anndata, cluster_label, method="t-test", key_added="t-test", n_genes=25):
    """Run rank_genes_groups and plot top marker genes."""
    sc.tl.rank_genes_groups(anndata, cluster_label, method=method, key_added=key_added)
    sc.pl.rank_genes_groups(anndata, n_genes=n_genes, key=key_added)


def plot_clusters(anndata, clusters_to_map, broad_cluster, sample_id_column="sample_id", key="t-test"):
    """Plot all clusters across samples, highlighting marker genes."""
    for broad in sorted(anndata.obs[broad_cluster].unique()):
        ad_b = anndata[anndata.obs[broad_cluster] == broad]
        for cluster in sorted(ad_b.obs[clusters_to_map].unique().astype(int)):
            genes = list(sc.get.rank_genes_groups_df(ad_b, group=str(cluster), key=key)["names"].head(10))
            print(f"Cluster {cluster} in {broad}: {' '.join(genes)}")


def plot_all_clusters(ad, cluster="leiden", spot_size=100, sample_id_col="sample_id"):
    """Plot cluster spatial localization across regions."""
    fig, axs = plt.subplots(3, ceil(len(ad.obs[sample_id_col].unique())/3), figsize=(20,10))
    axs = axs.ravel()
    for i, region in enumerate(sorted(ad.obs[sample_id_col].unique())):
        sc.pl.spatial(ad[ad.obs[sample_id_col]==region], color=cluster, spot_size=spot_size, ax=axs[i], show=False)
    plt.show()


def plot_specific_cluster(
    anndata, 
    clusters_to_map, 
    broad_cluster,
    cluster, 
    cluster_label_type=int,
    key="t-test", 
    size=0.5,
    number_of_marker_genes=10, 
    region_id_column="region_id", 
    dim_subplots=[3, 3]
): 
    """
    Plot a specific cluster across regions, highlighting its marker genes
    and spatial distribution.
    """
    # Dark theme
    mpl.rcParams["text.color"] = "w"
    plt.style.use("dark_background")

    for broad in sorted(anndata.obs[broad_cluster].unique()): 
        anndata_broad = anndata[anndata.obs[broad_cluster] == broad]

        print(f"\nMarker genes for cluster {cluster} in {broad}:")
        genes = list(
            sc.get.rank_genes_groups_df(
                anndata_broad, group=str(cluster), key=key
            )["names"].head(number_of_marker_genes)
        )
        print(" ".join(genes))

        # Cells of this cluster
        spatial_int = anndata_broad[anndata_broad.obs[clusters_to_map] == str(cluster)]

        # Subplot setup
        n_regions = len(anndata_broad.obs[region_id_column].unique())
        n_cols = dim_subplots[1]
        n_rows = ceil(n_regions / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.001)
        fig.suptitle(f"Cluster: {cluster}")

        axs = axs.ravel()

        for q, region_id in enumerate(sorted(anndata_broad.obs[region_id_column].unique())):
            region_data = anndata[anndata.obs[region_id_column] == region_id]
            cluster_data = spatial_int[spatial_int.obs[region_id_column] == region_id]

            axs[q].plot(region_data.obs.x, region_data.obs.y,
                        marker="s", linestyle="", ms=size,
                        color="grey", alpha=0.2)
            axs[q].plot(cluster_data.obs.x, cluster_data.obs.y,
                        marker="s", linestyle="", ms=size,
                        color="yellow")
            axs[q].set_title(str(region_id))
            axs[q].axis("scaled")
            axs[q].axis("off")

        plt.show()


# -----------------------------
# Neighborhood & tiling
# -----------------------------
from sklearn.metrics.pairwise import euclidean_distances
from skimage.segmentation import expand_labels
from skimage.measure import label

def spatial_neighborhood(anndata, cluster_label="leiden_0.5", max_distance_allowed=300, leiden_resolution=0.2):
    """Construct neighborhood graph based on Euclidean distances in XY."""
    coords = np.array([anndata.obs["x"], anndata.obs["y"]]).T
    distances = euclidean_distances(coords, coords)
    dist_binary = ((distances < max_distance_allowed) & (distances != 0)).astype(int)
    ad = sc.AnnData(pd.DataFrame(dist_binary, index=anndata.obs.index, columns=anndata.obs.index))
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=leiden_resolution, key_added="local_neighborhood")
    return ad


def create_ann_tiles(sample_path, segmentation_folder="/cell_segmentation/", expand=True, expand_distance=30):
    """Build AnnData object from per-tile segmentation + decoded spots."""
    path = Path(sample_path) / segmentation_folder
    spots = pd.read_csv(Path(sample_path)/"decoded.csv").dropna()
    adatas = []
    for f in sorted(path.glob("tile*.npz")):
        image = load_npz(f).toarray()
        if len(np.unique(image)) == 1:
            continue
        if expand:
            labels = label(expand_labels(image, expand_distance))
        else:
            labels = label(image)
        cells = get_object_info(labels)
        assigned = assign_spots_to_cells(labels, spots)
        hm = assigned.groupby(["target","cell"]).size().unstack(fill_value=0)
        hm = hm.drop(columns=0, errors="ignore")
        ad = sc.AnnData(hm.T)
        ad.obs = cells
        adatas.append(ad)
    return sc.concat(adatas)


def add_fov_number(spots_file, tile_pos_file, tile_size=2000, conversion_factor=0.1625, new_tile_column="fov_2000"):
    """Annotate decoded spots with FOV/tile number based on XY coordinates."""
    spots = pd.read_csv(spots_file)
    tile_pos = pd.read_csv(tile_pos_file, header=None)
    spots["x_pixels"] = spots["xc"] / conversion_factor
    spots["y_pixels"] = spots["yc"] / conversion_factor
    df_list = []
    for i, (x,y) in tile_pos.iterrows():
        cut = spots[(spots.x_pixels>x) & (spots.x_pixels<x+tile_size) &
                    (spots.y_pixels>y) & (spots.y_pixels<y+tile_size)]
        cut[new_tile_column] = i
        df_list.append(cut)
    return pd.concat(df_list)


# -----------------------------
# PCIseq support
# -----------------------------
def pciseq_anndata(cellData_file, geneData_file, mostProbable_file, output, write_ann=True):
    """Build AnnData from PCIseq outputs (cellData, geneData, mostProbable)."""
    cellData = pd.read_json(cellData_file)
    geneData = pd.read_json(geneData_file)
    mostProbable = pd.read_csv(mostProbable_file)
    hm = geneData.groupby(["Gene","neighbour"]).size().unstack(fill_value=0)
    hm = hm.drop(columns=0, errors="ignore")
    ad = sc.AnnData(hm.T)
    ad = ad[ad.obs.index.astype(int).isin(cellData.Cell_Num)]
    ad.obs = cellData[cellData.Cell_Num.isin(ad.obs.index.astype(int))]
    ad.obs["MostProbableCellType"] = mostProbable["ClassName"]
    ad.obs["Prob"] = mostProbable["Prob"]
    if write_ann:
        ad.write(output)
    return ad

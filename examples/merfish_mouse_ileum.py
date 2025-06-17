import numpy as np
from pathlib import Path
import subprocess
import hashlib
import spatialdata as sd
from dask_image.imread import imread
import pandas as pd
from anndata import AnnData
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point
from napari_spatialdata import Interactive

# download the data: https://datadryad.org/dataset/doi:10.5061/dryad.jm63xsjb2
out_path = Path(__file__).parent.parent / "out"
download_path = out_path / "data_release_baysor_merfish_gut.zip"
unzipped_path = out_path / "data_release_baysor_merfish_gut"

# download the example data
CHECKSUM_DOWNLOAD = "501a206666b5895e9182245dda8d4e60"
#
if (
    not download_path.exists()
    or CHECKSUM_DOWNLOAD != hashlib.md5(download_path.read_bytes()).hexdigest()
):
    print(
        "Data missing or wrong checksum. Please download the data from "
        "https://datadryad.org/dataset/doi:10.5061/dryad.jm63xsjb2"
    )

# unzip the downloaded file
if not unzipped_path.exists():
    subprocess.run(
        f'unzip -o "{download_path}" -d "{out_path}"', shell=True, check=True
    )

# parse raw images
data = imread(unzipped_path / "raw_data" / "dapi_stack.tif")
dapi_stack = sd.models.Image2DModel.parse(data)

data = imread(unzipped_path / "raw_data" / "membrane_stack.tif")
membrane_stack = sd.models.Image2DModel.parse(data)

# parse transcripts locations
points_path = unzipped_path / "raw_data" / "molecules.csv"
df = pd.read_csv(points_path)
molecules = sd.models.PointsModel.parse(
    df, coordinates={"x": "x_pixel", "y": "y_pixel", "z": "z_pixel"}
)

sdata = sd.SpatialData.init_from_elements(
    {
        "dapi": dapi_stack,
        "membrane": membrane_stack,
        "molecules": molecules,
    }
)

# parse cellpose segmentation (cell centroids, counts, cluster assignment)
df_coords = pd.read_csv(
    unzipped_path / "data_analysis/cellpose/segmentation/cell_coords.csv"
)
df_counts = pd.read_csv(
    unzipped_path / "data_analysis/cellpose/segmentation/segmentation_counts.tsv",
    sep="\t",
)
df_cluster = pd.read_csv(
    unzipped_path / "data_analysis/cellpose/clustering/cell_assignment.csv",
)

x = df_counts.iloc[:, range(1, df_counts.shape[1])].values.T
cell_ids = np.arange(1, x.shape[0] + 1)
assert np.array_equal(df_cluster["cell"], cell_ids)
var_name = df_counts.iloc[:, 0]
obs = pd.DataFrame({"cluster": df_cluster["leiden_final"]})
adata = AnnData(X=x, var=pd.DataFrame(index=var_name), obs=obs)
adata.obs["region"] = "cells_centroids_cellpose"
adata.obs["region"] = adata.obs["region"].astype("category")
adata.obs["cell_id"] = cell_ids
adata = sd.models.TableModel.parse(
    adata,
    region="cells_centroids_cellpose",
    region_key="region",
    instance_key="cell_id",
)
df_coords.index = cell_ids
cells = sd.models.PointsModel.parse(df_coords)

sdata["cells_centroids_cellpose"] = cells
sdata["gene_expression_cellpose"] = adata

# parse cellpose segmentation (dapi and membrane labels)
data = imread(
    unzipped_path / "data_analysis/cellpose/cell_boundaries/results/cellpose_dapi.tif"
)
dapi_labels = sd.models.Labels3DModel.parse(data)
data = imread(
    unzipped_path
    / "data_analysis/cellpose/cell_boundaries/results/cellpose_membrane.tif"
)
membrane_labels = sd.models.Labels3DModel.parse(data)
sdata["dapi_labels"] = dapi_labels
sdata["membrane_labels"] = membrane_labels

##
# parse baysor segmentation (2.5D shapes, segmentation cell stats, counts"
df_segmentation = pd.read_csv(
    unzipped_path / "data_analysis/baysor/segmentation/segmentation.csv"
)
df_cell_stats = pd.read_csv(
    unzipped_path / "data_analysis/baysor/segmentation/segmentation_cell_stats.csv"
)
df_counts = pd.read_csv(
    unzipped_path / "data_analysis/baysor/segmentation/segmentation_counts.tsv",
    sep="\t",
)

x = df_counts.iloc[:, range(1, df_counts.shape[1])].values.T
cell_ids = np.arange(1, x.shape[0] + 1)
assert np.array_equal(df_cell_stats["cell"], cell_ids)

adata = AnnData(
    X=x,
    var=pd.DataFrame(index=df_counts.iloc[:, 0]),
    obs=pd.DataFrame({"cell_id": cell_ids, "region": "cells_circles_baysor"}),
)
adata.obs["region"] = adata.obs["region"].astype("category")
adata = sd.models.TableModel.parse(
    adata, region="cells_circles_baysor", region_key="region", instance_key="cell_id"
)

##
adata.obs = pd.merge(
    adata.obs,
    df_cell_stats.drop(columns=["x", "y"], axis=1),
    left_on="cell_id",
    right_on="cell",
    how="left",
).drop(columns=["cell"], axis=1)
##
xy = df_cell_stats[["x", "y"]].values
radii = (df_cell_stats["area"].to_numpy() / np.pi) ** 0.5
gdf = GeoDataFrame(
    {"radius": radii},
    geometry=GeoSeries([Point(xy[i, 0], xy[i, 1]) for i in range(len(xy))]),
    index=df_cell_stats["cell"],
)
print(
    "Baysor segmentation: {}/{} cells have NaN area; dropping them".format(
        np.sum(df_cell_stats["area"].isna()), len(df_cell_stats)
    )
)
gdf = gdf[~gdf.radius.isna()]
gdf = sd.models.ShapesModel.parse(gdf)

# note, the transcripts from baysor have the same coordinates and order as the
# raw transcripts, but since we have 2 different baysor segmentations, we keep all of
# them in separate objects
assert np.array_equal(molecules["x"].compute(), df_segmentation["x"])
assert np.array_equal(molecules["y"].compute(), df_segmentation["y"])
assert np.allclose(molecules["z"].compute(), df_segmentation["z"])

points = sd.models.PointsModel.parse(df_segmentation)

sdata["gene_expression_baysor"] = adata
sdata["cells_circles_baysor"] = gdf
sdata["molecule_baysor"] = points

# TODO: here parse "poly_per_z.json"

# we could also parse "baysor_membrane_prior". It is analogous to the above except that
# "poly_per_z.json" is missing

##

Interactive(sdata)

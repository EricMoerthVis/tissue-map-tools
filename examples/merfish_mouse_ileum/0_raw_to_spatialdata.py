"""
Parse the raw data into a SpatialData object.
"""

import numpy as np
from pathlib import Path
import subprocess
import hashlib
import spatialdata as sd
from dask_image.imread import imread
import pandas as pd
from anndata import AnnData
import geopandas as gpd
from shapely import Point, Polygon
from scipy.optimize import curve_fit
import dask.array as da
from spatialdata.models import points_dask_dataframe_to_geopandas
from geopandas import sjoin
from geopandas import GeoDataFrame

# download the data: https://datadryad.org/dataset/doi:10.5061/dryad.jm63xsjb2
out_path = Path(__file__).parent.parent.parent / "out"
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

dapi_data = imread(unzipped_path / "raw_data" / "dapi_stack.tif")
dapi_data = da.reshape(dapi_data, (1, *dapi_data.shape))

membrane_data = imread(unzipped_path / "raw_data" / "membrane_stack.tif")
membrane_data = da.reshape(membrane_data, (1, *membrane_data.shape))

# the data is in the format (channel, z, y, x)
data = da.concatenate([dapi_data, membrane_data], axis=0)
img_stack = sd.models.Image3DModel.parse(
    data, scale_factors=[2, 2], c_coords=["DAPI", "Membrane"]
)

# parse transcripts locations
points_path = unzipped_path / "raw_data" / "molecules.csv"
df = pd.read_csv(points_path)
molecules = sd.models.PointsModel.parse(
    df,
    coordinates={"x": "x_pixel", "y": "y_pixel", "z": "z_pixel"},
    feature_key="gene",
)


def affine(x, a, b):
    return a * x + b


# infer the pixel size data for the image data using the molecules data
# this is explained in the README.txt (and can and easily seen from the data)
def z_raw_to_layer_index(z_raw: float) -> float:
    return (z_raw - 2.5) / 1.5 + 1


def layer_index_to_z_raw(layer_index: float) -> float:
    return (layer_index - 1) * 1.5 + 2.5


# quick sanity-check
assert 3 == z_raw_to_layer_index(layer_index_to_z_raw(3))

z_pixels_values = df.z_pixel.value_counts().sort_index().index.tolist()
(a, b), _ = curve_fit(affine, z_pixels_values, list(range(1, 10)))

df = df.assign(layer=lambda df: affine(x=df["z_pixel"], a=a, b=b))
df["layer"] = df["layer"].round(0).astype(int)

affine_correct_z_pixel_raster = sd.transformations.Affine(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1 / a, -b / a],
        [0, 0, 0, 1],
    ],
    input_axes=("x", "y", "z"),
    output_axes=("x", "y", "z"),
)
# (a_z, b_z), _ = curve_fit(affine, df.z_pixel, df.z_um)
# (a_y, b_y), _ = curve_fit(affine, df.y_pixel, df.y_um)
# (a_x, b_x), _ = curve_fit(affine, df.x_pixel, df.x_um)
#
# pixel_to_um = sd.transformations.Affine(
#     [[a_x, 0, 0, b_x], [0, a_y, 0, b_y], [0, 0, a_z, b_z], [0, 0, 0, 1]],
#     input_axes=("x", "y", "z"),
#     output_axes=("x", "y", "z"),
# )
# # to be more accurate we could reconstruct the matrix from scratch as above
# um_to_pixel = pixel_to_um.inverse()

sd.transformations.set_transformation(
    img_stack,
    transformation=affine_correct_z_pixel_raster,
    to_coordinate_system="global",
)
sdata = sd.SpatialData.init_from_elements(
    {
        "stains": img_stack,
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
dapi_labels = sd.models.Labels3DModel.parse(
    data,
    transformations={"global": affine_correct_z_pixel_raster},
    scale_factors=[2, 2],
)
data = imread(
    unzipped_path
    / "data_analysis/cellpose/cell_boundaries/results/cellpose_membrane.tif"
)
membrane_labels = sd.models.Labels3DModel.parse(
    data,
    transformations={"global": affine_correct_z_pixel_raster},
    scale_factors=[2, 2],
)
# problem in the data: the same cell across dapi_labels and membrane_labels have
# different index value
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
gdf = gpd.GeoDataFrame(
    {"radius": radii},
    geometry=gpd.GeoSeries([Point(xy[i, 0], xy[i, 1]) for i in range(len(xy))]),
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

points = sd.models.PointsModel.parse(df_segmentation, feature_key="gene")
points["cell"] = points["cell"].round(0).astype(int)

sdata["gene_expression_baysor"] = adata
sdata["cells_circles_baysor"] = gdf
sdata["molecule_baysor"] = points

# TODO: here parse "poly_per_z.json"

# we could also parse "baysor_membrane_prior". It is analogous to the above except that
# "poly_per_z.json" is missing

##
path = unzipped_path / "data_analysis/baysor/segmentation/poly_per_z.json"
# the poly_per_z.json file seems to be using a legacy format and it's not geojson, see
# more here: https://github.com/kharchenkolab/Baysor/issues/129
# newer versions of Baysor use GeoJSON:
# https://github.com/kharchenkolab/Baysor/blob/master/CHANGELOG.md#071--2024-11-19
# this works for GeoJSON files:
# polygons = gpd.read_file("GeoJSON:" + str(path))


# let's parse the JSON file manually
def geometry_from_dict(d):
    geom_type = d.get("type")
    coords = d.get("coordinates")
    if geom_type == "Polygon":
        return Polygon(coords[0])
    # untested:
    # elif geom_type == 'MultiPolygon':
    #     return MultiPolygon([Polygon(p[0]) for p in coords])
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")


df = pd.read_json(path)
##
shapes_per_layer = {}
skipped = 0
for row in df.itertuples():
    # print(row._fields)  # gives ('Index', 'z_id', 'geometries', 'type')
    gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([geometry_from_dict(shape) for shape in row.geometries]),
    )
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.buffer(0))
    gdf["layer"] = row.z_id
    # some geometries are empty (I haven't checked if this happens in the raw data or
    # after calling .buffer(0))
    mask = gdf.geometry.apply(lambda geom: not geom.is_empty)
    gdf = gdf[mask]
    gdf = sd.models.ShapesModel.parse(gdf)
    shapes_per_layer[row.z_id] = gdf
    print(f"Layer {row.z_id}: {len(gdf)} shapes, skipped {skipped} empty geometries")
##

##
for layer, shapes in shapes_per_layer.items():
    sdata[f"cells_layer_{layer}_baysor"] = shapes

##
# 1-based indexing
z_layer = (
    points["z_raw"]
    .apply(z_raw_to_layer_index, meta=("z_raw", "float64"))
    .astype(int)
    .compute()
)
points["layer"] = z_layer

for layer in list(shapes_per_layer.keys()):
    shapes_in_layer = shapes_per_layer[layer]
    points_in_layer = points[points["layer"] == layer]
    # sjoin to transfer the column ('cell') from the points to the shapes

    points_geopandas = points_dask_dataframe_to_geopandas(points_in_layer)

    joined = sjoin(
        left_df=shapes_in_layer,
        right_df=points_geopandas,
        how="left",
        predicate="contains",
    )["cell"]
    # cells with no points gets set to 0 (background)
    joined = joined.fillna(0).astype(int)
    df = pd.DataFrame({"shape": joined.index, "assigned_cell": joined.tolist()})
    # group by 'shape' and get the most frequent 'assigned_cell' for each shape
    most_abundant = (
        df.groupby("shape")["assigned_cell"]
        # get the first value in case of a tie
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"assigned_cell": "most_abundant_cell"})
    )
    most_abundant.set_index("shape", inplace=True)
    shapes_in_layer["most_abundant_cell"] = most_abundant["most_abundant_cell"]
    # ##
    # # debug
    # some asserts in the for loop will fail! this because for some reasons points
    # that are at the border of shapes are selected by "contains" in sjoin, but fail
    # the .contains() check below. This is fine.
    # for shape_iloc in range(1000):
    #     cell = shapes_in_layer.iloc[shape_iloc].most_abundant_cell.item()
    #     if cell == 0:
    #         print(f"Skipping shape (iloc) {shape_iloc} with cell {cell}")
    #     else:
    #         point_iloc = np.where(
    #             points_geopandas.cell == cell
    #         )[0][0]
    #         print(point_iloc)
    #         points_geopandas.iloc[point_iloc]
    #         assert shapes_in_layer.iloc[shape_iloc].geometry.contains(
    #             points_geopandas.iloc[point_iloc].geometry
    #         )
    #         polygon = shapes_in_layer.iloc[shape_iloc].geometry
    #         point = points_geopandas.iloc[point_iloc].geometry
    #         import matplotlib.pyplot as plt
    #         x, y = polygon.exterior.xy
    #         plt.plot(x, y, 'b-', linewidth=2, label='Polygon')
    #         plt.fill(x, y, color='blue', alpha=0.2)
    #         plt.plot(point.x, point.y, 'ro', label='Point')
    #         plt.show()
    #
    # ##
    pass

# shapes_per_layer[2]
# merge the shapes_per_layer into a single GeoDataFrame
gdf_all_layers = pd.concat(shapes_per_layer.values(), ignore_index=True)
gdf_all_layers = gpd.GeoDataFrame(gdf_all_layers, geometry="geometry")

##
# code from owkin hackathon


def match_cells_iomin(
    layer1: GeoDataFrame, layer2: GeoDataFrame, threshold: float
) -> dict[int, list[int]]:
    """
    Matches polygons between two layers based on Intersection over Minimum Area (IoMin).

    Parameters
    ----------
    layer1
        The first set of polygons.
    layer2
        The second set of polygons.
    threshold
        IoMin threshold for considering polygons as matching.

    Returns
    -------
    dict[int, list[int]]
        Dictionary where keys are indices from `layer1` and values are lists of matching indices from `layer2`.
    """
    matched_indices: dict[int, list[int]] = {}
    indices_layer1, indices_layer2 = list(
        layer2.sindex.query(layer1["geometry"], predicate="intersects")
    )

    # Map the positional indices to label-based indices
    label_indices_layer1 = layer1.index[indices_layer1]
    label_indices_layer2 = layer2.index[indices_layer2]

    for i in range(len(label_indices_layer1)):
        idx1 = label_indices_layer1[i]
        idx2 = label_indices_layer2[i]
        poly1 = layer1.loc[idx1].geometry
        poly2 = layer2.loc[idx2].geometry

        # Compute overlap area and mean area
        overlap_area = poly1.intersection(poly2).area
        min_area = min(poly1.area, poly2.area)

        # Calculate IoMean and apply threshold
        iomax = overlap_area / min_area
        if iomax >= threshold:
            matched_indices.setdefault(idx1, []).append(idx2)

    return matched_indices


def process_pseudo3D_shapes(
    gdf: GeoDataFrame, threshold: float = 0.3, cell_column: str = "cell_id"
) -> GeoDataFrame:
    """
    Combines layers in a GeoDataFrame, assigning unique cell indices based on
    spatial matches between polygons in adjacent layers.

    Parameters
    ----------
    gdf
        GeoDataFrame with layers to process.
    threshold
        IoMean threshold for matching polygons between layers.
    cell_column
        Name of the column in the updated GeoDataFrame containing the unique cell indices.

    Returns
    -------
    GeoDataFrame
        Updated GeoDataFrame with the `cell_column` column populated.
    """
    nb_layers = sorted(gdf["layer"].unique())
    cell_index = 0
    if cell_column in gdf.columns:
        raise ValueError(
            f"Column '{cell_column}' already exists in the GeoDataFrame. "
            "Please choose a different value for `cell_column`."
        )
    gdf[cell_column] = None

    # Process each pair of adjacent layers
    for a, b in zip(nb_layers[:-1], nb_layers[1:]):
        layer_a = gdf[gdf["layer"] == a]
        layer_b = gdf[gdf["layer"] == b]

        # Match polygons between the layers
        matched_indices = match_cells_iomin(layer_a, layer_b, threshold)

        # Update cell indices based on matches
        for idx1, idx2_list in matched_indices.items():
            for idx2 in idx2_list:
                if gdf.loc[idx1, cell_column] is not None:
                    gdf.loc[idx2, cell_column] = gdf.loc[idx1, cell_column]
                else:
                    gdf.loc[idx1, cell_column] = cell_index
                    gdf.loc[idx2, cell_column] = cell_index
                    cell_index += 1

    unmatched = gdf[cell_column].isna()
    gdf.loc[unmatched, cell_column] = range(cell_index, cell_index + unmatched.sum())

    return gdf


##
params, _ = curve_fit(affine, points.z_raw.compute(), points.z.compute())
a, b = params

gdf_all_layers = gdf_all_layers.assign(
    z_raw=lambda df: df["layer"].apply(layer_index_to_z_raw)
).assign(z=lambda df: affine(x=df["z_raw"], a=a, b=b))


gdf_all_layers = sd.models.ShapesModel.parse(gdf_all_layers)

processed = process_pseudo3D_shapes(
    gdf_all_layers, threshold=0.3, cell_column="cell_id"
)

# sdata["cells_baysor"] = gdf_all_layers
sdata["cells_baysor"] = processed

sdata.write(out_path / "merfish_mouse_ileum.sdata.zarr", overwrite=True)


# ##
# Interactive(sdata)

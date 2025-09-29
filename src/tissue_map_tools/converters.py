import dask.array
import numpy as np
from pathlib import Path

from typing import Any
from cloudvolume.dask import to_cloudvolume
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from xarray import DataArray, DataTree
from dask.dataframe import DataFrame as DaskDataFrame
from numpy.random import default_rng
import pandas as pd
import json


from tissue_map_tools.data_model.annotations import (
    AnnotationInfo,
    AnnotationProperty,
    AnnotationRelationship,
    compute_spatial_index,
    write_annotation_id_index,
    write_related_object_id_index,
    write_spatial_index,
    get_coordinates_and_kd_tree,
)
from tissue_map_tools.data_model.annotations_utils import (
    from_pandas_column_to_annotation_property,
)

RNG = default_rng(42)

# behavior around this should be improved and made consistent across all the functions
# that convert to precomputed format
FACTOR = 1000


def from_ome_zarr_04_raster_to_precomputed_raster(
    ome_zarr_path: str | Path,
    precomputed_path: str | Path,
    is_labels: bool | None = None,
) -> None:
    """
    Convert OME-Zarr v0.4 to Precomputed format.

    Parameters
    ----------
    ome_zarr_path
        Path to the OME-Zarr directory.
    precomputed_path
        Path to save the Precomputed data.
    is_labels
        If True, the data is treated as labels (i.e., the precomputed format will not have
        the "c" axis, which is used for channels). If False, the data is treated as an image.
        If None, the function will try to infer it from the data.
    """
    # read raster data
    ome_zarr_path = Path(ome_zarr_path)
    if not ome_zarr_path.exists():
        raise FileNotFoundError(f"OME-Zarr path {ome_zarr_path} does not exist.")

    reader = Reader(parse_url(ome_zarr_path))
    nodes = list(reader())
    if not nodes:
        raise ValueError(
            "No nodes found in the OME-Zarr directory. If you used "
            "bioformats2raw you may need to pass the path to the "
            "subdirectory '0'"
        )
    found = -1
    for i, node in enumerate(nodes):
        node_path = Path(node.zarr.path).resolve()
        if node_path == ome_zarr_path.resolve():
            found = i
            break
    if found == -1:
        raise ValueError(
            f"OME-Zarr path {ome_zarr_path} does not match any node in the OME-Zarr "
            f"reader. Available nodes: {[str(node.zarr.path) for node in nodes]}"
        )
    dask_data_multiscale = node.data
    dask_data_scale0 = dask_data_multiscale[0]
    print(
        f"Found OME-Zarr node: {node.zarr.path}, with {len(dask_data_multiscale)} "
        f"scales."
        f" Scale 0 has shape {dask_data_scale0.shape} and dtype {dask_data_scale0.dtype}"
    )

    axes = [ax["name"] for ax in node.metadata["axes"]]
    axes_index = {ax: i for i, ax in enumerate(axes)}

    # doesn't validate; bug in bioformats2raw, ome-zarr-models-py, or both
    # import ome_zarr_models
    # import json
    # with open(Path(node.zarr.path) / '.zattrs') as f:
    #     zattrs = f.read()
    #     zattrs = json.loads(zattrs)
    #     del zattrs['multiscales'][0]['metadata']
    #     del zattrs['omero']
    #     ome_zarr_models.v04.Image.model_validate_json(json.dumps(zattrs))

    # TODO: if the validation with ome-zarr-models-py worked, the code below would
    #  not be needed (and we could deal with the general case)
    transformations = node.metadata["coordinateTransformations"]
    scale0 = transformations[0]
    if len(scale0) > 1 or "scale" not in scale0[0]:
        raise ValueError(
            "Only scale transformations are currently supported, not scale + translation."
        )
    scale_factors = dict(zip(axes, scale0[0]["scale"]))

    if "t" in axes:
        if dask_data_scale0.shape[axes_index["t"]] == 1:
            dask_data_scale0 = dask_data_scale0.squeeze(axis=axes_index["t"])
            axes.remove("t")
            axes_index = {ax: i for i, ax in enumerate(axes)}
        else:
            raise ValueError(
                "The OME-Zarr data contains a time dimension, which is not supported by "
                "the Precomputed format. Please convert the data to a single time point."
            )

    # remove c
    if is_labels is not None:
        remove_c = is_labels
        if remove_c and "c" not in axes:
            # nothing to do
            remove_c = False
        if remove_c and dask_data_scale0.shape[axes_index["c"]] > 1:
            raise ValueError(
                "The OME-Zarr data contains multiple channels, but is_labels is True."
            )
        if remove_c and not np.isdtype(dask_data_scale0.dtype, "integral"):
            raise ValueError(
                "The OME-Zarr data is not of integral type, but is_labels is True."
            )
    else:
        is_labels = (
            "c" in axes
            and dask_data_scale0.shape[axes_index["c"]] == 1
            or "c" not in axes
        ) and np.isdtype(dask_data_scale0.dtype, "integral")
        remove_c = is_labels and "c" in axes

    if remove_c:
        dask_data_scale0 = dask_data_scale0.squeeze(axis=axes_index["c"])
        axes.remove("c")
        axes_index = {ax: i for i, ax in enumerate(axes)}

    # axes_cloudvolume = [ax for ax in ["x", "y", "z", "c"] if ax in axes]
    # # cloud volume wants x, y, z(, c) axes order
    # transposed = dask_data_scale0.transpose(
    #     *[axes_index[ax] for ax in axes_cloudvolume]
    # )
    transposed = _transpose_dask_data_for_cloudvolume(dask_data_scale0, axes=axes)

    pixel_sizes = {
        axis: round(FACTOR * scale_factors[axis]) for axis in ["x", "y", "z"]
    }

    layer_type = "segmentation" if is_labels else "image"
    to_cloudvolume(
        arr=transposed,
        layer_type=layer_type,
        cloudpath=precomputed_path,
        resolution=[pixel_sizes["x"], pixel_sizes["y"], pixel_sizes["z"]],
    )
    print(
        f"Converted OME-Zarr data from {ome_zarr_path} to the Precomputed format ("
        f"{layer_type}) at {precomputed_path} with pixel sizes {pixel_sizes} and axes {_get_axes_cloudvolume(axes)}."
    )


def _get_axes_cloudvolume(
    axes: list[str],
) -> list[str]:
    return [ax for ax in ["x", "y", "z", "c"] if ax in axes]


def _transpose_dask_data_for_cloudvolume(
    dask_data: dask.array.Array, axes: list[str]
) -> dask.array.Array:
    # cloud volume wants x, y, z(, c) axes order
    axes_index = {ax: i for i, ax in enumerate(axes)}
    axes_cloudvolume = _get_axes_cloudvolume(axes)
    return dask_data.transpose(*[axes_index[ax] for ax in axes_cloudvolume])


def from_spatialdata_raster_to_precomputed_raster(
    raster: DataArray | DataTree,
    precomputed_path: str | Path,
) -> None:
    import spatialdata as sd

    model = sd.models.get_model(raster)
    if model not in (sd.models.Labels3DModel, sd.models.Image3DModel):
        raise ValueError(
            f"Unsupported model {model}. Only Labels3DModel and Image3DModel are supported."
        )
    transformation = sd.transformations.get_transformation(raster)
    axes = sd.models.get_axes_names(raster)

    transposed = _transpose_dask_data_for_cloudvolume(
        dask_data=raster.data,
        axes=axes,
    )

    layer_type = "segmentation" if model == sd.models.Labels3DModel else "image"

    affine = transformation.to_affine_matrix(
        input_axes=("x", "y", "z"), output_axes=("x", "y", "z")
    )
    if not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3]))):
        raise ValueError(
            "The transformation is not diagonal. Only diagonal transformations are "
            "currently supported."
        )
    pixel_sizes = dict(zip(["x", "y", "z"], np.diag(affine[:3, :3])))
    # the pixel sizes should be in nanometers. This code works for microns and will need
    # to be adapted for general units
    pixel_sizes = {k: round(FACTOR * v) for k, v in pixel_sizes.items()}
    # voxel offset doesn't seem to work. We need to discuss this in a bug to cloudvolume
    # we want to be able to translate the volume (after a cropping in SpatialData), using
    # the voxel_offset parameter
    # voxel_offset = tuple([round(t) for t in affine[:3, 3]])

    ##
    to_cloudvolume(
        arr=transposed,
        layer_type=layer_type,
        cloudpath=precomputed_path,
        resolution=(pixel_sizes["x"], pixel_sizes["y"], pixel_sizes["z"]),
        # voxel_offset=voxel_offset,
    )
    ##
    print(
        f"Converted OME-Zarr data to the Precomputed format ("
        f"{layer_type}) at {precomputed_path} with pixel sizes {pixel_sizes} and axes {_get_axes_cloudvolume(axes)}."
    )


# specs: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
def from_spatialdata_points_to_precomputed_points(
    points: DaskDataFrame | pd.DataFrame,
    precomputed_path: str | Path,
    points_name: str | None = None,
    limit: int = 1000,
    starting_grid_shape: tuple[int, ...] | None = None,
) -> None:
    """

    Parameters
    ----------
    points
    precomputed_path
    points_name
    limit
    starting_grid_shape

    Returns
    -------

    Notes
    -----
    Only unshared points supported at the moment.
    """

    if isinstance(points, DaskDataFrame):
        points = points.compute()

    # # this is a semi-hardcoded example of a relationship; we could generalize
    # TODO: delete this code since AFAIU relationships are better suited for graphs/neighbors
    #  and storing a relationship between categorical values would require to store all the
    #  points that have a certain categorical value, for each point! -> N^2 storage
    # def get_relationships_by_categorical_values(
    #     df: pd.DataFrame, column: str
    # ) -> list[AnnotationRelationship]:
    #     if df[column].dtype != "category":
    #         raise ValueError(
    #             f"Column {column} is not of type 'category'. "
    #             "Relationships can only be created from categorical columns."
    #         )
    #     col = df[column]
    #     relationships = []
    #     for value in col.cat.categories:
    #         relationship = AnnotationRelationship(
    #             id=f"{column}_{value}",
    #             key=f"{column}_{value}",
    #         )
    #         relationships.append(relationship)
    #     return relationships

    xyz, kd_tree = get_coordinates_and_kd_tree(points)
    # this just a hardcoded example of relationship: we hardcode 3 random points and
    # we relate all the objects that are within a certain distance to those points
    ##
    clusters = ["a", "b", "c"]
    cluster_centers = {
        "a": [0.0, 0.0, 0.0],
        "b": [1.0, 1.0, 1.0],
        "c": [0.3, 0.3, 0.3],
    }
    cluster_radii = {
        "a": 0.1,
        "b": 0.2,
        "c": 0.3,
    }
    MAX_NEIGHBORS = 1000
    cluster_neighbors: dict[str, list[int]] = {}
    for cluster_id in clusters:
        cluster_center = cluster_centers[cluster_id]
        cluster_radius = cluster_radii[cluster_id]
        neighbors = kd_tree.query_ball_point(
            cluster_center,
            r=cluster_radius,
        )
        cluster_neighbors[cluster_id] = neighbors[:MAX_NEIGHBORS]
    id_to_cluster: dict[int, str] = {}
    for cluster_id, neighbors in cluster_neighbors.items():
        for neighbor in neighbors:
            id_to_cluster[neighbor] = cluster_id

    ##
    spatial_columns = ["x", "y", "z"]
    properties: list[AnnotationProperty] = [
        from_pandas_column_to_annotation_property(df=points, column=col)
        for col in points.columns
        if col not in spatial_columns
    ]
    # hardcoded example
    relationships = [
        AnnotationRelationship(id=f"neighbors_{cluster}", key=f"neighbors_{cluster}")
        for cluster in clusters
    ]

    ##
    # compute annotations_by_index_id, used in write_annotation_id_index()
    annotations_by_index_id: dict[
        int, tuple[list[float], dict[str, Any], dict[str, list[int]]]
    ] = {}
    # convert all categorical columns to codes and store them in a separate dataframe
    points_categorical = points.select_dtypes(include=["category"])
    for col in points_categorical.columns:
        points_categorical[col] = points[col].cat.codes

    # important: neuroglancer doesn't know about the df.index, it just knows about
    # the "iloc". Also, in iterrows() we do not consider the index, using an enumerate
    # instead
    for i, (_, row) in enumerate(points.iterrows()):
        coords = row[spatial_columns].values.tolist()
        properties_values = {}
        for k, v in row.items():
            if k in spatial_columns:
                continue
            if points[k].dtype == "category":
                k_index = points_categorical.columns.get_loc(k)
                v = points_categorical.iloc[i, k_index]
            properties_values[k] = v
        if i not in id_to_cluster:
            relationships_values = {}
        else:
            cluster = id_to_cluster[i]
            relationships_values = {f"neighbors_{cluster}": cluster_neighbors[cluster]}
        annotations_by_index_id[i] = (coords, properties_values, relationships_values)

    ##
    # compute annotations_by_relationship_id, used in write_related_object_id_index()
    annotations_by_object_id: dict[
        str,
        dict[int, list[tuple[int, list[float], dict[str, Any]]]],
    ] = {}
    for cluster in clusters:
        relationship_name = f"neighbors_{cluster}"
        annotations_by_object_id[relationship_name] = {}
        for neighbor_i in cluster_neighbors[cluster]:
            annotations_by_object_id[relationship_name][neighbor_i] = []
            for neighbor_j in neighbors:
                if neighbor_i == neighbor_j:
                    continue
                coords, properties_values, _ = annotations_by_index_id[neighbor_j]
                annotations_by_object_id[relationship_name][neighbor_i].append(
                    (neighbor_j, coords, properties_values)
                )
    ##
    grid = compute_spatial_index(
        xyz=xyz,
        kd_tree=kd_tree,
        limit=limit,
        starting_grid_shape=starting_grid_shape,
    )
    ##
    # TODO: ensure that the data is anisotropic by looking at the coordinate
    #  transformation
    # TODO: the dimensions belows need to be adjusted, here we are assuming the units
    #  to be meters
    spatial: list[dict[str, Any]] = []
    kw = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"x": [1.0, "um"], "y": [1.0, "um"], "z": [1.0, "um"]},
        "lower_bound": grid[0].mins,
        "upper_bound": grid[0].maxs,
        "annotation_type": "POINT",
        "properties": properties,
        "relationships": relationships,
        "by_id": {"key": "by_id", "sharding": None},
        "spatial": spatial,
    }
    for grid_level in grid.values():
        spatial_item = {
            "key": f"spatial{grid_level.level}",
            "sharding": None,
            "grid_shape": grid_level.grid_shape,
            "chunk_size": grid_level.chunk_size.tolist(),
            "limit": grid_level.limit,
        }
        spatial.append(spatial_item)
    annotation_info = AnnotationInfo(**kw)
    print(annotation_info.model_dump_json(indent=4))

    precomputed_path = Path(precomputed_path)
    if points_name is None:
        points_name = f"points_{limit}"

    try:
        with open(precomputed_path / "info") as infile:
            info = json.loads(infile.read())
            info["annotations"] = points_name
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Precomputed path {precomputed_path} does not exist or does not contain "
            "an 'info' file. Please create a precomputed volume first, e.g. by using "
            "from_ome_zarr_04_raster_to_precomputed_raster()"
        ) from e
    with open(precomputed_path / "info", "w") as outfile:
        outfile.write(json.dumps(info, indent=4))

    points_path = precomputed_path / points_name
    if points_path.exists():
        raise FileExistsError(
            f"Precomputed path {points_path} already exists. "
            "Please remove it or choose a different path."
        )
    points_path.mkdir(exist_ok=True)
    with open(points_path / "info", "w") as outfile:
        outfile.write(
            annotation_info.model_dump_json(
                indent=4, by_alias=True, exclude_unset=True, exclude_none=True
            )
        )

    write_annotation_id_index(
        info=annotation_info,
        root_path=points_path,
        annotations=annotations_by_index_id,
    )
    # from tissue_map_tools.data_model.annotations import read_annotation_id_index
    # debug = read_annotation_id_index(info=annotation_info, root_path=points_path)
    #
    # for k, v in debug.items():
    #     assert debug[k][1]['gene'] == annotations_by_index_id[k][1]['gene'].item()

    write_related_object_id_index(
        info=annotation_info,
        root_path=points_path,
        annotations_by_object_id=annotations_by_object_id,
    )

    ##
    # for each spatial index, compute annotations_by_spatial_chunk, used in
    # write_spatial_index()
    # TODO: we should not use enumerate here because the i is already present in the
    #  key of "AnnotationSpatialLevel"
    for i, annotation_spatial_level in enumerate(annotation_info.spatial):
        annotations_by_spatial_chunk: dict[
            str, list[tuple[int, list[float], dict[str, Any]]]
        ] = {}
        spatial_key = annotation_spatial_level.key
        grid_level = grid[i]
        for cell, indices in grid_level.populated_cells.items():
            cell_name = "_".join(map(str, cell))
            annotations_by_spatial_chunk[cell_name] = []

            for index in indices:
                coords_index, properties_values, _ = annotations_by_index_id[index]
                annotations_by_spatial_chunk[cell_name].append(
                    (index, coords_index, properties_values)
                )

        write_spatial_index(
            info=annotation_info,
            root_path=points_path,
            spatial_key=spatial_key,
            annotations_by_spatial_chunk=annotations_by_spatial_chunk,
        )
    from tissue_map_tools.data_model.annotations import read_spatial_index

    read_spatial_index(
        info=annotation_info, root_path=points_path, spatial_key="spatial2"
    )
    ##


if __name__ == "__main__":
    import shutil

    # CREATE = True
    CREATE = False
    # SHOW = True
    SHOW = True

    precomputed_path = Path("/Users/macbook/Desktop/test_precomputed_points")
    if CREATE:
        # from_ome_zarr_04_raster_to_precomputed_raster(
        #     ome_zarr_path="../../out/20_1_gloms/0",
        #     precomputed_path="../../out/20_1_gloms_precomputed",
        #     # is_labels=False,
        # )
        #
        # from_ome_zarr_04_raster_to_precomputed_raster(
        #     ome_zarr_path="../../out/20_1_gloms/0",
        #     precomputed_path="../../out/20_1_gloms_precomputed",
        #     # is_labels=False,
        # )
        #
        # N_POINTS = 100_000_000
        N_POINTS = 10_000
        # N_POINTS = 101
        df = pd.DataFrame(
            {
                "x": RNG.random(N_POINTS, dtype=np.float32),
                "y": RNG.random(N_POINTS, dtype=np.float32) * 1.5,
                "z": RNG.random(N_POINTS, dtype=np.float32),
                "intensity": RNG.integers(0, 255, N_POINTS, dtype=np.uint32),
                "categorical": RNG.choice(["a", "b", "c"], N_POINTS),
            }
        )
        df["categorical"] = df["categorical"].astype("category")

        if precomputed_path.exists():
            shutil.rmtree(precomputed_path)
        from_spatialdata_points_to_precomputed_points(
            points=df,
            precomputed_path=precomputed_path,
            limit=1000,
        )
    if SHOW:
        import neuroglancer
        import webbrowser

        viewer = neuroglancer.Viewer()

        url = "precomputed://http://localhost:8912"
        with viewer.txn() as s:
            s.layers["points"] = neuroglancer.AnnotationLayer(source=url)

        webbrowser.open(url=viewer.get_viewer_url(), new=2)
        pass

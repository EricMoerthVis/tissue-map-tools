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


from tissue_map_tools.data_model.annotations import (
    AnnotationInfo,
    compute_spatial_index,
)

RNG = default_rng(42)

# behavior around this should be improved and made consistent across all the functions
# that convert to precomputed format
ROUNDING_FACTOR = 1000


def from_ome_zarr_04_raster_to_precomputed_raster(
    ome_zarr_path: str | Path,
    precomputed_path: str | Path,
    is_labels: bool | None = None,
) -> None:
    """
    Convert OME-Zarr v0.4 to Precomputed format.

    Args:
        ome_zarr_path (str | Path): Path to the OME-Zarr directory.
        precomputed_path (str | Path): Path to save the Precomputed data.
        is_labels (bool, optional): If True, the data is treated as labels (i.e., the
            precomputed format will not have the "c" axis, which is used for channels).
            If False, the data is treated as an image. If None, the function will try to
            infer it from the data.
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
        axis: round(ROUNDING_FACTOR * scale_factors[axis]) for axis in ["x", "y", "z"]
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
    pixel_sizes = {k: round(ROUNDING_FACTOR * v) for k, v in pixel_sizes.items()}

    to_cloudvolume(
        arr=transposed,
        layer_type=layer_type,
        cloudpath=precomputed_path,
        resolution=[pixel_sizes["x"], pixel_sizes["y"], pixel_sizes["z"]],
    )
    print(
        f"Converted OME-Zarr data to the Precomputed format ("
        f"{layer_type}) at {precomputed_path} with pixel sizes {pixel_sizes} and axes {_get_axes_cloudvolume(axes)}."
    )


def from_spatialdata_points_to_precomputed_points(
    points: DaskDataFrame | pd.DataFrame,
    precomputed_path: str | Path,
    limit: int = 1000,
    starting_grid_shape: tuple[int, ...] | None = None,
) -> None:
    """

    Parameters
    ----------
    points
    precomputed_path

    Returns
    -------

    Notes
    -----
    Only unshared points supported at the moment.
    """

    if isinstance(points, DaskDataFrame):
        points = points.compute()
    # the neuroglancer specs also allow for "rgb" and "rgba", but these are not native
    # Python types
    SUPPORTED_DTYPES = [
        np.uint32,
        np.int32,
        np.float32,
        np.uint16,
        np.int16,
        np.uint8,
        np.int8,
        "category",
    ]
    for column in points.columns:
        dtype = points[column].dtype
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {dtype} for column {column}. "
                f"Supported dtypes are: {SUPPORTED_DTYPES}"
            )

    grid = compute_spatial_index(
        points=points,
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
        "dimensions": {"x": [1.0, "m"], "y": [1.0, "m"], "z": [1.0, "m"]},
        "lower_bound": grid[0].mins,
        "upper_bound": grid[0].maxs,
        "annotation_type": "POINT",
        "properties": [],
        "relationships": [],
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
    if precomputed_path.exists():
        raise FileExistsError(
            f"Precomputed path {precomputed_path} already exists. "
            "Please remove it or choose a different path."
        )
    precomputed_path.mkdir(exist_ok=True)
    with open(precomputed_path / "info", "w") as outfile:
        outfile.write(annotation_info.model_dump_json(indent=4))
    pass
    ##


if __name__ == "__main__":
    import shutil

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

    precomputed_path = Path("/Users/macbook/Desktop/test_precomputed_points")
    if precomputed_path.exists():
        shutil.rmtree(precomputed_path)
    from_spatialdata_points_to_precomputed_points(
        points=df,
        precomputed_path=precomputed_path,
        limit=1000,
    )

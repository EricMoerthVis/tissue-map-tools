import dask.array
import numpy as np
from pathlib import Path

from cloudvolume.dask import to_cloudvolume
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from xarray import DataArray, DataTree
from dask.dataframe import DataFrame as DaskDataFrame
from numpy.random import default_rng
import pandas as pd
import itertools
from scipy.spatial import KDTree
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from tissue_map_tools.data_model.annotations import AnnotationInfo

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


class GridLevel:
    level: int
    grid_shape: list[int]
    chunk_size: NDArray[float]
    limit: int

    def __init__(
        self,
        level: int,
        grid_shape: list[int],
        mins: NDArray[float],
        maxs: NDArray[float],
        limit: int,
        parent_cells: list[tuple[int, int, int]],
        parent_grid_shape: list[int],
    ) -> None:
        self.level = level
        self.grid_shape = grid_shape
        self.mins = mins
        self.maxs = maxs
        self.limit = limit

        # derived quantities
        self.sizes = np.array(maxs) - np.array(mins)
        self.chunk_size = self.sizes / np.array(self.grid_shape)
        self.cells: list[tuple[int, int, int]] = []

        # quantities set later
        self.populated_cells: dict[tuple[int, int, int], NDArray[float]] = {}

        for parent_cell in parent_cells:
            new_cells_by_dim: dict[int, list[int]] = {}
            for dim in range(3):
                index = parent_cell[dim]
                factor = grid_shape[dim] // parent_grid_shape[dim]
                if factor == 1:
                    new_cells_by_dim[dim] = [index]
                else:
                    new_cells_by_dim[dim] = [index * factor, index * factor + 1]
            new_cells = list(itertools.product(*new_cells_by_dim.values()))
            self.cells.extend(new_cells)

    def iter_full_grid(self):
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                for k in range(self.grid_shape[2]):
                    yield (i, j, k)

    def iter_cells(self):
        for i, j, k in self.cells:
            yield (i, j, k)

    def centroid(self, index: tuple[int, int, int]) -> NDArray[float]:
        """Calculate the centroid of the grid cell."""
        return np.array(index) * self.chunk_size + self.chunk_size / 2 + self.mins

    def get_next_grid_shape(self) -> list[int]:
        """Get the shape of the next grid level so that we get isotropic chunks.

        Notes
        -----
        The specs say: "each component of chunk_size of each successively level
        should be either equal to, or half of, the corresponding component of the
        prior level chunk_size, whichever results in a more spatially isotropic
        chunk."

        We implement this as follows: if a chunk lenght for a given axis is half
        (or less) than the size of any other chunk, then we leave this axis as is,
        otherwise we divide the chunk size by 2.
        """
        next_grid_shape = self.grid_shape.copy()
        for i in range(3):
            if any(
                [
                    self.chunk_size[i] * 2 <= self.chunk_size[j].item()
                    for j in range(3)
                    if j != i
                ]
            ):
                continue
            else:
                next_grid_shape[i] *= 2
        return next_grid_shape


PRINT_DEBUG = False


def from_spatialdata_points_to_precomputed_points(
    points: DaskDataFrame | pd.DataFrame,
    precomputed_path: str | Path,
    limit: int = 1000,
    starting_grid_shape: list[int] | None = None,
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
    # TODO: only points are supported at the moment, not lines, axis-aligned bounding
    #  boxes and ellipsoids
    if starting_grid_shape is None:
        starting_grid_shape = [1, 1, 1]

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
    # TODO: we can generalize to 2D points. 1D points do not make much sense
    xyz = points[["x", "y", "z"]].values
    tree = KDTree(xyz)

    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)

    remaining_indices = set(range(len(xyz)))

    grid: dict[int, GridLevel] = {}
    grid_level = GridLevel(
        level=0,
        grid_shape=starting_grid_shape,
        mins=mins,
        maxs=maxs,
        limit=limit,
        parent_cells=[(0, 0, 0)],
        parent_grid_shape=starting_grid_shape,
    )
    # to avoid the risk of points in the boundary of the grid not being included
    eps = 1e-6
    previous_remaining_indices = len(remaining_indices)

    while len(remaining_indices) > 0:
        # initialization
        grid[grid_level.level] = grid_level
        if PRINT_DEBUG or True:
            print(
                f"Processing grid level {grid_level.level} with shape {grid_level.grid_shape} "
                f"and chunk size {grid_level.chunk_size}. Remaining points: {len(remaining_indices)}"
            )

        # main logic
        if PRINT_DEBUG:
            print("Active cells: ", grid_level.cells)
        for i, j, k in grid_level.iter_cells():
            # calculate the centroid of the grid cell
            centroid = grid_level.centroid((i, j, k))

            # find points in the grid cell
            # this filter points by a radius r, but we have different values per axis
            indices = tree.query_ball_point(
                centroid, r=max(grid_level.chunk_size) / 2 + eps, p=np.inf
            )
            filtered = xyz[indices]
            mask = (
                (centroid[0] - grid_level.chunk_size[0] - eps <= filtered[:, 0])
                & (filtered[:, 0] <= centroid[0] + grid_level.chunk_size[0] + eps)
                & (centroid[1] - grid_level.chunk_size[1] - eps <= filtered[:, 1])
                & (filtered[:, 1] <= centroid[1] + grid_level.chunk_size[1] + eps)
                & (centroid[2] - grid_level.chunk_size[2] - eps <= filtered[:, 2])
                & (filtered[:, 2] <= centroid[2] + grid_level.chunk_size[2] + eps)
            )
            discarded = np.sum(~mask).item()
            if discarded > 0:
                # TODO: possible bug! This message is not printed while I would
                #  expect that the kDTree query would return more points than the mask
                #  would allow (this should happend when chunk_size has different
                #  dimensions
                if PRINT_DEBUG or True:
                    print(
                        f"-----------------> {discarded} points where filtered out of"
                        f" {len(indices)}"
                    )
            indices = np.array(indices)[mask].tolist()

            # filter out points that are not in the grid cell
            indices = [i for i in indices if i in remaining_indices]

            if len(indices) > 0:
                if len(indices) <= limit:
                    emitted = indices
                else:
                    emitted = RNG.choice(indices, size=limit, replace=False)
                if PRINT_DEBUG:
                    print(
                        f"Emitting {len(emitted)} points for grid cell ({i}, {j}, {k})"
                    )
                grid_level.populated_cells[(i, j, k)] = xyz[emitted]
                remaining_indices.difference_update(emitted)

                # create a new layer for this grid cell
                # layer_name = f"level_{grid_level.level}_cell_{i}_{j}_{k}"

                # here we would save the points to the precomputed format
                # e.g., save_points_to_precomputed(points[indices], precomputed_path, layer_name)

        # np.take(xyz, indices, axis=0)
        remaining_xyz = xyz[list(remaining_indices)]

        # visual debug
        VISUAL_DEBUG = False
        if VISUAL_DEBUG:
            plt.figure(figsize=(10, 10))
            chunk_size = grid_level.chunk_size
            lines_x = np.arange(
                grid_level.mins[0],
                grid_level.maxs[0] + chunk_size[0],
                chunk_size[0] + eps,
            )
            lines_y = np.arange(
                grid_level.mins[1],
                grid_level.maxs[1] + chunk_size[1],
                chunk_size[1] + eps,
            )
            for x in lines_x:
                plt.plot(
                    [x, x],
                    [grid_level.mins[1], grid_level.maxs[1]],
                    color="red",
                    linewidth=0.5,
                )
            for y in lines_y:
                plt.plot(
                    [grid_level.mins[0], grid_level.maxs[0]],
                    [y, y],
                    color="red",
                    linewidth=0.5,
                )

            plt.scatter(
                remaining_xyz[:, 0],
                remaining_xyz[:, 1],
                s=100,
                c=remaining_xyz[:, 2],
            )
            if len(remaining_xyz) > 0:
                cbar = plt.colorbar()
                cbar.set_ticks([remaining_xyz[:, 2].min(), remaining_xyz[:, 2].max()])
                cbar.set_ticklabels(
                    [f"{remaining_xyz[:, 2].min()}", f"{remaining_xyz[:, 2].max()}"]
                )
            plt.show()

        # prepare for the next level
        grid_level = GridLevel(
            level=grid_level.level + 1,
            grid_shape=grid_level.get_next_grid_shape(),
            mins=grid_level.mins,
            maxs=grid_level.maxs,
            limit=limit,
            parent_cells=list(grid_level.populated_cells.keys()),
            parent_grid_shape=grid_level.grid_shape,
        )

        # sanity check
        if len(remaining_indices) == previous_remaining_indices:
            raise ValueError(
                "No points were emitted in this grid level. This is likely due to the "
                "grid size being too small."
            )
        previous_remaining_indices = len(remaining_indices)

    print('spatial index computed, now saving to precomputed format')
    pass
    # TODO: ensure that the data is anisotropic, the dimensions belows need to be
    #  adjusted
    ##
    kw = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"x": [1.0, "m"], "y": [1.0, "m"], "z": [1.0, "m"]},
        "lower_bound": grid[0].mins,
        "upper_bound": grid[0].maxs,
        "annotation_type": "POINT",
        "properties": [],
        "relationships": [],
        "by_id": {"key": "by_id", "sharding": None},
        "spatial": [],
    }
    for grid_level in grid.values():
        spatial = {
            "key": f"spatial{grid_level.level}",
            "sharding": None,
            "grid_shape": grid_level.grid_shape,
            "chunk_size": grid_level.chunk_size.tolist(),
            "limit": grid_level.limit,
        }
        kw["spatial"].append(spatial)
    annotation_info = AnnotationInfo(**kw)
    print(annotation_info.model_dump_json(indent=4))
    pass
    ##


if __name__ == "__main__":
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
    from_spatialdata_points_to_precomputed_points(
        points=df,
        precomputed_path="/Users/macbook/Desktop/test_precomputed_points",
        limit=1000,
    )

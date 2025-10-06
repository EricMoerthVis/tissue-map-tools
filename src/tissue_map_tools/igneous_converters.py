"""
This module is optional and contains convenience wrappers around igneous functions.
Igneous parameters are mostly not exposed and are chosen with the aim of producing
sharded, multi-level-of-detail meshes from the largest raster scale and by
parallelizing the computation.
The user is strongly encouraged to use igneous directly when more control on the
meshing operations is needed.

Furthermore, please notice that Igneous is licensed under GPL-3.0; users with licensing
constraints may consider using a separate library for converting the data to the
sharded format and for creating meshes.
"""

from cloudvolume import CloudVolume
from igneous.task_creation.mesh import (
    create_meshing_tasks,
    create_sharded_multires_mesh_tasks,
    create_unsharded_multires_mesh_tasks,
)
from igneous.task_creation.image import (
    create_downsampling_tasks,
    create_image_shard_downsample_tasks,
)
from taskqueue import LocalTaskQueue
from pathlib import Path
from xarray import DataArray, DataTree
from tissue_map_tools.converters import (
    from_ome_zarr_04_raster_to_precomputed_raster,
    from_spatialdata_raster_to_precomputed_raster,
    DEFAULT_UNITS_FACTOR,
)

# same as igneous defaults
DEFAULT_NLOD = 0
DEFAULT_SHAPE = (448, 448, 448)
DEFAULT_MIN_CHUNK_SIZE = (256, 256, 256)


def from_precomputed_raster_modify_scales_and_sharding(
    data_path: str,
    multiscale: bool,
    sharded: bool,
    num_mips: int = 4,
    parallel: int | bool = True,
):
    task_queue = LocalTaskQueue(parallel=parallel)

    # TODO: sharded multiscale is affected by this bug:
    #
    # if sharded and multiscale:
    #     sharded = False
    #     warnings.warn(
    #         "sharded multiscale meshes are currently not supported due to "
    #         "https://github.com/seung-lab/igneous/issues/211\nsetting sharded=False.",
    #         UserWarning,
    #         stacklevel=2,
    #     )

    if multiscale:
        for i in range(num_mips):
            cv = CloudVolume(data_path, mip=i)
            factor = get_downsampling_factor(cv.shape)
            if sharded:
                task = create_image_shard_downsample_tasks(
                    cloudpath=data_path,
                    factor=factor,
                    mip=i,
                )
            else:
                task = create_downsampling_tasks(
                    layer_path=data_path,
                    mip=i,
                    num_mips=1,
                    factor=factor,
                )
            task_queue.insert(task)
            task_queue.execute()
        return
    else:
        cv = CloudVolume(data_path)
        info = cv.meta.info
        is_single_scale = len(info["scales"]) == 1
        if is_single_scale:
            is_sharded = (
                "sharding" in info["scales"][0]
                and info["scales"][0]["sharding"] is not None
            )
            if is_sharded == sharded:
                # the data is already in the desired format
                return

        if sharded:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        # ask_queue.insert(task)
        # task_queue.execute()


def get_downsampling_factor(shape: tuple[int, ...]) -> tuple[int, ...]:
    # in the past only (2, 2, 2) and (2, 2, 1) were supported. Check if this changed
    # https://github.com/seung-lab/igneous/issues/70#issuecomment-638927656
    return 2, 2, 1


def from_precomputed_raster_to_precomputed_meshes(
    data_path: str,
    mesh_name: str | None = None,
    object_ids: list[int] | None = None,
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    nlod: int = DEFAULT_NLOD,
    min_chunk_size: tuple[int, int, int] = DEFAULT_MIN_CHUNK_SIZE,
    parallel: int | bool = True,
    sharded: bool = True,
):
    task_queue = LocalTaskQueue(parallel=parallel)
    # the actual computation of the meshes happens in
    # igneous › tasks › mesh › muitires.py › process_mesh()

    forge_task = create_meshing_tasks(
        layer_path=data_path,
        shape=shape,
        mip=0,
        mesh_dir=mesh_name,
        object_ids=object_ids,
        sharded=sharded,
    )
    task_queue.insert(forge_task)
    task_queue.execute()

    if sharded:
        merge_task = create_sharded_multires_mesh_tasks(
            cloudpath=data_path,
            num_lod=nlod,
            min_chunk_size=min_chunk_size,
        )
    else:
        merge_task = create_unsharded_multires_mesh_tasks(
            cloudpath=data_path,
            num_lod=nlod,
            min_chunk_size=min_chunk_size,
        )
    task_queue.insert(merge_task)
    task_queue.execute()


def from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes(
    ome_zarr_path: str | Path,
    precomputed_path: str | Path,
    is_labels: bool | None = None,
    multiscale: bool = True,
    sharded_raster: bool = True,
    mesh_name: str | None = None,
    units_factor: int = DEFAULT_UNITS_FACTOR,
    object_ids: list[int] | None = None,
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    nlod: int = DEFAULT_NLOD,
    min_chunk_size: tuple[int, int, int] = DEFAULT_MIN_CHUNK_SIZE,
    parallel: int | bool = True,
):
    from_ome_zarr_04_raster_to_precomputed_raster(
        ome_zarr_path=ome_zarr_path,
        precomputed_path=precomputed_path,
        is_labels=is_labels,
        units_factor=units_factor,
    )
    from_precomputed_raster_modify_scales_and_sharding(
        data_path=str(precomputed_path),
        parallel=parallel,
        sharded=sharded_raster,
        multiscale=multiscale,
    )
    from_precomputed_raster_to_precomputed_meshes(
        data_path=str(precomputed_path),
        mesh_name=mesh_name,
        object_ids=object_ids,
        shape=shape,
        nlod=nlod,
        min_chunk_size=min_chunk_size,
        parallel=parallel,
    )


def from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
    raster: DataArray | DataTree,
    precomputed_path: str | Path,
    multiscale: bool = True,
    sharded_raster: bool = True,
    mesh_name: str | None = None,
    units_factor: int = DEFAULT_UNITS_FACTOR,
    object_ids: list[int] | None = None,
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    nlod: int = DEFAULT_NLOD,
    min_chunk_size: tuple[int, int, int] = DEFAULT_MIN_CHUNK_SIZE,
    parallel: int | bool = True,
):
    """
    Convert a SpatialData raster element to a sharded precomputed volume with meshes.

    Parameters
    ----------
    raster
        (Single)scale or multiscale Image or Labels element as defined in the
        SpatialData data model.
    precomputed_path
        Path to the output precomputed volume to output.
    multiscale
        Whether to create a multiscale volume.
    sharded_raster
        Whether to convert the raster to a sharded format.
    mesh_name
        Name of the output mesh directory. If None, the mesh will be stored using the
        default name assigned by igneous.
    units_factor
        The conversion factor so that the data units are in nanometers. In practice, a
        large number may lead to an overflow bug reported here
        https://github.com/seung-lab/igneous/issues/209
    object_ids
        The IDs of the meshes to create. If None, meshes will be created for all.
    shape
        The shape of the chunk for the finest meshes level (=best level of details).
        Chunks should not be too small, otherwise the visualization will be less
        performant.
    nlod
        The number of levels of details for the meshes. This number is ignored if the
        `min_chunk_size` is too large compared to the size of individual meshes. See the
        code in igneous › tasks › mesh › muitires.py › process_mesh() for details.
    min_chunk_size
        The minimum chunk size for the coarsest meshes level (=worst level of details).
        This is a tuple of 3 integers. A high value will lead to a smaller actual number
        of levels of detail (see above).
    parallel
        Parallel processing.

    Returns
    -------
    None
        Writes the precomputed volume and the meshes to disk.

    Notes
    -----
    Regarding the volume data that is produced, the recommendation is to use
    `multiscale=True` and `sharded_raster=True` for better performance with volume data.

    Regarding the meshes, the precomputed format supports multi-level of details meshes,
    but the number of meshes remains the same. This means that if you have 1M meshes,
    the performance will improve by using a coarser level of details, but you will still
    have to load 1M meshes. A workaround is to use `object_ids` to create meshes only
    when you need them and to ensure that the level-of-details is used correctly.
    For this you can look at the histogram in the Neuroglancer mesh layer, under:
    "Settings icon > Render > Resolution (mesh)". Please see the explanation above for
    `nlod`, `shape`, and `min_chunk_size` for controlling the level-of-details.
    """
    # TODO: support datatree
    if isinstance(raster, DataTree):
        raster = raster["scale0"]["image"]
    from_spatialdata_raster_to_precomputed_raster(
        raster=raster,
        precomputed_path=precomputed_path,
        units_factor=units_factor,
    )
    from_precomputed_raster_modify_scales_and_sharding(
        data_path=str(precomputed_path),
        parallel=parallel,
        sharded=sharded_raster,
        multiscale=multiscale,
    )
    from_precomputed_raster_to_precomputed_meshes(
        data_path=str(precomputed_path),
        mesh_name=mesh_name,
        object_ids=object_ids,
        shape=shape,
        nlod=nlod,
        min_chunk_size=min_chunk_size,
        parallel=parallel,
    )


if __name__ == "__main__":
    from_precomputed_raster_to_precomputed_meshes(
        data_path="/Users/macbook/Desktop/moffitt_precomputed",
    )

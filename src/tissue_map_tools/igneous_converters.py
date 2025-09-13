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

from igneous.task_creation.mesh import (
    create_meshing_tasks,
    create_sharded_multires_mesh_tasks,
    create_unsharded_multires_mesh_tasks,
)
from taskqueue import LocalTaskQueue
from pathlib import Path
from xarray import DataArray, DataTree
from tissue_map_tools.converters import (
    from_ome_zarr_04_raster_to_precomputed_raster,
    from_spatialdata_raster_to_precomputed_raster,
)

DEFAULT_NLOD = 4


def from_precomputed_raster_to_precomputed_meshes(
    data_path: str,
    mesh_name: str | None = None,
    nlod: int = DEFAULT_NLOD,
    parallel: int | bool = True,
    sharded: bool = True,
):
    task_queue = LocalTaskQueue(parallel=parallel)

    forge_task = create_meshing_tasks(
        layer_path=data_path,
        mip=0,
        mesh_dir=mesh_name,
        sharded=sharded,
    )
    task_queue.insert(forge_task)
    task_queue.execute()

    if sharded:
        merge_task = create_sharded_multires_mesh_tasks(
            cloudpath=data_path,
            num_lod=nlod,
        )
    else:
        merge_task = create_unsharded_multires_mesh_tasks(
            cloudpath=data_path,
            num_lod=nlod,
        )
    task_queue.insert(merge_task)
    task_queue.execute()


def from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes(
    ome_zarr_path: str | Path,
    precomputed_path: str | Path,
    is_labels: bool | None = None,
    mesh_name: str | None = None,
    nlod: int = DEFAULT_NLOD,
    parallel: int | bool = True,
):
    from_ome_zarr_04_raster_to_precomputed_raster(
        ome_zarr_path=ome_zarr_path,
        precomputed_path=precomputed_path,
        is_labels=is_labels,
    )
    from_precomputed_raster_to_precomputed_meshes(
        data_path=str(precomputed_path),
        mesh_name=mesh_name,
        nlod=nlod,
        parallel=parallel,
    )


def from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
    raster: DataArray | DataTree,
    precomputed_path: str | Path,
    mesh_name: str | None = None,
    nlod: int = DEFAULT_NLOD,
    parallel: int | bool = True,
):
    from_spatialdata_raster_to_precomputed_raster(
        raster=raster,
        precomputed_path=precomputed_path,
    )
    from_precomputed_raster_to_precomputed_meshes(
        data_path=str(precomputed_path),
        mesh_name=mesh_name,
        nlod=nlod,
        parallel=parallel,
    )


if __name__ == "__main__":
    from_precomputed_raster_to_precomputed_meshes(
        data_path="/Users/macbook/Desktop/moffitt_precomputed",
    )

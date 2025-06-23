"""
This module is optional and contains convenience wrappers around igneous functions.
Igneous parameters are mostly not exposed and are chosen with the aim of producing
sharded, multi-level-of-detail meshes from the largest raster scale and by
parallelizing the computation.
The user is strongly encouraged to use igneous directly if more control is needed.

Furthermore, please notice that Igneous is licensed under GPL-3.0; users with licensing
constraints may consider using a separate library for converting the data to the
sharded format and for creating meshes.
"""

from igneous.task_creation.image import create_image_shard_transfer_tasks
from igneous.task_creation.mesh import (
    create_sharded_multires_mesh_from_unsharded_tasks,
    create_sharded_multires_mesh_tasks,
    create_xfer_meshes_tasks,
)
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue
from pathlib import Path


def from_unshared_precomputed_to_sharded_precomputed(
    data_path: str,
    raster: bool = True,
    meshes: bool = True,
    parallel: int | bool = True,
):
    cv = CloudVolume(cloudpath=data_path)
    raster_name = cv.info["scales"][0]["key"]
    if "mesh" not in cv.meta.info:
        meshes = False

    task_queue = LocalTaskQueue(parallel=parallel)
    is_sharded = cv.image.is_sharded(mip=0)

    if raster and not is_sharded:
        raster_subpath = Path(cv.layerpath) / raster_name
        task = create_image_shard_transfer_tasks(
            src_layer_path=data_path, dst_layer_path=str(raster_subpath), mip=0
        )
        task_queue.insert(task)
        pass
        # igneous.

    if meshes:
        mesh_subpath = cv.meta.info["mesh"]

    task_queue.execute()


def from_precomputed_raster_to_precomputed_meshes(parallel: int | bool = True):
    pass


def from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes(
    parallel: int | bool = True,
):
    pass


def from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
    parallel: int | bool = True,
):
    pass


if __name__ == "__main__":
    from_unshared_precomputed_to_sharded_precomputed(
        data_path="/Users/macbook/Desktop/moffitt_precomputed",
    )

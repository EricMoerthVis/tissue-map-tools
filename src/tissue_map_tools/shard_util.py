from cloudvolume.datasource.precomputed.sharding import (
    ShardReader,
    ShardingSpecification,
)
from cloudvolume import CloudVolume
from pathlib import Path


def get_ids_from_mesh_files(
    data_path: str | Path,
    root_data_path: str | Path,
    shard_filename: str | None = None,
) -> list[int]:
    """
    Get mesh IDs from shard files in a specified directory.

    Parameters
    ----------
    data_path
        path to the folder containing the shard files
    root_data_path
        path to the CloudVolume data, used to access metadata and cache (for
        instance, for meshes this is the parent folder of the mesh directory files).
    shard_filename
        filenames for the shard files (including the .shard extension). If unspecified,
        all shard files in the directory will be considered.
    Returns
    -------
    list[int]
        list of mesh IDs found in the shard files
    """
    root_data_path = Path(root_data_path)
    data_path = Path(data_path)
    cv = CloudVolume(cloudpath=str(root_data_path))
    meta = cv.mesh.meta
    cache = cv.mesh.cache
    if "sharding" in cv.mesh.meta.info:
        data = cv.mesh.meta.info["sharding"]
        data["type"] = data["@type"]
        del data["@type"]
        sharding_specification = ShardingSpecification(**data)
        shard_reader = ShardReader(meta=meta, cache=cache, spec=sharding_specification)
        # list all shard files
        if shard_filename is None:
            shard_files = [str(f) for f in data_path.glob("*.shard")]
        else:
            shard_files = [str(Path(data_path) / shard_filename)]
        mesh_ids = []
        for shard_file in shard_files:
            ids = shard_reader.list_labels(shard_file)
            ids = [int(id) for id in ids]
            mesh_ids.extend(ids)
    else:
        index_files = [f.stem for f in data_path.glob("*.index")]
        mesh_ids = [int(f) for f in index_files]
    return mesh_ids

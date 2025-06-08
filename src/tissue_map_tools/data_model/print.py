from cloudvolume import CloudVolume
import xarray as xr


def print_cloudvolume_volume(data: CloudVolume):
    # Assume `data` is your CloudVolumePrecomputed object
    # data.info is your metadata dict
    # data.to_dask() returns your Dask array

    # Extract metadata
    info = data.info
    scale = info["scales"][0]

    # Create the xarray DataArray using Dask array and metadata
    da = xr.DataArray(
        data.to_dask(),  # Dask array
        dims=("x", "y", "z", "channel"),
        attrs={
            "data_type": info["data_type"],
            "encoding": scale["encoding"],
            "key": scale["key"],
            "resolution": scale["resolution"],
            "voxel_offset": scale["voxel_offset"],
            "chunk_sizes": scale["chunk_sizes"],
            "size": scale["size"],
            "num_channels": info["num_channels"],
            "type": info["type"],
            "mesh": info["mesh"],
        },
    )

    # Wrap in a DataTree
    tree = xr.DataTree.from_dict({"root": {"segmentation": da}})

    # Optionally, attach global metadata to the root node
    tree.attrs = info

    # Now `tree` is your DataTree containing both the metadata and Dask data

    pass


def print_cloudvolume_mesh(data: CloudVolume):
    pass


if __name__ == "__main__":
    data_path = "../../../out/20_1_gloms_precomputed"
    cv = CloudVolume(data_path)
    print_cloudvolume_volume(cv)
    print_cloudvolume_mesh(cv)

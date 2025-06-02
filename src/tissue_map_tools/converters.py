from pathlib import Path

from cloudvolume.dask import to_cloudvolume
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def from_ome_zarr_04_labels_to_precomputed(
    ome_zarr_path: str | Path,
    precomputed_path: str | Path,
):
    """
    Convert OME-Zarr v0.4 to Precomputed format.

    Args:
        ome_zarr_path (str): Path to the OME-Zarr directory.
        precomputed_path (str): Path to save the Precomputed data.
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

    # cloud volume wants x, y, z(, c) axes order
    transposed = dask_data_scale0.transpose(
        axes_index["x"], axes_index["y"], axes_index["z"], axes_index["c"]
    )

    # remove c (this should be optional depending on an argument)
    dask_data_scale0 = dask_data_scale0.squeeze(axis=axes_index["c"])
    axes.remove("c")
    axes_index = {ax: i for i, ax in enumerate(axes)}

    ROUNDING_FACTOR = 1000
    pixel_sizes = {
        axis: round(ROUNDING_FACTOR * scale_factors[axis]) for axis in ["x", "y", "z"]
    }

    to_cloudvolume(
        arr=transposed,
        cloudpath=precomputed_path,
        resolution=[pixel_sizes["x"], pixel_sizes["y"], pixel_sizes["z"]],
    )


if __name__ == "__main__":
    from_ome_zarr_04_labels_to_precomputed(
        ome_zarr_path="../../out/20_1_gloms/0",
        precomputed_path="../../out/20_1_gloms_precomputed",
    )

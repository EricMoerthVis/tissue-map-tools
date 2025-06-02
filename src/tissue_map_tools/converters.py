import numpy as np
from pathlib import Path

from cloudvolume.dask import to_cloudvolume
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def from_ome_zarr_04_raster_to_precomputed(
    ome_zarr_path: str | Path,
    precomputed_path: str | Path,
    is_labels: bool | None = None,
):
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
            raise ValueError(
                "The OME-Zarr data does not contain a 'c' axis, but is_labels is True."
            )
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

    axes_cloudvolume = [ax for ax in ["x", "y", "z", "c"] if ax in axes]
    # cloud volume wants x, y, z(, c) axes order
    transposed = dask_data_scale0.transpose(
        *[axes_index[ax] for ax in axes_cloudvolume]
    )

    ROUNDING_FACTOR = 1000
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
        f"{layer_type}) at {precomputed_path} with pixel sizes {pixel_sizes} and axes {axes_cloudvolume}."
    )


if __name__ == "__main__":
    from_ome_zarr_04_raster_to_precomputed(
        ome_zarr_path="../../out/20_1_gloms/0",
        precomputed_path="../../out/20_1_gloms_precomputed",
        # is_labels=False,
    )

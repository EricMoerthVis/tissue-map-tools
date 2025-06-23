import webbrowser

import warnings
import napari
import neuroglancer
from cloudvolume import CloudVolume
from pathlib import Path
from numpy.random import default_rng
import numpy as np

RNG = default_rng(42)


def view_precomputed_in_neuroglancer(
    data_path: str,
    layer_name: str | None = None,
    mesh_layer_name: str | None = None,
    mesh_ids: list[int] | None = None,
    show_meshes: bool = True,
    port: int = 10001,
    viewer: neuroglancer.Viewer | None = None,
    open_browser: bool = True,
    host_local_data: bool = True,
) -> neuroglancer.Viewer:
    if viewer is None:
        viewer = neuroglancer.Viewer()

    cv = CloudVolume(cloudpath=data_path)
    data_type = cv.info["type"]
    # layer_name = layer_name if layer_name is not None else Path(cv.layerpath).name
    layer_name = layer_name if layer_name is not None else cv.info['scales'][0]['key']

    if viewer is None:
        viewer = neuroglancer.Viewer()
    url = f"precomputed://http://localhost:{port}"
    cv
    with viewer.txn() as s:
        if data_type == "image":
            s.layers[layer_name] = neuroglancer.ImageLayer(
                source=url,
            )
        elif data_type == "segmentation":
            s.layers[layer_name] = neuroglancer.SegmentationLayer(
                source=url,
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        if show_meshes:
            if "mesh" in cv.meta.info:
                mesh_subpath = cv.meta.info["mesh"]
                mesh_layer_name = mesh_layer_name if mesh_layer_name else mesh_subpath
                s.layers[mesh_layer_name] = neuroglancer.SegmentationLayer(
                    source=url + f"/{mesh_subpath}",
                    segments=mesh_ids,
                )

    if open_browser:
        webbrowser.open(url=viewer.get_viewer_url(), new=2)
    if host_local_data:
        cv.viewer(port=port)

    return viewer


def view_precomputed_in_napari(
    data_path: str,
    layer_name: str | None = None,
    mesh_layer_name: str | None = None,
    mesh_ids: list[int] | None = None,
    show_raster: bool = False,
    show_meshes: bool = True,
    show_axes: bool = True,
    viewer: napari.Viewer | None = None,
    open: bool = True,
):
    if viewer is None:
        viewer = napari.Viewer(ndisplay=3)

    cv = CloudVolume(data_path)
    layer_name = layer_name if layer_name is not None else cv.info['scales'][0]['key']

    if show_raster:
        raster = cv.to_dask()
        if raster.ndim != 4:
            raise ValueError(
                f"Expected raster to have 4 dimensions (z, y, x, c), got {raster.ndim}."
            )
        # convert from zyxc to czyx
        raster = raster.transpose(3, 0, 1, 2)
        type = cv.meta.info["type"]
        zyx_scale_factors = cv.meta.info["scales"][0]["resolution"]
        affine = np.diag(zyx_scale_factors + [1])
        if type == "image":
            viewer.add_image(
                raster,
                name=layer_name,
                colormap="gray",
                affine=affine,
            )
        elif type == "segmentation":
            viewer.add_labels(
                raster,
                name=layer_name,
                affine=affine,
            )
        else:
            raise ValueError(f"Unsupported data type: {type}")

    if show_meshes:
        if mesh_ids is None:
            raise NotImplementedError("mesh_ids must be provided for mesh layers.")

        mesh_layer_name = mesh_layer_name if mesh_layer_name else cv.info['mesh']

        meshes = cv.mesh.get(segids=mesh_ids[1:])

        random_colors = RNG.random((len(unique_labels), 3))

        data_mins_xyz: list[float] = []
        data_maxs_xyz: list[float] = []
        for mesh_id in mesh_ids:
            if mesh_id == 0:
                continue
            mesh = meshes[mesh_id]
            vertices = mesh.vertices
            faces = mesh.faces
            vertex_colors = np.full((len(vertices), 3), random_colors[mesh_id])
            values = np.full(len(vertices), mesh_id)
            surface = (vertices, faces, values)
            viewer.add_surface(
                surface,
                vertex_colors=vertex_colors,
                name=f"{mesh_layer_name}_{mesh_id}",
            )

            if show_axes:
                mins, maxs = np.min(vertices, axis=0), np.max(vertices, axis=0)
                if not data_mins_xyz:
                    data_mins_xyz = mins.tolist()
                    data_maxs_xyz = maxs.tolist()
                else:
                    data_mins_xyz = np.minimum(data_mins_xyz, mins).tolist()
                    data_maxs_xyz = np.maximum(data_maxs_xyz, maxs).tolist()

    if show_axes:
        if not show_meshes:
            warnings.warn(
                "Currently show_axes is only supported when show_meshes is True."
            )
        viewer.add_vectors(
            [
                # z axis
                [
                    [0, 0, 0],
                    [0, 0, data_maxs_xyz[2] - data_mins_xyz[2]],
                ],
                # y axis
                [
                    [0, 0, 0],
                    [0, data_maxs_xyz[1] - data_mins_xyz[1], 0],
                ],
                # x axis
                [
                    [0, 0, 0],
                    [data_maxs_xyz[0] - data_mins_xyz[0], 0, 0],
                ],
            ],
            # vectors,
            edge_color=["blue", "green", "red"],
            edge_width=5000,
            name="xyz axes (rgb)",
        )

    if open:
        napari.run()


if __name__ == "__main__":
    # fmt: off
    # unique_labels = [
    #     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    #     22, 23, 24,
    # ]
    # fmt: on
    # viewer = view_precomputed_in_neuroglancer(
    #     data_path="../../out/20_1_gloms_precomputed",
    #     mesh_layer_name="mesh_mip_0_err_40",
    #     mesh_ids=unique_labels,
    # )
    # viewer = view_precomputed_in_napari(
    #     data_path="../../out/20_1_gloms_precomputed",
    #     mesh_layer_name="glom",
    #     mesh_ids=unique_labels,
    # )
    unique_labels = np.arange(5929).astype(int).tolist()
    viewer = view_precomputed_in_neuroglancer(
        data_path="/Users/macbook/Desktop/moffitt_precomputed",
        mesh_ids=unique_labels,
    )
    # unique_labels = np.arange(100).astype(int).tolist()
    # viewer = view_precomputed_in_napari(
    #     data_path="/Users/macbook/Desktop/moffitt_precomputed",
    #     mesh_ids=unique_labels,
    # )
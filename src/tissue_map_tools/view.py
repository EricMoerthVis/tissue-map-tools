import webbrowser

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
    port: int = 10001,
    viewer: neuroglancer.Viewer | None = None,
    open_browser: bool = True,
    host_local_data: bool = True,
) -> neuroglancer.Viewer:
    if mesh_ids is not None and mesh_layer_name is None:
        raise ValueError("mesh_layer_name must be provided if mesh_ids are specified.")

    if viewer is None:
        viewer = neuroglancer.Viewer()

    cv = CloudVolume(cloudpath=data_path)
    data_type = cv.info["type"]
    layer_name = layer_name if layer_name is not None else Path(cv.layerpath).name

    if viewer is None:
        viewer = neuroglancer.Viewer()
    url = f"precomputed://http://localhost:{port}"
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

        if mesh_layer_name is not None:
            s.layers[mesh_layer_name] = neuroglancer.SegmentationLayer(
                source=url + f"/{mesh_layer_name}",
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
    show_axes: bool = True,
    viewer: napari.Viewer | None = None,
    open: bool = True,
):
    if viewer is None:
        viewer = napari.Viewer(ndisplay=3)

    cv = CloudVolume(data_path)

    if mesh_ids is None:
        raise NotImplementedError("mesh_ids must be provided for mesh layers.")

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
        viewer.add_surface(surface, vertex_colors=vertex_colors)

        if show_axes:
            mins, maxs = np.min(vertices, axis=0), np.max(vertices, axis=0)
            if not data_mins_xyz:
                data_mins_xyz = mins.tolist()
                data_maxs_xyz = maxs.tolist()
            else:
                data_mins_xyz = np.minimum(data_mins_xyz, mins).tolist()
                data_maxs_xyz = np.maximum(data_maxs_xyz, maxs).tolist()

    if show_axes:
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
            name="xyz axes: rgb",
        )

    if open:
        napari.run()


if __name__ == "__main__":
    # fmt: off
    unique_labels = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24
    ]
    # fmt: on
    # viewer = view_precomputed_in_neuroglancer(
    #     data_path="../../out/20_1_gloms_precomputed",
    #     mesh_layer_name="mesh_mip_0_err_40",
    #     mesh_ids=unique_labels,
    # )
    viewer = view_precomputed_in_napari(
        data_path="../../out/20_1_gloms_precomputed",
        mesh_layer_name="mesh_mip_0_err_40",
        mesh_ids=unique_labels,
    )

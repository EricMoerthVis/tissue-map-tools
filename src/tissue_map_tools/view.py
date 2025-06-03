import webbrowser
import neuroglancer
from cloudvolume import CloudVolume
from pathlib import Path


def view_precomputed_in_neuroglancer(
    local_data_path: str,
    layer_name: str | None = None,
    mesh_layer_name: str | None = None,
    mesh_labels: list[int] | None = None,
    port: int = 10001,
    viewer: neuroglancer.Viewer | None = None,
    open_browser: bool = True,
    host_local_data: bool = True,
) -> neuroglancer.Viewer:
    if mesh_labels is not None and mesh_layer_name is None:
        raise ValueError(
            "mesh_layer_name must be provided if mesh_labels are specified."
        )

    if viewer is None:
        viewer = neuroglancer.Viewer()

    cv = CloudVolume(cloudpath=local_data_path)
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
                segments=mesh_labels,
            )

    if open_browser:
        webbrowser.open(url=viewer.get_viewer_url(), new=2)
    if host_local_data:
        cv.viewer(port=port)

    return viewer


if __name__ == "__main__":
    # fmt: off
    unique_labels = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24
    ]
    # fmt: on
    viewer = view_precomputed_in_neuroglancer(
        local_data_path="../../out/20_1_gloms_precomputed",
        layer_name=None,
        mesh_layer_name="mesh_mip_0_err_40",
        mesh_labels=unique_labels,
    )

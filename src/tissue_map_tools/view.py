import webbrowser
import neuroglancer
from cloudvolume import CloudVolume
from pathlib import Path


def view_precomputed_in_neuroglancer(
    local_data_path: str,
    port: int = 10001,
    viewer: neuroglancer.Viewer | None = None,
    layer_name: str | None = None,
    open_browser: bool = True,
    host_local_data: bool = True,
    unique_labels: list[int] | None = None,
) -> neuroglancer.Viewer:
    if viewer is None:
        viewer = neuroglancer.Viewer()

    data_type: str | None = None
    try:
        cv = CloudVolume(cloudpath=local_data_path)
    except KeyError as e:
        if str(e) == "'scales'":
            parent = Path(local_data_path).parent
            mesh_dir = Path(local_data_path).name
            cv = CloudVolume(cloudpath=str(parent), mesh_dir=mesh_dir)
            data_type = "mesh"
        else:
            raise e
    if data_type is None:
        data_type = cv.info["type"]
    layer_name = layer_name if layer_name is not None else Path(cv.layerpath).name

    if unique_labels is not None and data_type != "mesh":
        raise ValueError(
            "unique_labels can only be used with mesh data type, but the data type "
            f"is: {data_type}"
        )

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
        elif data_type == "mesh":
            s.layers[layer_name] = neuroglancer.SegmentationLayer(
                # s.layers[layer_name] = neuroglancer.SingleMeshLayer(
                source=url + f"/{Path(local_data_path).name}",
                segments=unique_labels,
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

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
        open_browser=False,
        host_local_data=False,
        # unique_labels=unique_labels,
    )
    view_precomputed_in_neuroglancer(
        viewer=viewer,
        local_data_path="../../out/20_1_gloms_precomputed/mesh_mip_0_err_40",
        layer_name="meshes",
        open_browser=True,
        host_local_data=True,
        unique_labels=unique_labels,
    )

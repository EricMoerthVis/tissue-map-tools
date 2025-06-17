from cloudvolume import CloudVolume
import xarray as xr
from tissue_map_tools.data_model.mesh_info import MultilodDracoInfo
from devtools import pprint
from tqdm import tqdm


def print_cloudvolume_volume(data: CloudVolume):
    info = data.info.copy()
    scales = info["scales"]

    xdatas = []
    for i, scale in enumerate(scales):
        cv = CloudVolume(data.cloudpath, mip=i)
        dask_data = cv.to_dask()
        da = xr.DataArray(
            dask_data,
            dims=("x", "y", "z", "c"),
            attrs=scale.copy(),
        )
        xdatas.append(da)

    datasets = {}
    for i, da in enumerate(xdatas):
        ds = xr.Dataset(
            {"image": da},
        )
        datasets[f"scale{i}"] = ds

    tree = xr.DataTree.from_dict(datasets)
    info.pop("scales")
    tree.attrs = info
    return tree


def print_cloudvolume_mesh(data: CloudVolume, unique_ids: list[int] | None = None):
    if unique_ids is None:
        raise NotImplementedError(
            "Currently, unique_ids cannot be inferred from the mesh."
        )
    info = data.info.copy()
    if "mesh" not in info:
        return
    # name = info["mesh"]
    data.mesh.__dict__
    mesh_info = data.mesh.meta.info
    manifest = data.mesh.get_manifest(segid=1)
    manifest.__dict__
    multilod_draco_info = MultilodDracoInfo.model_validate(mesh_info)
    pprint(multilod_draco_info)

    ##
    # the number of lods needs to be inferred from the mesh info
    for lod in [0, 1, 2, 3]:
        for segid in tqdm(unique_ids, desc="Downloading meshes"):
            if segid == 0:
                continue
            mesh = data.mesh.get(segids=segid, lod=lod)[segid]
            print(mesh)
            mesh.vertices
            mesh.faces
    ##
    pass


if __name__ == "__main__":
    data_path = "../../../out/20_1_gloms_precomputed_multiscale"
    cv = CloudVolume(data_path)
    dt = print_cloudvolume_volume(cv)
    print(dt)
    print(dt["scale0"].image)
    portion = dt["scale0"].image.sel(
        x=slice(50, 100), y=slice(50, 100), z=slice(50, 100)
    )
    portion.data.compute().shape
    cv.__dir__()
    cv.image.__dir__()

    from cloudvolume import Bbox

    bbox = Bbox((50, 50, 50), (100, 100, 100), unit="vx")
    cv.image.download(bbox=bbox, mip=0).shape

    cv_mesh = CloudVolume("../../../out/20_1_gloms_precomputed")
    # fmt: off
    unique_ids = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24]
    # fmt: on
    print_cloudvolume_mesh(cv_mesh, unique_ids=unique_ids)

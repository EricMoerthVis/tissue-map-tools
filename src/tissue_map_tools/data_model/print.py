from cloudvolume import CloudVolume
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from tissue_map_tools.shard_util import get_ids_from_shard_files
from xarray import DataTree


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
    info = data.info.copy()
    if "mesh" not in info:
        return []

    if unique_ids is None:
        path = Path(data.meta.path.basepath) / data.meta.path.layer
        mesh = data.info["mesh"]
        data_path = str(path / mesh)
        unique_ids = get_ids_from_shard_files(root_data_path=path, data_path=data_path)

    datatrees = []
    for segid in tqdm(unique_ids, desc="Downloading meshes"):
        manifest = data.mesh.get_manifest(segid=segid)
        lod_datasets = {}
        for lod in range(manifest.num_lods):
            mesh = data.mesh.get(segids=segid, lod=lod)[segid]
            ds = xr.Dataset(
                {
                    "vertices": (("vertex", "xyz"), mesh.vertices),
                    "faces": (("face", "corners"), mesh.faces),
                }
            )
            # Optionally add normals if present
            if (
                hasattr(mesh, "normals")
                and mesh.normals is not None
                and len(mesh.normals) > 0
            ):
                ds["normals"] = (("vertex", "normal"), mesh.normals)
            ds.attrs = {"segid": segid, "lod": lod}
            lod_datasets[f"lod{lod}"] = ds
        tree = DataTree.from_dict(lod_datasets)
        tree.attrs = {"segid": segid}
        datatrees.append(tree)
    return datatrees


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
    dt = print_cloudvolume_mesh(cv_mesh)
    print(dt)

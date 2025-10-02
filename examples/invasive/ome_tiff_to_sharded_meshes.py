#
# TODO: this code is almost identical to the code for the melanoma example. We should
#  bundle it into APIs and reuse them
from pathlib import Path
from dask_image.imread import imread
import xmltodict
import tifffile
from spatialdata import SpatialData
from spatialdata.models import Labels3DModel
from igneous.task_creation.mesh import (  # noqa: F401
    create_meshing_tasks,
    create_sharded_multires_mesh_tasks,
    create_unsharded_multires_mesh_tasks,
)
from taskqueue import LocalTaskQueue
from tissue_map_tools.igneous_converters import (  # noqa: F401
    from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes,
    from_spatialdata_raster_to_precomputed_raster,
    from_precomputed_raster_to_precomputed_meshes,
)
from tissue_map_tools.view import (  # noqa: F401
    view_precomputed_in_neuroglancer,
)

f = Path(__file__).parent.parent.parent / "data" / "invasive_mask.ome.tiff"

##
if not f.exists():
    raise FileNotFoundError(
        f"File {f} does not exist. Please use symlinks to make the data available."
    )

##
data = imread(f)
data = data.rechunk((256, 512, 512))

xml = tifffile.TiffFile(f).ome_metadata
xml_dict = xmltodict.parse(xml)
sizes = {
    ax: int(xml_dict["OME"]["Image"]["Pixels"][f"@Size{ax.upper()}"]) for ax in "xyz"
}

# quick hack: to find the dimension of the data we use the fact that the sizes are unique
assert len(set(sizes.values())) == 3
dims = []
for size in data.shape:
    for ax, ax_size in sizes.items():
        if size == ax_size:
            dims.append(ax)
            break

labels = Labels3DModel.parse(data, dims=dims)
sdata = SpatialData.init_from_elements({"labels": labels})
sdata.write("/Users/macbook/Desktop/invasive.zarr", overwrite=True)

##
# read again to take advantage of the Zarr chunking
sdata = SpatialData.read("/Users/macbook/Desktop/invasive.zarr")


##

DEFAULT_NLOD = 4
mesh_name: str | None = None
nlod: int = DEFAULT_NLOD
parallel: int | bool = True
# sharded format currently not working: https://github.com/seung-lab/igneous/issues/209
sharded: bool = False
data_path = "/Users/macbook/Desktop/invasive_precomputed"

task_queue = LocalTaskQueue(parallel=parallel)

forge_task = create_meshing_tasks(
    layer_path=data_path,
    mip=0,
    mesh_dir=mesh_name,
    sharded=sharded,
)
task_queue.insert(forge_task)
task_queue.execute()

if sharded:
    merge_task = create_sharded_multires_mesh_tasks(
        cloudpath=data_path,
        num_lod=nlod,
    )
else:
    merge_task = create_unsharded_multires_mesh_tasks(
        cloudpath=data_path,
        num_lod=nlod,
    )
task_queue.insert(merge_task)
task_queue.execute()

##
# from_precomputed_raster_to_precomputed_meshes(
#     data_path="/Users/macbook/Desktop/invasive_precomputed",
# )

##
# from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
#     raster=sdata["labels"],
#     precomputed_path="/Users/macbook/Desktop/invasive_precomputed",
# )

##


viewer = view_precomputed_in_neuroglancer(
    data_path="/Users/macbook/Desktop/invasive_precomputed",
)

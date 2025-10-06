#
# TODO: this code is almost identical to the code for the melanoma example. We should
#  bundle it into APIs and reuse them
from pathlib import Path
from dask_image.imread import imread  # noqa: F401
import xmltodict  # noqa: F401
import tifffile  # noqa: F401
from spatialdata import SpatialData
from spatialdata.models import Labels3DModel  # noqa: F401
from igneous.task_creation.mesh import (  # noqa: F401
    create_meshing_tasks,
    create_sharded_multires_mesh_tasks,
    create_unsharded_multires_mesh_tasks,
)
from tissue_map_tools.igneous_converters import (  # noqa: F401
    from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes,
    from_precomputed_raster_to_precomputed_meshes,
    from_precomputed_raster_modify_scales_and_sharding,
)
from tissue_map_tools.converters import from_spatialdata_raster_to_precomputed_raster  # noqa: F401
from tissue_map_tools.view import (  # noqa: F401
    view_precomputed_in_neuroglancer,
)

##
f = Path(__file__).parent.parent.parent / "data" / "invasive_mask.ome.tiff"
if not f.exists():
    raise FileNotFoundError(
        f"File {f} does not exist. Please use symlinks to make the data available."
    )

##
data = imread(f)
data = data.rechunk((256, 256, 256))

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
from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
    raster=sdata["labels"],
    precomputed_path="/Users/macbook/Desktop/invasive_precomputed",
    units_factor=1000,
    # object_ids=list(range(1000)),
    shape=(128, 128, 128),
    nlod=3,
    min_chunk_size=(32, 32, 32),
)

##
viewer = view_precomputed_in_neuroglancer(
    data_path="/Users/macbook/Desktop/invasive_precomputed", mesh_ids=[5]
)

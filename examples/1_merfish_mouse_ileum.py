"""
Constructs the meshes from the volumetric data and from the 2.5D shapes.
"""

import napari_spatialdata.constants.config
import spatialdata as sd
from pathlib import Path
from tissue_map_tools.converters import (
    from_spatialdata_points_to_precomputed_points,
)

napari_spatialdata.constants.config.PROJECT_3D_POINTS_TO_2D = False
napari_spatialdata.constants.config.PROJECT_2_5D_SHAPES_TO_2D = False

##
# load the data
f = Path("/Users/macbook/Desktop/moffitt.zarr")
sdata = sd.read_zarr(f)
print(sd.get_extent(sdata["molecules"]))

##
# subset the data
molecules_cropped = sd.bounding_box_query(
    sdata["molecule_baysor"],
    axes=("x", "y", "z"),
    min_coordinate=[1500, 1500, -10],
    max_coordinate=[3000, 3000, 200],
    target_coordinate_system="global",
)
sdata["molecule_baysor"] = molecules_cropped

cells_baysor_cropped = sd.bounding_box_query(
    sdata["cells_baysor"],
    axes=("x", "y", "z"),
    min_coordinate=[1500, 1500, -10],
    max_coordinate=[3000, 3000, 200],
    target_coordinate_system="global",
)
sdata["cells_baysor"] = cells_baysor_cropped

##
# create the meshes from the volumetric data
##
# ome_zarr_path = sdata.path / "labels" / "dapi_labels"
#
# from_ome_zarr_04_raster_to_precomputed(
#     ome_zarr_path=str(ome_zarr_path),
#     precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
#     # is_labels=True,
# )

##

# from_spatialdata_raster_to_precomputed_raster(
#     raster=sdata["dapi_labels"],
#     precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
# )
# from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
#     raster=sdata["dapi_labels"],
#     precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
# )
from_spatialdata_points_to_precomputed_points(
    sdata["molecule_baysor"],
    precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
)

##
pass
# from napari_spatialdata import Interactive
#
# # plot the data
# interactive = Interactive(sdata, headless=True)
# # interactive.add_element('cells_layer_1_baysor', element_coordinate_system='global')
# interactive.add_element("cells_baysor", element_coordinate_system="global")
# # interactive.add_element("molecule_baysor", element_coordinate_system="global")
# interactive.run()

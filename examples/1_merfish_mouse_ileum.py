"""
Constructs the meshes from the volumetric data and from the 2.5D shapes.
"""

import napari_spatialdata.constants.config
import spatialdata as sd
from pathlib import Path


napari_spatialdata.constants.config.PROJECT_3D_POINTS_TO_2D = False
napari_spatialdata.constants.config.PROJECT_2_5D_SHAPES_TO_2D = False

##
# load the data
f = Path("/Users/macbook/Desktop/moffitt.zarr")
sdata = sd.read_zarr(f)
# print(sd.get_extent(sdata["molecules"]))

##
# # subset the data
molecules_cropped = sd.bounding_box_query(
    sdata["molecule_baysor"],
    axes=("x", "y", "z"),
    min_coordinate=[1500, 1500, -10],
    max_coordinate=[3000, 3000, 200],
    target_coordinate_system="global",
)
sdata["molecule_baysor"] = molecules_cropped
#
# cells_baysor_cropped = sd.bounding_box_query(
#     sdata["cells_baysor"],
#     axes=("x", "y", "z"),
#     min_coordinate=[1500, 1500, -10],
#     max_coordinate=[3000, 3000, 200],
#     target_coordinate_system="global",
# )
# sdata["cells_baysor"] = cells_baysor_cropped

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
from tissue_map_tools.converters import (
    # from_spatialdata_raster_to_precomputed_raster,
    from_spatialdata_points_to_precomputed_points,
)
from tissue_map_tools.igneous_converters import (
    from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes,
)

# from_spatialdata_raster_to_precomputed_raster(
#     raster=sdata["dapi_labels"],
#     precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
# )
# from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
#     raster=sdata["dapi_labels"],
#     precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
# )

# manual fix dtypes
##
import pandas as pd
import warnings
import numpy as np


def make_dtypes_compatible_with_precomputed_annotations(
    df: pd.DataFrame, max_categories: int = 1000, check_for_overflow: bool = True
) -> pd.DataFrame:
    """
    Convert the dtypes of the DataFrame to be compatible with precomputed annotations.
    """
    dtypes = set()
    old_column_order = df.columns.tolist()
    for column in df.columns:
        dtype = df[column].dtype
        dtypes.add(dtype.name)

    # Convert float columns to float32, checking for overflow
    for column in df.select_dtypes(include=["float64"]).columns:
        col = df[column]
        if check_for_overflow:
            min_float32, max_float32 = (
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
            )
            if (col.min() < min_float32) or (col.max() > max_float32):
                raise ValueError(
                    f"Column '{column}' has values outside float32 range! "
                    "Please check the data before converting."
                )
        df[column] = col.astype("float32")

    # Convert int columns to int32, checking for overflow
    for column in df.select_dtypes(include=["int64"]).columns:
        col = df[column]
        if check_for_overflow:
            min_int32, max_int32 = np.iinfo(np.int32).min, np.iinfo(np.int32).max
            if (col.min() < min_int32) or (col.max() > max_int32):
                raise ValueError(
                    f"Column '{column}' has values outside int32 range! "
                    "Please check the data before converting."
                )
        df[column] = col.astype("int32")

    # Convert object columns to category
    for column in df.select_dtypes(include=["object", "string"]).columns:
        n_unique = df[column].unique()
        if len(n_unique) > max_categories:
            warnings.warn(
                f"Column '{column}' has {len(n_unique)} unique values, which exceeds "
                f"the maximum of {max_categories}. "
                "Skipping conversion to category and dropping the column."
            )
        df[column] = df[column].astype("category")

    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype("int8")

    converted_dtypes = {"float64", "int64", "object", "string", "bool"}
    if len(dtypes.difference(converted_dtypes)) > 0:
        warnings.warn(
            f"Some columns have dtypes {dtypes.difference(converted_dtypes)} that are not "
            "converted to compatible types. Excluding these columns from the DataFrame."
        )
        df = df.select_dtypes(include=converted_dtypes)

    # Reorder columns to match the original order
    old_column_order = [col for col in old_column_order if col in df.columns]
    return df[old_column_order]


sdata["molecule_baysor"] = sd.models.PointsModel.parse(
    make_dtypes_compatible_with_precomputed_annotations(
        sdata["molecule_baysor"].compute(),
        max_categories=100,
        check_for_overflow=True,
    )
)


##


# TODO: there should be no need to add the subpath (we should be able to specify the
#  parent cloud volume object
# TODO: the info file in the parent volume should be updated to include the points
# TODO: the view APIs show include the points
import shutil

path = Path("/Users/macbook/Desktop/moffitt_precomputed/molecule_baysor")
if path.exists():
    shutil.rmtree(path)
from_spatialdata_points_to_precomputed_points(
    sdata["molecule_baysor"],
    precomputed_path="/Users/macbook/Desktop/moffitt_precomputed/molecule_baysor",
)

##
# pass
# from napari_spatialdata import Interactive
#
# # plot the data
# interactive = Interactive(sdata, headless=True)
# # interactive.add_element('cells_layer_1_baysor', element_coordinate_system='global')
# interactive.add_element("cells_baysor", element_coordinate_system="global")
# # interactive.add_element("molecule_baysor", element_coordinate_system="global")
# interactive.run()

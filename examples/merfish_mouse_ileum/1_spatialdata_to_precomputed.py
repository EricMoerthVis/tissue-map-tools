"""
Constructs the meshes from the volumetric data and from the 2.5D shapes.
"""

import napari_spatialdata.constants.config
import spatialdata as sd
from pathlib import Path
from numpy.random import default_rng
import pandas as pd
import warnings
import numpy as np
import shutil
import time

from tissue_map_tools.converters import (
    from_spatialdata_points_to_precomputed_points,
)


RNG = default_rng(42)

napari_spatialdata.constants.config.PROJECT_3D_POINTS_TO_2D = False
napari_spatialdata.constants.config.PROJECT_2_5D_SHAPES_TO_2D = False

##
# load the data
f = Path("/Users/macbook/Desktop/moffitt.zarr")
sdata = sd.read_zarr(f)
# print(sd.get_extent(sdata["molecules"]))

##
# subset the data
sdata_small = sd.bounding_box_query(
    sdata,
    axes=("x", "y", "z"),
    min_coordinate=[4000, 0, -10],
    max_coordinate=[5000, 1500, 200],
    target_coordinate_system="global",
)

transformation = sd.transformations.get_transformation(sdata_small["dapi"])
translation_vector = transformation.to_affine_matrix(
    input_axes=("x", "y", "z"), output_axes=("x", "y", "z")
)[:3, 3]
translation = sd.transformations.Translation(translation_vector, axes=("x", "y", "z"))
for _, element_name, _ in sdata_small.gen_spatial_elements():
    old_transformation = sd.transformations.get_transformation(
        sdata_small[element_name]
    )
    sequence = sd.transformations.Sequence([old_transformation, translation.inverse()])
    sd.transformations.set_transformation(
        sdata_small[element_name],
        transformation=sequence,
        to_coordinate_system="global",
    )
    transformed = sd.transform(sdata_small[element_name], to_coordinate_system="global")
    sdata_small[element_name] = transformed

sdata = sdata_small
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
# from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
#     raster=sdata["dapi_labels"],
#     precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
# )
# from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
#     raster=sdata["membrane_labels"],
#     precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
# )


##
# manual fix dtypes
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
            continue
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


subset = RNG.choice(len(sdata["molecule_baysor"]), 10000, replace=False)

print(sdata["molecule_baysor"].columns)
# subset_df = sdata["molecule_baysor"].compute().iloc[subset]
subset_df = sdata["molecule_baysor"].compute()
subset_df = subset_df[
    [
        # working
        "x",
        "y",
        "z",
        "gene",
        "area",
        #
        "mol_id",
        "x_raw",
        "y_raw",
        "z_raw",
        "brightness",
        "total_magnitude",
        "compartment",
        "nuclei_probs",
        "assignment_confidence",
        #
        "cell",
        "is_noise",  # TODO: bool not working at the moment
        # # "ncv_color",  # TODO: represent as RGB
        "layer",
    ]
]
sdata["molecule_baysor"] = sd.models.PointsModel.parse(
    make_dtypes_compatible_with_precomputed_annotations(
        subset_df,
        max_categories=250,
        check_for_overflow=True,
    )
)


##
# debug
points = sdata["molecule_baysor"].compute().iloc[:2]
print("point 0")
print(points.iloc[0])
print("")
print("point 1")
print(points.iloc[1])
print("")
print(points.x.dtype)
# print(points.gene.cat.categories)
print(points.gene.cat.categories.get_loc(points.gene.iloc[0]))
##
print("converting the points to the precomputed format")

# TODO: there should be no need to add the subpath (we should be able to specify the
#  parent cloud volume object
# TODO: the info file in the parent volume should be updated to include the points
# TODO: the view APIs show include the points

start = time.time()
path = Path("/Users/macbook/Desktop/moffitt_precomputed/molecule_baysor2")
if path.exists():
    shutil.rmtree(path)
from_spatialdata_points_to_precomputed_points(
    sdata["molecule_baysor"],
    precomputed_path="/Users/macbook/Desktop/moffitt_precomputed",
    points_name="molecule_baysor3",
    limit=1000,
    # limit=500,
)
print(f"conversion of points: {time.time() - start}")

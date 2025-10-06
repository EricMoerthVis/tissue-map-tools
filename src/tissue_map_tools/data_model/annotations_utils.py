import pandas as pd
from typing import Any
import warnings
import numpy as np
from cloudvolume import CloudVolume
from pathlib import Path
from tissue_map_tools.data_model.annotations import (
    AnnotationProperty,
    SUPPORTED_DTYPES,
    AnnotationInfo,
    read_spatial_index,
)


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

    converted_dtypes = {"float64", "int64", "object", "string", "bool", "category"}
    if len(dtypes.difference(converted_dtypes)) > 0:
        warnings.warn(
            f"Some columns have dtypes {dtypes.difference(converted_dtypes)} that are not "
            "converted to compatible types. Excluding these columns from the DataFrame."
        )
        df = df.select_dtypes(include=converted_dtypes)

    # Reorder columns to match the original order
    old_column_order = [col for col in old_column_order if col in df.columns]
    return df[old_column_order]


def from_pandas_column_to_annotation_property(
    df: pd.DataFrame, column: str
) -> AnnotationProperty:
    dtype = df[column].dtype
    enum_values = None
    enum_labels = None
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype} for column {column}. "
            f"Supported dtypes are: {SUPPORTED_DTYPES}"
        )
    if dtype == "category":
        enum_labels = df[column].cat.categories.tolist()
        enum_values = list(range(len(enum_labels)))
        type_ = df[column].cat.codes.dtype.name
    elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating):
        type_ = dtype.name
    else:
        raise ValueError(f"Unsupported dtype {dtype} for column {column}. ")
    return AnnotationProperty(
        id=column,
        type=type_,
        description="",
        enum_values=enum_values,
        enum_labels=enum_labels,
    )


def from_annotation_property_to_pandas_column(
    properties: list[AnnotationProperty],
) -> pd.DataFrame:
    data = {}
    for prop in properties:
        if prop.type in ["rgb", "rgba"]:
            raise NotImplementedError("RGB and RGBA types are not implemented yet.")
        else:
            if prop.enum_values is not None:
                codes = prop.enum_values
                if not np.array_equal(np.arange(len(codes)), codes):
                    raise ValueError(
                        f"Unable to intialize {prop.id} as a categorical column. Codes "
                        f"are not sequential integers starting from 0. Codes: {codes}"
                    )
                labels = prop.enum_labels
                cat = pd.Categorical(
                    values=[],
                    categories=labels,
                )
                series = pd.Series(cat, dtype="category")
            else:
                dtype = np.dtype(prop.type)
                series = pd.Series(dtype=dtype)
            data[prop.id] = series
    return pd.DataFrame(data=data)


def from_annotation_type_and_dimensions_to_pandas_column(
    annotation_type: str, dimensions: dict[str, Any]
) -> pd.DataFrame:
    spatial_columns = list(dimensions.keys())

    if annotation_type == "POINT":
        coords = spatial_columns
        dtypes = {coord: "float32" for coord in coords}
    else:
        raise NotImplementedError(f"Unsupported annotation type: {annotation_type}")
    return pd.DataFrame(
        data={col: pd.Series(dtype=dtype) for col, dtype in dtypes.items()}
    )


def parse_annotations(data_path: Path) -> pd.DataFrame:
    cv = CloudVolume(str(data_path))
    annotations = cv.info["annotations"]
    annotations_path = data_path / annotations
    annotations_info_file = annotations_path / "info"

    with open(annotations_info_file, "r") as f:
        annotations_info = AnnotationInfo.model_validate_json(f.read())

    spatial_index = {}
    for spatial in annotations_info.spatial:
        key = spatial.key
        si = read_spatial_index(
            info=annotations_info, root_path=annotations_path, spatial_key=key
        )
        spatial_index[key] = si
    df_positions = from_annotation_type_and_dimensions_to_pandas_column(
        annotation_type=annotations_info.annotation_type,
        dimensions=annotations_info.dimensions,
    )
    df_properties = from_annotation_property_to_pandas_column(
        annotations_info.properties
    )
    # for the dtypes
    df_merged = pd.concat(
        [
            df_positions,
            df_properties,
            pd.DataFrame(
                {
                    "__spatial_index__": pd.Series([], dtype=object),
                    "__chunk_key__": pd.Series([], dtype=object),
                    "index": pd.Series([], dtype=np.uint64),
                }
            ),
        ],
        axis=1,
    )

    df_data: dict[str, Any] = {
        col: [] for col in list(df_positions.columns) + list(df_properties.columns)
    } | {"index": []}
    df_data["__spatial_index__"] = []
    df_data["__chunk_key__"] = []
    for key, si in spatial_index.items():
        for chunk_key, data in si.items():
            for annotation_id, positions, properties in data:
                df_data["index"].append(annotation_id)
                for col, position in zip(df_positions.columns, positions):
                    df_data[col].append(position)
                for property_key, property_value in properties.items():
                    if df_properties[property_key].dtype == "category":
                        property_category = df_properties[property_key].cat.categories[
                            property_value
                        ]
                        df_data[property_key].append(property_category)
                    else:
                        df_data[property_key].append(property_value)
                df_data["__spatial_index__"].append(key)
                df_data["__chunk_key__"].append(chunk_key)

    # set the correct dtypes
    # TODO: this could be done directly (earlier) when creating the df_data dict
    df_data = {k: pd.Series(v, dtype=df_merged[k].dtype) for k, v in df_data.items()}

    df = pd.DataFrame(data=df_data).set_index("index")
    df.index.name = None
    return df

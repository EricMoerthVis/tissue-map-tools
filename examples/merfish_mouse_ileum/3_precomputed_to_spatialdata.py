import pandas as pd
from typing import Any
from pathlib import Path
from cloudvolume import CloudVolume
from tissue_map_tools.data_model.annotations import read_spatial_index, AnnotationInfo
from tissue_map_tools.data_model.annotations_utils import (
    from_annotation_property_to_pandas_column,
    from_annotation_type_and_dimensions_to_pandas_column,
)

if __name__ == "__main__":
    data_path = Path("/Users/macbook/Desktop/moffitt_precomputed")
    cv = CloudVolume(str(data_path))

    # parse raster

    # parse meshes

    # parse annotations
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
                    df_data[property_key].append(property_value)
                df_data["__spatial_index__"].append(key)
                df_data["__chunk_key__"].append(chunk_key)

    df = pd.DataFrame(data=df_data).set_index("index")
    pass

from pathlib import Path
from tissue_map_tools.data_model.annotations_utils import parse_annotations


if __name__ == "__main__":
    data_path = Path("/Users/macbook/Desktop/moffitt_precomputed")

    # parse raster

    # parse meshes

    # parse annotations
    df_annotations = parse_annotations(data_path)
    print(df_annotations)
    print(df_annotations["x"].max())
    print(df_annotations["y"].max())
    print(df_annotations["z"].max())
    print(df_annotations["x"].min())
    print(df_annotations["y"].min())
    print(df_annotations["z"].min())

    pass

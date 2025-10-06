from pathlib import Path
from tissue_map_tools.data_model.annotations_utils import parse_annotations

out_path = Path(__file__).parent.parent.parent / "out"
precomputed_path = out_path / "merfish_mouse_ileum_precomputed"

if __name__ == "__main__":
    data_path = precomputed_path

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
    print(df_annotations.iloc[0])

    pass

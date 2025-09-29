from tissue_map_tools.view import (
    view_precomputed_in_napari,
)

# viewer = view_precomputed_in_neuroglancer(
#     data_path="/Users/macbook/Desktop/moffitt_precomputed",
# )
view_precomputed_in_napari(
    data_path="/Users/macbook/Desktop/moffitt_precomputed",
    show_meshes=False,
    show_raster=True,
    show_points=True,
)

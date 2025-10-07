import os
from os import path as path
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import trimesh
import zarr
from ome_zarr.io import parse_url
from skimage import measure

from experiments.subvolume_analysis import _calc_mask_to_brick
from experiments.mesher_third_party import mesher_util
from experiments.mesher_third_party.mesher import (
    generate_decimated_meshes,
    generate_neuroglancer_multires_mesh,
)


def get_meshes(mask_path, out_path, csv_out, entity_name, smoothing=0, test=False):
    """
    Creating meshes to a given 3D OME-NGFF Segmentation Mask Volume

    Parameters
    ----------
    mask_path: Path to the 3D OME-NGFF Segmentation mask
    out_path: Output path to store the Mesh Files at
    csv_out: File Path where to save the resulting csv file with basic statistics for each entity
    entity_name: Entity name describing which entities the segmentation mask represents
    smoothing: Smoothing parameter to get a more biological meaningful representation - value represents how many iterations of laplacian smoothing is applied
    test: If set to true only the first mesh is generated in order to test parameters like smoothing

    Returns
    -------
    Creates individual mesh files in .obj format for each entity in the given segmentation volume
    """
    if not path.exists(out_path):
        os.makedirs(out_path)

    # Accessing the data as store:
    root = parse_url(mask_path, mode="r")
    store = root.store
    data = zarr.open(store).get("0")

    print("Initialising the Mesh Creation")
    # Getting the bricks for each mask
    mask_to_brick = _calc_mask_to_brick(data)
    chunk_shape = data.chunks

    columns = ["id", "voxels", "chunk_keys"]
    dataOut = []

    print("Starting the Object Creation")
    i = 0
    for _key in mask_to_brick:
        i = i + 1
        b = str(i) + "/" + str(len(mask_to_brick))
        print("\r", b, end="")
        # print(_key, mask_to_brick[_key])
        chunk_keys = mask_to_brick[_key]
        all_start_coords = []
        all_end_coords = []
        # Loop over each chunk key to calculate start and end coordinates
        for key in chunk_keys:
            # Parse the chunk indices from the chunk key
            chunk_indices = tuple(map(int, key.split(".")))
            # Calculate the start and end coordinates for each dimension
            start_coords = [
                index * size for index, size in zip(chunk_indices, chunk_shape)
            ]
            end_coords = [
                start + size for start, size in zip(start_coords, chunk_shape)
            ]
            # Append to lists
            all_start_coords.append(start_coords)
            all_end_coords.append(end_coords)

        # Find the minimum start and maximum end coordinates across all chunks
        combined_start = [min(coords) for coords in zip(*all_start_coords)]
        combined_end = [max(coords) for coords in zip(*all_end_coords)]
        combined_slices = tuple(
            slice(start, end) for start, end in zip(combined_start, combined_end)
        )
        combined_data = data.get_basic_selection(combined_slices)[0, 0, :, :, :]
        combined_data = np.pad(
            combined_data, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0
        )
        mask = np.where(combined_data == _key, 1, 0)

        v2, f2, n2, values2 = measure.marching_cubes(volume=mask)

        new_obj = open(
            out_path + "/" + str(entity_name) + "_" + str(_key) + ".obj", "w"
        )
        f2 = f2 + 1
        for item in v2:
            new_obj.write(
                "v {0} {1} {2}\n".format(
                    item[0] + combined_start[2],
                    item[1] + combined_start[3],
                    item[2] + combined_start[4],
                )
            )
        for item in n2:
            new_obj.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

        for item in f2:
            new_obj.write(
                "f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2])
            )

        new_obj.close()

        if smoothing > 0:
            mesh = o3d.io.read_triangle_mesh(
                out_path + "/" + str(entity_name) + "_" + str(_key) + ".obj"
            )
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=smoothing)
            o3d.io.write_triangle_mesh(
                out_path + "/" + str(entity_name) + "_" + str(_key) + ".obj", mesh
            )

        out = [int(_key), int(np.count_nonzero(mask)), chunk_keys]
        dataOut.append(out)
        df = pd.DataFrame(dataOut, columns=columns)
        df.to_csv(csv_out, index=False, header=True)

        if test:
            print("Ran as Test")
            break
    print("\r", "done", end="")


def get_meshes_ng(mask_path, out_path, csv_out, entity_name, smoothing=0, test=False):
    """
    Creating meshes to a given 3D OME-NGFF Segmentation Mask Volume

    Parameters
    ----------
    mask_path: Path to the 3D OME-NGFF Segmentation mask
    out_path: Output path to store the Mesh Files at
    csv_out: File Path where to save the resulting csv file with basic statistics for each entity
    entity_name: Entity name describing which entities the segmentation mask represents
    smoothing: Smoothing parameter to get a more biological meaningful representation - value represents how many iterations of laplacian smoothing is applied
    test: If set to true only the first mesh is generated in order to test parameters like smoothing

    Returns
    -------
    Creates individual mesh files in .obj format for each entity in the given
    segmentation volume
    """
    if not path.exists(out_path):
        os.makedirs(out_path)

    os.makedirs(f"{out_path}/", exist_ok=True)
    temp_mesh_dir = Path(f"{out_path}/")

    # Accessing the data as store:
    root = parse_url(mask_path, mode="w")
    store = root.store
    data = zarr.open(store).get("0")

    print("Initialising the Mesh Creation")
    # Getting the bricks for each mask
    mask_to_brick = _calc_mask_to_brick(data)
    chunk_shape = data.chunks

    print("Starting the Object Creation")
    i = 0
    for _key in mask_to_brick:
        i = i + 1
        b = str(i) + "/" + str(len(mask_to_brick))
        print("\r", b, end="")
        # print(_key, mask_to_brick[_key])
        chunk_keys = mask_to_brick[_key]
        all_start_coords = []
        all_end_coords = []
        # Loop over each chunk key to calculate start and end coordinates
        for key in chunk_keys:
            # Parse the chunk indices from the chunk key
            chunk_indices = tuple(map(int, key.split(".")))
            # Calculate the start and end coordinates for each dimension
            start_coords = [
                index * size for index, size in zip(chunk_indices, chunk_shape)
            ]
            end_coords = [
                start + size for start, size in zip(start_coords, chunk_shape)
            ]
            # Append to lists
            all_start_coords.append(start_coords)
            all_end_coords.append(end_coords)

        # Find the minimum start and maximum end coordinates across all chunks
        combined_start = [min(coords) for coords in zip(*all_start_coords)]
        combined_end = [max(coords) for coords in zip(*all_end_coords)]
        combined_slices = tuple(
            slice(start, end) for start, end in zip(combined_start, combined_end)
        )
        combined_data = data.get_basic_selection(combined_slices)[0, 0, :, :, :]
        combined_data = np.pad(
            combined_data, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0
        )
        mask = np.where(combined_data == _key, 1, 0)

        v2, f2, n2, values2 = measure.marching_cubes(volume=mask)

        # Bug: trimesh_mesh and the objects created from it are not used
        trimesh_mesh = trimesh.Trimesh(vertices=v2, faces=f2)

        # Apply smoothing
        print(f"Smoothing mesh {_key} ({len(trimesh_mesh.vertices)} vertices)...")
        trimesh_mesh = trimesh_mesh.smoothed(lamb=0.75, iterations=smoothing)

        # Fix any mesh issues
        if not trimesh_mesh.is_watertight:
            print(f"Fixing non-watertight mesh {_key}...")
            trimesh_mesh.fill_holes()

        trimesh_mesh.fix_normals()
        v2 = trimesh_mesh.vertices
        f2 = trimesh_mesh.faces

        print("generate decimated mesh")
        generate_decimated_meshes(
            temp_mesh_dir,
            list(range(3)),
            [{"vertices": v2, "faces": f2, "id": _key}],
            4,
            10,
        )
        print("generate neuroglancer multires mesh")
        generate_neuroglancer_multires_mesh(
            output_path=temp_mesh_dir,
            id=_key,
            lods=list(range(3)),
            original_ext=".ply",
            lod_0_box_size=8,  # TODO: check
        )

        print("writing info metadata")
        multires_output_path = f"{temp_mesh_dir}/multires"
        mesher_util.write_segment_properties_file(multires_output_path)
        mesher_util.write_info_file(multires_output_path)

        if test:
            print("Ran as Test")
            break
    print("\r", "done", end="")


if __name__ == "__main__":
    get_meshes_ng(
        "http://127.0.0.1:8080",
        "/Users/ericmoerth/ws/tissue-map-tools/out",
        "/Users/ericmoerth/ws/tissue-map-tools/out/out.csv",
        "gloms",
        10,
        True,
    )

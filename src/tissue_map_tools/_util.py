from ome_zarr.io import parse_url
import pandas as pd
import math
import zarr
from itertools import product
from skimage import measure
import open3d as o3d
import scipy
import os.path as path
import requests
import ome_types
import numpy as np
from pathlib import Path
import trimesh
import pyfqmr
import os
import urllib.parse
from trimesh.intersections import slice_faces_plane
import subprocess


def stats(array):
    """
    Internal function to calculate statistics from a given 3D Array

    Parameters
    ----------
    array : the 3D numpy array

    Returns
    -------
    map[float]: Collection of Statistics for the given Array

    Examples
    --------
    >>> stats(arr)
    """
    arr = array[array != 0]
    min_val = np.min(arr)
    max_val = np.max(arr)
    return map(float, (
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        np.sum(arr),
        max_val - min_val,
        np.percentile(arr, 5),
        np.percentile(arr, 25),
        np.percentile(arr, 50),
        np.percentile(arr, 75),
        np.percentile(arr, 95),
        scipy.stats.skew(arr.flatten()),
        scipy.stats.kurtosis(arr.flatten()),
    ))


def sub_volume_analysis(mask_path, raw_path, ome_xml_path, csv_out, mask_generation_res='0'):
    """
    Performing sub volume analysis for the given segmentation mask and the raw original volume

    Parameters
    ----------
    mask_path: Path to the 3D OME-NGFF Segmentation mask
    raw_path: Path to the raw original 3D OME-NGFF data
    csv_out: File Path where to save the resulting csv file
    mask_generation_res: The resolution level at which the segmentation mask was generated from to match with the raw volume data

    Returns
    -------
    Writes the statistics for each entity into the given csv output path
    """
    root = parse_url(mask_path, mode="w")
    store = root.store
    data = zarr.open(store).get('0')

    root_data = parse_url(raw_path, mode="w")
    store_data = root_data.store
    data_raw = zarr.open(store_data).get(
        mask_generation_res)  # depends on the resolution at which the masks were generated


    parsed = urllib.parse.urlparse(ome_xml_path)
    if parsed.scheme == '':
        # local file
        with open(ome_xml_path, 'r') as file:
            content = file.read()
    else:
        # remote file
        response = requests.get(ome_xml_path)
        content = response.text.replace("Â", "")
    ome_xml = ome_types.from_xml(content)
    channel_names = [c.name for c in ome_xml.images[0].pixels.channels]

    # Read the csv file and get the already worked IDs
    worked_ids = list([])
    dataOut = []
    if path.exists(csv_out):
        df = pd.read_csv(csv_out)
        df['id'] = df['id'].astype(int)  # Ensure column A is integer
        worked_ids = list(df['id'])
        dataOut = df.to_numpy(dtype='object').tolist()

    chunk_shape = data.chunks
    columns = ["id", "voxels", "chunk_keys"]
    for c in ome_xml.images[0].pixels.channels:
        # Grab the original data
        channel = int(c.id.split(":")[2])
        columns.append(str(channel) + "_mean")
        columns.append(str(channel) + "_std")
        columns.append(str(channel) + "_min")
        columns.append(str(channel) + "_max")
        columns.append(str(channel) + "_sum")
        columns.append(str(channel) + "_range")
        columns.append(str(channel) + "_p5")
        columns.append(str(channel) + "_p25")
        columns.append(str(channel) + "_p50")
        columns.append(str(channel) + "_p75")
        columns.append(str(channel) + "_p95")
        columns.append(str(channel) + "_skew")
        columns.append(str(channel) + "_kurtosis")

    # Getting the bricks for each mask
    print("Initialising the Statistics Calcuation")
    mask_to_brick = _calc_mask_to_brick(data)
    chunk_shape = data.chunks

    i = 0
    for _key in mask_to_brick:
        i = i + 1
        if (_key not in worked_ids):
            b = str(i) + "/" + str(len(mask_to_brick))
            print("\r", b, end="")
            # print(_key, mask_to_brick[_key])
            chunk_keys = mask_to_brick[_key]
            all_start_coords = []
            all_end_coords = []
            # Loop over each chunk key to calculate start and end coordinates
            for key in chunk_keys:
                # Parse the chunk indices from the chunk key
                chunk_indices = tuple(map(int, key.split('.')))
                # Calculate the start and end coordinates for each dimension
                start_coords = [index * size for index, size in zip(chunk_indices, chunk_shape)]
                end_coords = [start + size for start, size in zip(start_coords, chunk_shape)]
                # Append to lists
                all_start_coords.append(start_coords)
                all_end_coords.append(end_coords)

            # Find the minimum start and maximum end coordinates across all chunks
            combined_start = [min(coords) for coords in zip(*all_start_coords)]
            combined_end = [max(coords) for coords in zip(*all_end_coords)]

            try:
                # Create the slices for combined selection
                combined_slices = tuple(slice(start, end) for start, end in zip(combined_start, combined_end))
                # Retrieve the data for the combined region
                combined_data = data.get_basic_selection(combined_slices)[0, 0, :, :, :]
                mask = np.where(combined_data == _key, 1, 0)

                ## For all Channels get the original data:
                combined_end[1] = len(channel_names)
                combined_slices = tuple(slice(start, end) for start, end in zip(combined_start, combined_end))
                rawData = data_raw.get_basic_selection(combined_slices)[0, :, :, :, :]

                out = [int(_key), int(np.count_nonzero(mask)), chunk_keys]
                for c in ome_xml.images[0].pixels.channels:
                    # Grab the original data
                    channel = int(c.id.split(":")[2])
                    channel_data = rawData[channel, :, :, :]
                    masked = channel_data * mask

                    stats_out = stats(masked)
                    out.extend([float(x) for x in list(stats_out)])

                dataOut.append(out)
                df = pd.DataFrame(dataOut, columns=columns)
                df.to_csv(csv_out, index=False, header=True)
            except:
                print("Error", _key)
    print("done")


def _chunk_indices(arr):
    """
    Internal function to retrieve chunk indices for zarr retrieval

    Parameters
    ----------
    arr: given array of data

    Returns
    -------
    chunk index of the array
    """
    chunks = [math.ceil(ds / cs) for ds, cs in zip(arr.shape, arr.chunks)]
    return product(*(range(nc) for nc in chunks))


def _calc_mask_to_brick(data):
    """
    Internal function to calculate which Zarr bricks need to be loaded for each segmentation mask entity

    Parameters
    ----------
    data: the 3D OME-NGFF segmentation volume

    Returns
    -------
    map between segmentation entities and the Zarr bricks they are located at
    """
    mask_to_brick = {}
    for chunk_coord in _chunk_indices(data):
        t, c, _z, _y, _x = chunk_coord
        chunk_key = ".".join(map(str, chunk_coord))
        chunk = data.get_block_selection(chunk_coord)
        unique_values = np.unique(chunk)
        if unique_values.size > 1:
            for val in unique_values[1:]:
                if val not in mask_to_brick:
                    mask_to_brick[val] = [chunk_key]
                else:
                    curr_val = mask_to_brick[val]
                    curr_val.append(chunk_key)
                    mask_to_brick[val] = curr_val

    return mask_to_brick


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
    Creates indiviudal mesh files in .obj format for each entity in the given segmentation volume
    """
    if not path.exists(out_path):
        os.makedirs(out_path)

    # Accessing the data as store:
    root = parse_url(mask_path, mode="r")
    store = root.store
    data = zarr.open(store).get('0')

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
            chunk_indices = tuple(map(int, key.split('.')))
            # Calculate the start and end coordinates for each dimension
            start_coords = [index * size for index, size in zip(chunk_indices, chunk_shape)]
            end_coords = [start + size for start, size in zip(start_coords, chunk_shape)]
            # Append to lists
            all_start_coords.append(start_coords)
            all_end_coords.append(end_coords)

        # Find the minimum start and maximum end coordinates across all chunks
        combined_start = [min(coords) for coords in zip(*all_start_coords)]
        combined_end = [max(coords) for coords in zip(*all_end_coords)]
        combined_slices = tuple(slice(start, end) for start, end in zip(combined_start, combined_end))
        combined_data = data.get_basic_selection(combined_slices)[0, 0, :, :, :]
        combined_data = np.pad(combined_data, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        mask = np.where(combined_data == _key, 1, 0)

        try:
            v2, f2, n2, values2 = measure.marching_cubes(volume=mask)

            new_obj = open(out_path + "/" + str(entity_name) + "_" + str(_key) + '.obj', 'w')
            f2 = f2 + 1
            for item in v2:
                new_obj.write("v {0} {1} {2}\n".format(item[0] + combined_start[2], item[1] + combined_start[3],
                                                       item[2] + combined_start[4]))
            for item in n2:
                new_obj.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

            for item in f2:
                new_obj.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))

            new_obj.close()

            if smoothing > 0:
                mesh = o3d.io.read_triangle_mesh(out_path + "/" + str(entity_name) + "_" + str(_key) + '.obj')
                mesh = mesh.filter_smooth_laplacian(number_of_iterations=smoothing)
                o3d.io.write_triangle_mesh(out_path + "/" + str(entity_name) + "_" + str(_key) + '.obj', mesh)

            out = [int(_key), int(np.count_nonzero(mask)), chunk_keys]
            dataOut.append(out)
            df = pd.DataFrame(dataOut, columns=columns)
            df.to_csv(csv_out, index=False, header=True)
        except:
            print("Error with", _key)

        if (test):
            print("Ran as Test")
            break
    print("\r", "done", end="")


''' Functions taken over from "multiresolution-mesh-creator'''


def pyfqmr_decimate(vertices, faces, output_path, id, lod, decimation_factor,
                    aggressiveness):
    """Mesh decimation using pyfqmr."""
    desired_faces = max(len(faces) // (decimation_factor ** lod), 4)
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(vertices, faces)
    del vertices
    del faces
    mesh_simplifier.simplify_mesh(target_count=desired_faces,
                                  aggressiveness=aggressiveness,
                                  preserve_border=False,
                                  verbose=False)
    vertices, faces, _ = mesh_simplifier.getMesh()
    del mesh_simplifier

    mesh = trimesh.Trimesh(vertices, faces)
    del vertices
    del faces
    _ = mesh.export(f"{output_path}/s{lod}/{id}.ply")


def generate_decimated_meshes(output_path, lods, infos,
                              decimation_factor, aggressiveness):
    """Generate decimatated meshes for all ids in `ids`, over all lod in `lods`.
    """

    results = []
    for current_lod in lods:
        os.makedirs(f"{output_path}/mesh_lods/s{current_lod}", exist_ok=True)
        for info in infos:
            if current_lod == 0:
                ## Write out the original ply to file
                mesh = trimesh.Trimesh(info.get("vertices"), info.get("faces"))
                _ = mesh.export(f"{output_path}/mesh_lods/s{current_lod}/{info.get('id')}.ply")
            else:
                # Getting the vertices and faces for the mesh
                pyfqmr_decimate(info.get("vertices"), info.get("faces"),
                                f"{output_path}/mesh_lods",
                                info.get("id"), current_lod,
                                decimation_factor,
                                aggressiveness)


def my_slice_faces_plane(vertices, faces, plane_normal, plane_origin):
    """Wrapper for trimesh slice_faces_plane to catch error that happens if the
    whole mesh is to one side of the plane.

    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        plane_normal: Normal of plane
        plane_origin: Origin of plane

    Returns:
        vertices, faces: Vertices and faces
    """

    if len(vertices) > 0 and len(faces) > 0:
        try:
            vertices, faces, uv = slice_faces_plane(vertices, faces, plane_normal,
                                                    plane_origin)
        except ValueError as e:
            if str(e) != "input must be 1D integers!":
                raise
            else:
                pass

    return vertices, faces


def update_fragment_dict(dictionary, fragment_pos, vertices, faces,
                         lod_0_fragment_pos):
    """Update dictionary (in place) whose keys are fragment positions and
    whose values are `Fragment` which is a class containing the corresponding
    fragment vertices, faces and corresponding lod 0 fragment positions.

    This is necessary since each fragment (above lod 0) must be divisible by a
    2x2x2 grid. So each fragment is technically split into many "subfragments".
    Thus the dictionary is used to map all subfragments to the proper parent
    fragment. The lod 0 fragment positions are used when writing out the index
    files because if a subfragment is empty it still needs to be included in
    the index file. By tracking all the corresponding lod 0 fragments of a
    given lod fragment, we can ensure that all necessary empty fragments are
    included.

    Args:
        dictionary: Dictionary of fragment pos keys and fragment info values
        fragment_pos: Current lod fragment position
        vertices: Vertices
        faces: Faces
        lod_0_fragment_pos: Corresponding lod 0 fragment positions
                            corresponding to fragment_pos
    """

    if fragment_pos in dictionary:
        fragment = dictionary[fragment_pos]
        fragment.update(vertices, faces, lod_0_fragment_pos)
        dictionary[fragment_pos] = fragment
    else:
        dictionary[fragment_pos] = mesh_util.Fragment(vertices, faces,
                                                      [lod_0_fragment_pos])


def _encode_mesh_draco(vertices: np.ndarray, faces: np.ndarray,
                       bounds_min: np.ndarray, box_size: float, position_quantization_bits, temp_dir) -> bytes:
    """Encode a mesh using Google's Draco encoder with enhanced quality settings."""
    # Normalize vertices to quantization range
    vertices = vertices.copy()

    # Improved quantization to prevent vertex snapping at boundaries
    # Add a slight padding to avoid precision issues at the boundaries
    # padding = box_size * 0.001  # 0.1% padding
    bounds_min = bounds_min  # - padding
    # box_size = box_size # + 2 * padding
    #
    vertices -= bounds_min
    vertices /= box_size
    vertices *= (2 ** position_quantization_bits - 1)
    # vertices = vertices.astype(np.int32)

    # Create temporary files for the mesh
    obj_path = os.path.join(temp_dir, "temp.obj")
    drc_path = os.path.join(temp_dir, "temp.drc")

    # Write simple OBJ file
    with open(obj_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces + 1:  # OBJ indices are 1-based
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    # Get the appropriate draco_encoder path
    if os.path.exists("/opt/homebrew/bin/draco_encoder"):
        draco_path = "/opt/homebrew/bin/draco_encoder"
    elif os.path.exists("/usr/bin/draco_encoder"):
        draco_path = "/usr/bin/draco_encoder"
    elif os.path.exists("/usr/local/bin/draco_encoder"):
        draco_path = "/usr/local/bin/draco_encoder"
    else:
        raise RuntimeError("Cannot find draco_encoder. Please install it.")

    # Run draco_encoder
    cmd = [
        draco_path,
        "-i", obj_path,
        "-o", drc_path,
        "-qp", str(position_quantization_bits),
        "-qt", "8",  # Higher quality tangents
        "-qn", "8",  # Higher quality normals
        "-qtx", "8",  # Higher quality texture coordinates
        "-cl", "10",  # Maximum compression level
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(drc_path, "rb") as f:
            return f.read()

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Draco encoding failed: {e.stdout} {e.stderr}")


def generate_mesh_decomposition(mesh_path, lod_0_box_size, grid_origin,
                                start_fragment, end_fragment, current_lod,
                                num_chunks):
    """Dask delayed function to decompose a mesh, provided as vertices and
    faces, into fragments of size lod_0_box_size * 2**current_lod. Each
    fragment is also subdivided by 2x2x2. This is performed over a limited
    xrange in order to parallelize via dask.

    Args:
        mesh_path: Path to current lod mesh
        lod_0_box_size: Base chunk shape
        grid_origin: The lod 0 mesh grid origin
        start_fragment: Start fragment position (x,y,z)
        end_fragment: End fragment position (x,y,z)
        x_start: Starting x position for this dask task
        x_end: Ending x position for this dask task
        current_lod: The current level of detail

    Returns:
        fragments: List of `CompressedFragments` (named tuple)
    """
    vertices, faces = mesh_util.mesh_loader(mesh_path)

    combined_fragments_dictionary = {}
    fragments = []

    nyz, nxz, nxy = np.eye(3)

    if current_lod != 0:
        # Want each chunk for lod>0 to be divisible by 2x2x2 region,
        # so multiply coordinates by 2
        start_fragment *= 2
        end_fragment *= 2

        # 2x2x2 subdividing box size
        sub_box_size = lod_0_box_size * 2 ** (current_lod - 1)
    else:
        sub_box_size = lod_0_box_size

    vertices -= grid_origin

    n = np.eye(3)
    for dimension in range(3):
        if num_chunks[dimension] > 1:
            n_d = n[dimension, :]
            plane_origin = n_d * end_fragment[dimension] * sub_box_size
            vertices, faces = my_slice_faces_plane(vertices, faces, -n_d,
                                                   plane_origin)
            if len(vertices) == 0:
                return None
            plane_origin = n_d * start_fragment[dimension] * sub_box_size
            vertices, faces = my_slice_faces_plane(vertices, faces, n_d,
                                                   plane_origin)

    if len(vertices) == 0:
        return None

    # Get chunks of desired size by slicing in x,y,z and ensure their chunks
    # are divisible by 2x2x2 chunks
    for x in range(start_fragment[0], end_fragment[0]):
        plane_origin_yz = nyz * (x + 1) * sub_box_size
        vertices_yz, faces_yz = my_slice_faces_plane(vertices, faces, -nyz,
                                                     plane_origin_yz)

        for y in range(start_fragment[1], end_fragment[1]):
            plane_origin_xz = nxz * (y + 1) * sub_box_size
            vertices_xz, faces_xz = my_slice_faces_plane(
                vertices_yz, faces_yz, -nxz, plane_origin_xz)

            for z in range(start_fragment[2], end_fragment[2]):
                plane_origin_xy = nxy * (z + 1) * sub_box_size
                vertices_xy, faces_xy = my_slice_faces_plane(
                    vertices_xz, faces_xz, -nxy, plane_origin_xy)

                lod_0_fragment_position = tuple(np.array([x, y, z]))
                if current_lod != 0:
                    fragment_position = tuple(np.array([x, y, z]) // 2)
                else:
                    fragment_position = lod_0_fragment_position

                update_fragment_dict(combined_fragments_dictionary,
                                     fragment_position, vertices_xy, faces_xy,
                                     list(lod_0_fragment_position))

                vertices_xz, faces_xz = my_slice_faces_plane(
                    vertices_xz, faces_xz, nxy, plane_origin_xy)

            vertices_yz, faces_yz = my_slice_faces_plane(
                vertices_yz, faces_yz, nxz, plane_origin_xz)

        vertices, faces = my_slice_faces_plane(vertices, faces, nyz,
                                               plane_origin_yz)

    # Return combined_fragments_dictionary
    for fragment_pos, fragment in combined_fragments_dictionary.items():
        if fragment.vertices.size > 0:
            current_box_size = lod_0_box_size * 2 ** current_lod
            # draco_bytes = encode_faces_to_custom_drc_bytes(
            #     fragment.vertices,
            #     np.zeros(np.shape(fragment.vertices)),
            #     fragment.faces,
            #     np.asarray(3 * [current_box_size]),
            #     np.asarray(fragment_pos) * current_box_size,
            #     position_quantization_bits=10)

            draco_bytes = _encode_mesh_draco(fragment.vertices, fragment.faces,
                                             np.asarray(3 * [current_box_size]),
                                             np.asarray(fragment_pos) * current_box_size, position_quantization_bits=10,
                                             temp_dir=os.path.dirname(mesh_path))

            if len(draco_bytes) > 12:
                # Then the mesh is not empty
                fragment = mesh_util.CompressedFragment(
                    draco_bytes, np.asarray(fragment_pos), len(draco_bytes),
                    np.asarray(fragment.lod_0_fragment_pos))
                fragments.append(fragment)

    return fragments


def generate_neuroglancer_multires_mesh(output_path, id, lods,
                                        original_ext, lod_0_box_size):
    """function to generate multiresolution mesh in neuroglancer
    mesh format using prewritten meshes at different levels of detail.
    """
    os.makedirs(f"{output_path}/multires", exist_ok=True)
    os.system(
        f"rm -rf {output_path}/multires/{id} {output_path}/multires/{id}.index"
    )

    results = []
    for idx, current_lod in enumerate(lods):
        if current_lod == 0:
            mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}{original_ext}"
        else:
            mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}.ply"

        vertices, _ = mesh_util.mesh_loader(mesh_path)

        if vertices is not None:

            if current_lod == 0:
                max_box_size = lod_0_box_size * 2 ** lods[-1]
                grid_origin = (vertices.min(axis=0) // max_box_size -
                               1) * max_box_size
            vertices -= grid_origin

            current_box_size = lod_0_box_size * 2 ** current_lod
            start_fragment = np.maximum(
                vertices.min(axis=0) // current_box_size - 1,
                np.array([0, 0, 0])).astype(int)
            end_fragment = (vertices.max(axis=0) // current_box_size +
                            1).astype(int)

            del vertices

            # Want to divide the mesh up into upto num_workers chunks. We do
            # that by first subdividing the largest dimension as much as
            # possible, followed by the next largest dimension etc so long
            # as we don't exceed num_workers slices. If we instead slice each
            # dimension once, before slicing any dimension twice etc, it would
            # increase the number of mesh slice operations we perform, which
            # seems slow.

            max_number_of_chunks = (end_fragment - start_fragment)
            dimensions_sorted = np.argsort(-max_number_of_chunks)
            num_chunks = np.array([1, 1, 1])

            for d in dimensions_sorted:
                if num_chunks[d] < max_number_of_chunks[d]:
                    num_chunks[d] += 1

            stride = np.ceil(1.0 * (end_fragment - start_fragment) /
                             num_chunks).astype(np.int32)

            # Scattering here, unless broadcast=True, causes this issue:
            # https://github.com/dask/distributed/issues/4612. But that is
            # slow so we are currently electing to read the meshes each time
            # within generate_mesh_decomposition.
            # vertices_to_send = client.scatter(vertices, broadcast=True)
            # faces_to_send = client.scatter(faces, broadcast=True)

            decomposition_results = []
            for x in range(start_fragment[0], end_fragment[0], stride[0]):
                for y in range(start_fragment[1], end_fragment[1], stride[1]):
                    for z in range(start_fragment[2], end_fragment[2],
                                   stride[2]):
                        current_start_fragment = np.array([x, y, z])
                        current_end_fragment = current_start_fragment + stride
                        # then we aren't parallelizing again
                        decomposition_results.append(
                            generate_mesh_decomposition(
                                mesh_path, lod_0_box_size, grid_origin,
                                current_start_fragment,
                                current_end_fragment, current_lod,
                                num_chunks))

            results = []

            # Remove empty slabs
            decomposition_results = [
                fragments for fragments in decomposition_results if fragments
            ]

            fragments = [
                fragment for fragments in decomposition_results
                for fragment in fragments
            ]

            del decomposition_results

            mesh_util.write_mesh_files(
                f"{output_path}/multires", f"{id}", grid_origin, fragments,
                current_lod, lods[:idx + 1],
                np.asarray([lod_0_box_size, lod_0_box_size, lod_0_box_size]))

            del fragments


''' END Functions'''


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
    Creates indiviudal mesh files in .obj format for each entity in the given segmentation volume
    """
    if not path.exists(out_path):
        os.makedirs(out_path)

    os.makedirs(f"{out_path}/", exist_ok=True)
    temp_mesh_dir = Path(f"{out_path}/")

    # Accessing the data as store:
    root = parse_url(mask_path, mode="w")
    store = root.store
    data = zarr.open(store).get('0')

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
            chunk_indices = tuple(map(int, key.split('.')))
            # Calculate the start and end coordinates for each dimension
            start_coords = [index * size for index, size in zip(chunk_indices, chunk_shape)]
            end_coords = [start + size for start, size in zip(start_coords, chunk_shape)]
            # Append to lists
            all_start_coords.append(start_coords)
            all_end_coords.append(end_coords)

        # Find the minimum start and maximum end coordinates across all chunks
        combined_start = [min(coords) for coords in zip(*all_start_coords)]
        combined_end = [max(coords) for coords in zip(*all_end_coords)]
        combined_slices = tuple(slice(start, end) for start, end in zip(combined_start, combined_end))
        combined_data = data.get_basic_selection(combined_slices)[0, 0, :, :, :]
        combined_data = np.pad(combined_data, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        mask = np.where(combined_data == _key, 1, 0)

        # try:
        v2, f2, n2, values2 = measure.marching_cubes(volume=mask)
        # f2 = f2 + 1
        trimesh_mesh = trimesh.Trimesh(vertices=v2, faces=f2)

        # Apply smoothing
        print(f"Smoothing mesh {_key} ({len(trimesh_mesh.vertices)} vertices)...")
        trimesh_mesh = trimesh_mesh.smoothed(lamb=0.75, iterations=smoothing)

        # Fix any mesh issues
        if not trimesh_mesh.is_watertight:
            print(f"Fixing non-watertight mesh {_key}...")
            trimesh_mesh.fill_holes()

        trimesh_mesh.fix_normals()

        generate_decimated_meshes(temp_mesh_dir, list(range(3)), [{"vertices": v2, "faces": f2, "id": _key}], 4, 10)
        generate_neuroglancer_multires_mesh(temp_mesh_dir, 1, list(range(3)), ".ply", 8)

        multires_output_path = f"{temp_mesh_dir}/multires"
        mesh_util.write_segment_properties_file(multires_output_path)
        mesh_util.write_info_file(multires_output_path)
        # except:
        #     print("Error with", _key)

        if (test):
            print("Ran as Test")
            break
    print("\r", "done", end="")


if __name__ == '__main__':
    get_meshes_ng("http://127.0.0.1:8080", "/Users/ericmoerth/ws/tissue-map-tools/out",
              "/Users/ericmoerth/ws/tissue-map-tools/out/out.csv", "gloms", 10, True)

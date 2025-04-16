from ome_zarr.io import parse_url
import numpy as np
import pandas as pd
import math
import zarr
from itertools import product
from skimage import measure
import open3d as o3d
import scipy
import os.path as path
import os
import requests
import ome_types


def stats(arr):
    """
    Internal function to calculate statistics from a given 3D Array

    Parameters
    ----------
    arr : the 3D numpy array

    Returns
    -------
    map[float]: Collection of Statistics for the given Array

    Examples
    --------
    >>> stats(arr)
    """
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


def sub_volume_analysis(mask_path, raw_path, csv_out, mask_generation_res='0'):
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

    root_data = parse_url(raw_path + "/0", mode="w")
    store_data = root_data.store
    data_raw = zarr.open(store_data).get(
        mask_generation_res)  # depends on the resolution at which the masks were generated

    response = requests.get(raw_path + "/OME/METADATA.ome.xml")
    ome_xml = ome_types.from_xml(response.text.replace("Â", ""))
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
    print("Initialising the Mesh Creation")
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

import math
import urllib.parse
from itertools import product
from os import path as path

import numpy as np
import ome_types
import pandas as pd
import requests
import scipy
import zarr
from ome_zarr.io import parse_url


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
    return map(
        float,
        (
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
        ),
    )


def sub_volume_analysis(
    mask_path, raw_path, ome_xml_path, csv_out, mask_generation_res="0"
):
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
    data = zarr.open(store).get("0")

    root_data = parse_url(raw_path, mode="w")
    store_data = root_data.store
    data_raw = zarr.open(store_data).get(
        mask_generation_res
    )  # depends on the resolution at which the masks were generated

    parsed = urllib.parse.urlparse(ome_xml_path)
    if parsed.scheme == "":
        # local file
        with open(ome_xml_path, "r") as file:
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
        df["id"] = df["id"].astype(int)  # Ensure column A is integer
        worked_ids = list(df["id"])
        dataOut = df.to_numpy(dtype="object").tolist()

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
    print("Initialising the Statistics Calculation")
    mask_to_brick = _calc_mask_to_brick(data)
    chunk_shape = data.chunks

    i = 0
    for _key in mask_to_brick:
        i = i + 1
        if _key not in worked_ids:
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

            # Create the slices for combined selection
            combined_slices = tuple(
                slice(start, end) for start, end in zip(combined_start, combined_end)
            )
            # Retrieve the data for the combined region
            combined_data = data.get_basic_selection(combined_slices)[0, 0, :, :, :]
            mask = np.where(combined_data == _key, 1, 0)

            # For all Channels get the original data:
            combined_end[1] = len(channel_names)
            combined_slices = tuple(
                slice(start, end) for start, end in zip(combined_start, combined_end)
            )
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

import numpy as np
from functools import cmp_to_key
import struct
import os
import json
import glob
from collections import namedtuple
import trimesh


class Fragment:
    """Fragment class used to store and update fragment chunk vertices, faces
    and corresponding lod 0 fragments
    """
    def __init__(self, vertices, faces, lod_0_fragment_pos):
        self.vertices = vertices
        self.faces = faces
        self.lod_0_fragment_pos = lod_0_fragment_pos

    def update_faces(self, new_faces):
        self.faces = np.vstack((self.faces, new_faces + len(self.vertices)))

    def update_vertices(self, new_vertices):
        self.vertices = np.vstack((self.vertices, new_vertices))

    def update_lod_0_fragment_pos(self, new_lod_0_fragment_pos):
        self.lod_0_fragment_pos.append(new_lod_0_fragment_pos)

    def update(self, new_vertices, new_faces, new_lod_0_fragment_pos):
        self.update_faces(new_faces)
        self.update_vertices(new_vertices)
        self.update_lod_0_fragment_pos(new_lod_0_fragment_pos)


CompressedFragment = namedtuple(
    'CompressedFragment',
    ['draco_bytes', 'position', 'offset', 'lod_0_positions'])


def unpack_and_remove(datatype, num_elements, file_content):
    """Read and remove bytes from binary file object

    Args:
        datatype: Type of data
        num_elements: Number of datatype elements to read
        file_content: Binary file object

    Returns:
        output: The data that was unpacked
        file_content: The file contents with the unpacked data removed
    """

    datatype = datatype * num_elements
    output = struct.unpack(datatype, file_content[0:4 * num_elements])
    file_content = file_content[4 * num_elements:]
    if num_elements == 1:
        return output[0], file_content
    else:
        return np.array(output), file_content


def mesh_loader(filepath):
    """Wrapper for trimesh mesh loading, with addition of ngmesh loading.

    Args:
        filepath ('str'): Path to mesh

    Returns:
        vertices: Vertices
        faces: Faces
    """
    def _load_ngmesh(filepath):
        """Load ngmesh formatted mesh files"""
        with open(filepath, mode='rb') as file:
            file_content = file.read()

        num_vertices, file_content = unpack_and_remove("I", 1, file_content)
        vertices, file_content = unpack_and_remove("f", 3 * num_vertices,
                                                   file_content)
        num_faces = int(len(file_content) / 12)
        faces, file_content = unpack_and_remove("I", 3 * num_faces,
                                                file_content)

        vertices = vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)

        return vertices, faces
    
    vertices = None
    faces = None
    
    _, ext = os.path.splitext(filepath)
    if ext == "" or ext == ".ngmesh" or ext == ".ng":
        vertices, faces = _load_ngmesh(filepath)
    else:
        mesh = trimesh.load(filepath)
        if hasattr(mesh, "vertices"):
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
        del mesh

    return vertices, faces


def _cmp_zorder(lhs, rhs) -> bool:
    """Used to check if two values are in correct z-curve order.
    Based on https://github.com/google/neuroglancer/issues/272#issuecomment-752212014

    Args:
        lhs: Left hand side to compare
        rhs: Right hand side to compare

    Returns:
        bool: True if in correct z order
    """
    def less_msb(x: int, y: int) -> bool:
        return x < y and x < (x ^ y)

    # Assume lhs and rhs array-like objects of indices.
    assert len(lhs) == len(rhs)
    # Will contain the most significant dimension.
    msd = 2
    # Loop over the other dimensions.
    for dim in [1, 0]:
        # Check if the current dimension is more significant
        # by comparing the most significant bits.
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] - rhs[msd]


def zorder_fragments(fragments):
    """Order the fragments in appropriate z curve order

    Args:
        fragments: Fragments

    Returns:
        fragments: Z-curve sorted fragments
    """

    fragments, _ = zip(
        *sorted(zip(fragments, [fragment.position for fragment in fragments]),
                key=cmp_to_key(lambda x, y: _cmp_zorder(x[1], y[1]))))
    return list(fragments)


def rewrite_index_with_empty_fragments(path, current_lod_fragments):
    """Based on existing fragments and newly created fragments
    (`current_lod_fragments`), rewrite index file with missing empty fragments.
    This is necessary because if an empty fragment is omitted, then it will
    not replace an existing fragment.

    Args:
        path ('str'): Path to index file
        current_lod_fragments: The current fragments ot add to the index file
    """

    # index file contains info from all previous lods
    with open(f"{path}.index", mode='rb') as file:
        file_content = file.read()

    chunk_shape, file_content = unpack_and_remove("f", 3, file_content)
    grid_origin, file_content = unpack_and_remove("f", 3, file_content)
    num_lods, file_content = unpack_and_remove("I", 1, file_content)
    lod_scales, file_content = unpack_and_remove("f", num_lods, file_content)
    vertex_offsets, file_content = unpack_and_remove("f", num_lods * 3,
                                                     file_content)

    num_fragments_per_lod, file_content = unpack_and_remove(
        "I", num_lods, file_content)
    if type(num_fragments_per_lod) == int:
        num_fragments_per_lod = np.array([num_fragments_per_lod])

    all_current_fragment_positions = []
    all_current_fragment_offsets = []

    for lod in range(num_lods):
        fragment_positions, file_content = unpack_and_remove(
            "I", num_fragments_per_lod[lod] * 3, file_content)
        fragment_positions = fragment_positions.reshape((3, -1)).T
        fragment_offsets, file_content = unpack_and_remove(
            "I", num_fragments_per_lod[lod], file_content)
        if type(fragment_offsets) == int:
            fragment_offsets = np.array([fragment_offsets])
        all_current_fragment_positions.append(fragment_positions.astype(int))
        all_current_fragment_offsets.append(fragment_offsets.tolist())

    # now we are going to add the new lod info and update lower lods
    current_lod = num_lods
    num_lods += 1
    all_current_fragment_positions.append(
        np.asarray([fragment.position
                    for fragment in current_lod_fragments]).astype(int))
    all_current_fragment_offsets.append(
        [fragment.offset for fragment in current_lod_fragments])

    # first process based on newly added fragments
    all_missing_fragment_positions = []
    for lod in range(num_lods):
        all_required_fragment_positions = set()

        if lod == current_lod:  # then we are processing newest lod
            # add those that are required based on lower lods
            for lower_lod in range(lod):
                all_required_fragment_positions_np = np.unique(
                    all_current_fragment_positions[lower_lod] //
                    2**(lod - lower_lod),
                    axis=0).astype(int)
                all_required_fragment_positions.update(
                    set(map(tuple, all_required_fragment_positions_np)))
        else:
            # update lower lods based on current lod
            for fragment in current_lod_fragments:
                # normally we would just do the following with -0 and +1, but because of quantization that occurs(?), this makes things extra conservative so we don't miss things
                # ensures that it is positive, otherwise wound up with -1 to uint, causing errors
                new_required_fragment_positions = fragment.lod_0_positions // 2**lod
                all_required_fragment_positions.update(
                    set(map(tuple, new_required_fragment_positions)))
        current_missing_fragment_positions = all_required_fragment_positions - \
            set(map(tuple, all_current_fragment_positions[lod]))
        all_missing_fragment_positions.append(
            current_missing_fragment_positions)

    num_fragments_per_lod = []
    all_fragment_positions = []
    all_fragment_offsets = []
    for lod in range(num_lods):
        if len(all_missing_fragment_positions[lod]) > 0:
            lod_fragment_positions = list(
                all_missing_fragment_positions[lod]) + list(
                    all_current_fragment_positions[lod])
            lod_fragment_offsets = list(
                np.zeros(len(all_missing_fragment_positions[lod]))
            ) + all_current_fragment_offsets[lod]
        else:
            lod_fragment_positions = all_current_fragment_positions[lod]
            lod_fragment_offsets = all_current_fragment_offsets[lod]

        lod_fragment_offsets, lod_fragment_positions = zip(
            *sorted(zip(lod_fragment_offsets, lod_fragment_positions),
                    key=cmp_to_key(lambda x, y: _cmp_zorder(x[1], y[1]))))
        all_fragment_positions.append(lod_fragment_positions)
        all_fragment_offsets.append(lod_fragment_offsets)
        num_fragments_per_lod.append(len(all_fragment_offsets[lod]))

    num_fragments_per_lod = np.array(num_fragments_per_lod)
    lod_scales = np.array([2**i for i in range(num_lods)])
    vertex_offsets = np.array([[0., 0., 0.] for _ in range(num_lods)])
    with open(f"{path}.index_with_empty_fragments", 'ab') as f:
        f.write(chunk_shape.astype('<f').tobytes())
        f.write(grid_origin.astype('<f').tobytes())

        f.write(struct.pack('<I', num_lods))
        f.write(lod_scales.astype('<f').tobytes())
        f.write(vertex_offsets.astype('<f').tobytes(order='C'))

        f.write(num_fragments_per_lod.astype('<I').tobytes())

        for lod in range(num_lods):
            fragment_positions = np.array(all_fragment_positions[lod]).reshape(
                -1, 3)
            fragment_offsets = np.array(all_fragment_offsets[lod]).reshape(-1)

            f.write(fragment_positions.T.astype('<I').tobytes(order='C'))
            f.write(fragment_offsets.astype('<I').tobytes(order='C'))

    os.system(f"mv {path}.index_with_empty_fragments {path}.index")
    return


def write_info_file(path):
    """Write info file for meshes

    Args:
        path ('str'): Path to meshes
    """
    # default to 10 quantization bits
    with open(f'{path}/info', 'w') as f:
        info = {
            '@type': 'neuroglancer_multilod_draco',
            'vertex_quantization_bits': 10,
            'transform': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            'lod_scale_multiplier': 1,
            'segment_properties': "segment_properties"
        }

        json.dump(info, f)


def write_segment_properties_file(path):
    """Create segment properties dir/file so that all meshes are selectable based
    on id in neuroglancer.

    Args:
        path ('str'): Path to meshes
    """

    segment_properties_directory = f"{path}/segment_properties"
    if not os.path.exists(segment_properties_directory):
        os.makedirs(segment_properties_directory)

    with open(f"{segment_properties_directory}/info", 'w') as f:
        ids = [
            index_file.split("/")[-1].split(".")[0]
            for index_file in glob.glob(f'{path}/*.index')
        ]
        ids.sort(key=int)
        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids":
                ids,
                "properties": [{
                    "id": "label",
                    "type": "label",
                    "values": [""] * len(ids)
                }]
            }
        }
        json.dump(info, f)


def write_index_file(path, grid_origin, fragments, current_lod, lods,
                     chunk_shape):
    """Write the index files for a mesh.

    Args:
        path: Path to mesh
        grid_origin: The lod 0 mesh grid origin
        fragments: Fragments for current lod
        current_lod: The current lod
        lods: A list of all the lods
        chunk_shape: Chunk shape.
    """

    # since we don't know if the lowest res ones will have meshes for all svs
    lods = [lod for lod in lods if lod <= current_lod]

    num_lods = len(lods)
    lod_scales = np.array([2**i for i in range(num_lods)])
    vertex_offsets = np.array([[0., 0., 0.] for _ in range(num_lods)])
    num_fragments_per_lod = np.array([len(fragments)])
    if current_lod == lods[0] or not os.path.exists(f"{path}.index"):  
        # then is highest res lod or if the file doesnt exist yet it failed
        # to write out the index file because s0 was draco compressed to nothing 
        # in encode_faces_to_custom_drc_bytes due to voxel size and chunk shape

        with open(f"{path}.index", 'wb') as f:
            f.write(chunk_shape.astype('<f').tobytes())
            f.write(grid_origin.astype('<f').tobytes())
            f.write(struct.pack('<I', num_lods))
            f.write(lod_scales.astype('<f').tobytes())
            f.write(vertex_offsets.astype('<f').tobytes(order='C'))
            f.write(num_fragments_per_lod.astype('<I').tobytes())
            f.write(
                np.asarray([fragment.position for fragment in fragments
                            ]).T.astype('<I').tobytes(order='C'))
            f.write(
                np.asarray([fragment.offset for fragment in fragments
                            ]).astype('<I').tobytes(order='C'))

    else:
        rewrite_index_with_empty_fragments(path, fragments)


def write_mesh_file(path, fragments):
    """Write out the actual draco formatted mesh

    Args:
        path: Path to mesh directory
        fragments: Fragments

    Returns:
        fragments: Fragments with their draco mesh deleted
    """
    with open(path, 'ab') as f:
        for idx, fragment in enumerate(fragments):
            f.write(fragment.draco_bytes)
            fragments[idx] = CompressedFragment(None, fragment.position,
                                                fragment.offset,
                                                fragment.lod_0_positions)

    return fragments


def write_mesh_files(mesh_directory, object_id, grid_origin, fragments,
                     current_lod, lods, chunk_shape):
    """Write out all relevant mesh files.

    Args:
        mesh_directory: Path to mesh directory
        object_id: Mesh id
        grid_origin: The lod 0 mesh grid origin
        fragments: Current lod fragments
        current_lod: The current lod
        lods: List of all lods
        chunk_shape: Shape of chunk
    """

    path = mesh_directory + "/" + object_id
    if len(fragments) > 0:
        # If len(fragments) == 0, that means that the mesh has been reduced to nothing after draco compression
        fragments = zorder_fragments(fragments)
        fragments = write_mesh_file(path, fragments)
        write_index_file(path, grid_origin, fragments, current_lod, lods,
                         chunk_shape)

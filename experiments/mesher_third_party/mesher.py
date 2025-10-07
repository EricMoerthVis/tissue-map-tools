import numpy as np
import trimesh
import pyfqmr
import os
from trimesh.intersections import slice_faces_plane
import subprocess

from experiments.mesher_third_party import mesher_util

""" Functions taken over from "multiresolution-mesh-creator"""
# originally from https://github.com/davidackerman/multiresolution-mesh-creator/tree/master
# please see the license in LICENSE_multiresolution-mesh-creator.md


def pyfqmr_decimate(
    vertices, faces, output_path, id, lod, decimation_factor, aggressiveness
):
    """Mesh decimation using pyfqmr."""
    desired_faces = max(len(faces) // (decimation_factor**lod), 4)
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(vertices, faces)
    del vertices
    del faces
    mesh_simplifier.simplify_mesh(
        target_count=desired_faces,
        aggressiveness=aggressiveness,
        preserve_border=False,
        verbose=False,
    )
    vertices, faces, _ = mesh_simplifier.getMesh()
    del mesh_simplifier

    mesh = trimesh.Trimesh(vertices, faces)
    del vertices
    del faces
    _ = mesh.export(f"{output_path}/s{lod}/{id}.ply")


def generate_decimated_meshes(
    output_path, lods, infos, decimation_factor, aggressiveness
):
    """Generate decimated meshes for all ids in `ids`, over all lod in `lods`."""

    for current_lod in lods:
        os.makedirs(f"{output_path}/mesh_lods/s{current_lod}", exist_ok=True)
        for info in infos:
            if current_lod == 0:
                # Write out the original ply to file
                mesh = trimesh.Trimesh(info.get("vertices"), info.get("faces"))
                _ = mesh.export(
                    f"{output_path}/mesh_lods/s{current_lod}/{info.get('id')}.ply"
                )
            else:
                # Getting the vertices and faces for the mesh
                pyfqmr_decimate(
                    info.get("vertices"),
                    info.get("faces"),
                    f"{output_path}/mesh_lods",
                    info.get("id"),
                    current_lod,
                    decimation_factor,
                    aggressiveness,
                )


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
            vertices, faces, uv = slice_faces_plane(
                vertices, faces, plane_normal, plane_origin
            )
        except ValueError as e:
            if str(e) != "input must be 1D integers!":
                raise
            else:
                pass

    return vertices, faces


def update_fragment_dict(dictionary, fragment_pos, vertices, faces, lod_0_fragment_pos):
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
        dictionary[fragment_pos] = mesher_util.Fragment(
            vertices, faces, [lod_0_fragment_pos]
        )


def _encode_mesh_draco(
    vertices: np.ndarray,
    faces: np.ndarray,
    bounds_min: np.ndarray,
    box_size: float,
    position_quantization_bits,
    temp_dir,
) -> bytes:
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
    vertices *= 2**position_quantization_bits - 1
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
        "-i",
        obj_path,
        "-o",
        drc_path,
        "-qp",
        str(position_quantization_bits),
        "-qt",
        "8",  # Higher quality tangents
        "-qn",
        "8",  # Higher quality normals
        "-qtx",
        "8",  # Higher quality texture coordinates
        "-cl",
        "10",  # Maximum compression level
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(drc_path, "rb") as f:
            return f.read()

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Draco encoding failed: {e.stdout} {e.stderr}")


def generate_mesh_decomposition(
    mesh_path,
    lod_0_box_size,
    grid_origin,
    start_fragment,
    end_fragment,
    current_lod,
    num_chunks,
):
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
    vertices, faces = mesher_util.mesh_loader(mesh_path)

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
            vertices, faces = my_slice_faces_plane(vertices, faces, -n_d, plane_origin)
            if len(vertices) == 0:
                return None
            plane_origin = n_d * start_fragment[dimension] * sub_box_size
            vertices, faces = my_slice_faces_plane(vertices, faces, n_d, plane_origin)

    if len(vertices) == 0:
        return None

    # Get chunks of desired size by slicing in x,y,z and ensure their chunks
    # are divisible by 2x2x2 chunks
    for x in range(start_fragment[0], end_fragment[0]):
        plane_origin_yz = nyz * (x + 1) * sub_box_size
        vertices_yz, faces_yz = my_slice_faces_plane(
            vertices, faces, -nyz, plane_origin_yz
        )

        for y in range(start_fragment[1], end_fragment[1]):
            plane_origin_xz = nxz * (y + 1) * sub_box_size
            vertices_xz, faces_xz = my_slice_faces_plane(
                vertices_yz, faces_yz, -nxz, plane_origin_xz
            )

            for z in range(start_fragment[2], end_fragment[2]):
                plane_origin_xy = nxy * (z + 1) * sub_box_size
                vertices_xy, faces_xy = my_slice_faces_plane(
                    vertices_xz, faces_xz, -nxy, plane_origin_xy
                )

                lod_0_fragment_position = tuple(np.array([x, y, z]))
                if current_lod != 0:
                    fragment_position = tuple(np.array([x, y, z]) // 2)
                else:
                    fragment_position = lod_0_fragment_position

                update_fragment_dict(
                    combined_fragments_dictionary,
                    fragment_position,
                    vertices_xy,
                    faces_xy,
                    list(lod_0_fragment_position),
                )

                vertices_xz, faces_xz = my_slice_faces_plane(
                    vertices_xz, faces_xz, nxy, plane_origin_xy
                )

            vertices_yz, faces_yz = my_slice_faces_plane(
                vertices_yz, faces_yz, nxz, plane_origin_xz
            )

        vertices, faces = my_slice_faces_plane(vertices, faces, nyz, plane_origin_yz)

    # Return combined_fragments_dictionary
    for fragment_pos, fragment in combined_fragments_dictionary.items():
        if fragment.vertices.size > 0:
            current_box_size = lod_0_box_size * 2**current_lod
            # draco_bytes = encode_faces_to_custom_drc_bytes(
            #     fragment.vertices,
            #     np.zeros(np.shape(fragment.vertices)),
            #     fragment.faces,
            #     np.asarray(3 * [current_box_size]),
            #     np.asarray(fragment_pos) * current_box_size,
            #     position_quantization_bits=10)

            draco_bytes = _encode_mesh_draco(
                fragment.vertices,
                fragment.faces,
                np.asarray(3 * [current_box_size]),
                np.asarray(fragment_pos) * current_box_size,
                position_quantization_bits=10,
                temp_dir=os.path.dirname(mesh_path),
            )

            if len(draco_bytes) > 12:
                # Then the mesh is not empty
                fragment = mesher_util.CompressedFragment(
                    draco_bytes,
                    np.asarray(fragment_pos),
                    len(draco_bytes),
                    np.asarray(fragment.lod_0_fragment_pos),
                )
                fragments.append(fragment)

    return fragments


def generate_neuroglancer_multires_mesh(
    output_path, id, lods, original_ext, lod_0_box_size
):
    """function to generate multiresolution mesh in neuroglancer
    mesh format using prewritten meshes at different levels of detail.
    """
    os.makedirs(f"{output_path}/multires", exist_ok=True)

    for idx, current_lod in enumerate(lods):
        if current_lod == 0:
            mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}{original_ext}"
        else:
            mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}.ply"

        vertices, _ = mesher_util.mesh_loader(mesh_path)

        if vertices is None:
            return

        if current_lod == 0:
            max_box_size = lod_0_box_size * 2 ** lods[-1]
            # TODO: check
            grid_origin = (vertices.min(axis=0) // max_box_size - 1) * max_box_size
        vertices -= grid_origin

        current_box_size = lod_0_box_size * 2**current_lod
        start_fragment = np.maximum(
            vertices.min(axis=0) // current_box_size - 1, np.array([0, 0, 0])
        ).astype(int)
        end_fragment = (vertices.max(axis=0) // current_box_size + 1).astype(int)

        del vertices

        # Want to divide the mesh up into up to num_workers chunks. We do
        # that by first subdividing the largest dimension as much as
        # possible, followed by the next largest dimension etc so long
        # as we don't exceed num_workers slices. If we instead slice each
        # dimension once, before slicing any dimension twice etc, it would
        # increase the number of mesh slice operations we perform, which
        # seems slow.

        max_number_of_chunks = end_fragment - start_fragment
        dimensions_sorted = np.argsort(-max_number_of_chunks)
        # TODO: check. This seems wrong
        num_chunks = np.array([1, 1, 1])

        for d in dimensions_sorted:
            if num_chunks[d] < max_number_of_chunks[d]:
                num_chunks[d] += 1
        # wrong until here

        stride = np.ceil((end_fragment - start_fragment) / num_chunks).astype(np.int32)

        # Scattering here, unless broadcast=True, causes this issue:
        # https://github.com/dask/distributed/issues/4612. But that is
        # slow so we are currently electing to read the meshes each time
        # within generate_mesh_decomposition.
        # vertices_to_send = client.scatter(vertices, broadcast=True)
        # faces_to_send = client.scatter(faces, broadcast=True)

        decomposition_results = []
        for x in range(start_fragment[0], end_fragment[0], stride[0]):
            for y in range(start_fragment[1], end_fragment[1], stride[1]):
                for z in range(start_fragment[2], end_fragment[2], stride[2]):
                    current_start_fragment = np.array([x, y, z])
                    current_end_fragment = current_start_fragment + stride
                    # then we aren't parallelizing again
                    decomposition_results.append(
                        generate_mesh_decomposition(
                            mesh_path=mesh_path,
                            lod_0_box_size=lod_0_box_size,
                            grid_origin=grid_origin,
                            start_fragment=current_start_fragment,
                            end_fragment=current_end_fragment,
                            current_lod=current_lod,
                            num_chunks=num_chunks,
                        )
                    )

        # Remove empty slabs
        decomposition_results = [
            fragments for fragments in decomposition_results if fragments
        ]

        fragments = [
            fragment for fragments in decomposition_results for fragment in fragments
        ]

        del decomposition_results

        mesher_util.write_mesh_files(
            mesh_directory=f"{output_path}/multires",
            object_id=f"{id}",
            grid_origin=grid_origin,
            fragments=fragments,
            current_lod=current_lod,
            lods=lods[: idx + 1],  # TODO: check
            chunk_shape=np.asarray([lod_0_box_size, lod_0_box_size, lod_0_box_size]),
        )

        del fragments


""" END Functions"""

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
import json
from pathlib import Path
import trimesh
import pyfqmr
import subprocess
import tempfile
import os
import struct
from typing import List, Dict, Optional, NamedTuple
import shutil
from zmesh import Mesher


class Fragment(NamedTuple):
    """Represents a mesh fragment with its position and encoded data."""
    position: np.ndarray  # 3D position of fragment in grid
    draco_bytes: bytes  # Draco-encoded mesh data
    size: int  # Size of encoded data in bytes
    lod: int  # Level of detail for this fragment


class NeuroglancerMeshWriter:
    def __init__(self, output_dir: str, box_size: int = 64,
                 vertex_quantization_bits: int = 10,
                 transform: Optional[List[float]] = None,
                 clean_output: bool = False,
                 data_type: str = "uint64"):
        """Initialize the mesh writer with output directory and parameters.

        Args:
            output_dir: Base output directory
            box_size: Size of the smallest (LOD 0) chunks
            vertex_quantization_bits: Number of bits for vertex quantization (10 or 16)
            transform: Optional 4x3 homogeneous transform matrix (12 values)
            clean_output: If True, remove existing directory before starting
        """
        if vertex_quantization_bits not in (10, 16):
            raise ValueError("vertex_quantization_bits must be 10 or 16")

        self.output_dir = Path(output_dir)
        self.box_size = box_size
        self.vertex_quantization_bits = vertex_quantization_bits
        self.transform = transform or [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        self.lod_scale_multiplier = 2.0
        self.data_type = data_type

        # Clean output directory if requested
        if clean_output and self.output_dir.exists():
            print(f"Cleaning existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)

        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_info_file(self):
        """Write the Neuroglancer info JSON file according to the specification at:
        https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#multi-resolution-mesh-info-json-file-format
        """
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": self.vertex_quantization_bits,
            "transform": self.transform,
            "lod_scale_multiplier": self.lod_scale_multiplier
        }

        with open(self.output_dir / "info", "w") as f:
            json.dump(info, f)

        print(f"Created info file: {self.output_dir / 'info'}")

    def write_binary_manifest(self, mesh_id: int, fragments_by_lod: Dict[int, List[Fragment]],
                              grid_origin: np.ndarray, num_lods: int):
        """Write the binary manifest file following the Neuroglancer precomputed mesh format.

        The manifest file format is described at:
        https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#multi-resolution-mesh-manifest-file-format
        """
        """Write the binary manifest file with debug logging."""
        # Ensure mesh ID is written as a base-10 string representation, as required by the spec
        mesh_id_str = str(mesh_id)
        manifest_path = self.output_dir / f"{mesh_id_str}.index"
        print(f"\nWriting manifest for mesh {mesh_id} to {manifest_path}")
        print(f"Number of LODs: {num_lods}")
        print(f"Grid origin: {grid_origin}")

        fragments_per_lod = [len(fragments_by_lod.get(lod, []))
                             for lod in range(num_lods)]
        print(f"Fragments per LOD: {fragments_per_lod}")

        try:
            with open(manifest_path, "wb") as f:
                # Write chunk shape (3x float32le)
                chunk_shape = np.array([self.box_size] * 3, dtype=np.float32)
                f.write(chunk_shape.tobytes())
                print(f"Wrote chunk shape: {chunk_shape}")

                # Write grid origin (3x float32le)
                grid_origin = np.array(grid_origin, dtype=np.float32)
                f.write(grid_origin.tobytes())
                print(f"Wrote grid origin: {grid_origin}")

                # Write num_lods (uint32le)
                f.write(struct.pack("<I", num_lods))
                print(f"Wrote num_lods: {num_lods}")

                # Write lod_scales (num_lods x float32le)
                lod_scales = np.array([self.box_size * (2 ** i) for i in range(num_lods)],
                                      dtype=np.float32)
                f.write(lod_scales.tobytes())
                print(f"Wrote lod_scales: {lod_scales}")

                # Write vertex_offsets ([num_lods, 3] array of float32le)
                vertex_offsets = np.zeros((num_lods, 3), dtype=np.float32)
                f.write(vertex_offsets.tobytes())
                print(f"Wrote vertex_offsets")

                # Write num_fragments_per_lod (num_lods x uint32le)
                fragments_per_lod_arr = np.array(fragments_per_lod, dtype=np.uint32)
                f.write(fragments_per_lod_arr.tobytes())
                print(f"Wrote fragments_per_lod: {fragments_per_lod_arr}")

                # Write fragment data for each LOD
                for lod in range(num_lods):
                    fragments = fragments_by_lod.get(lod, [])
                    if not fragments:
                        print(f"No fragments for LOD {lod}")
                        continue

                    print(f"Writing {len(fragments)} fragments for LOD {lod}")

                    # Sort fragments by Z-order
                    fragments = sorted(fragments,
                                       key=lambda f: self._compute_z_order(f.position))

                    # Write fragment positions ([num_fragments, 3] array of uint32le)
                    positions = np.array([f.position for f in fragments], dtype=np.uint32)
                    f.write(positions.tobytes())
                    print(f"Wrote fragment positions for LOD {lod}")

                    # Write fragment sizes (num_fragments x uint32le)
                    sizes = np.array([f.size for f in fragments], dtype=np.uint32)
                    f.write(sizes.tobytes())
                    print(f"Wrote fragment sizes for LOD {lod}")

            print(f"Successfully wrote manifest file: {manifest_path}")
            # Verify file exists and has content
            print(f"File exists: {manifest_path.exists()}")
            print(f"File size: {manifest_path.stat().st_size} bytes")

        except Exception as e:
            print(f"Error writing manifest for mesh {mesh_id}: {str(e)}")
            raise

    def write_fragment_data(self, mesh_id: int, fragments_by_lod: Dict[int, List[Fragment]]):
        """Write the fragment data file following the Neuroglancer precomputed mesh format.

        The fragment data file format is described at:
        https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#multi-resolution-mesh-fragment-data-file-format
        """
        """Write the fragment data file with debug logging."""
        # Ensure mesh ID is written as a base-10 string representation, as required by the spec
        mesh_id_str = str(mesh_id)
        data_path = self.output_dir / mesh_id_str
        print(f"\nWriting fragment data for mesh {mesh_id} to {data_path}")

        try:
            with open(data_path, "wb") as f:
                total_fragments = 0
                total_bytes = 0

                for lod in sorted(fragments_by_lod.keys()):
                    lod_fragments = fragments_by_lod[lod]
                    print(f"LOD {lod}: Writing {len(lod_fragments)} fragments")

                    for fragment in sorted(lod_fragments,
                                           key=lambda f: self._compute_z_order(f.position)):
                        f.write(fragment.draco_bytes)
                        total_fragments += 1
                        total_bytes += len(fragment.draco_bytes)

                print(f"Successfully wrote {total_fragments} fragments")
                print(f"Total bytes written: {total_bytes}")

            # Verify file exists and has content
            print(f"File exists: {data_path.exists()}")
            print(f"File size: {data_path.stat().st_size} bytes")

        except Exception as e:
            print(f"Error writing fragment data for mesh {mesh_id}: {str(e)}")
            raise

    def process_mesh(self, mesh_id: int, mesh: trimesh.Trimesh, num_lods: int = 3):
        """Process a single mesh with additional debug logging."""
        print(f"\nProcessing mesh {mesh_id}")
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Ensure mesh is watertight before processing
        if not mesh.is_watertight:
            print(f"Making mesh watertight before processing")
            mesh.fill_holes()
            mesh.fix_normals()

        # Apply moderate smoothing to improve quality
        print(f"Applying pre-processing smoothing")
        mesh = mesh.smoothed(lamb=0.5, iterations=3)

        # Generate LODs
        lod_meshes = self.generate_lods(mesh, num_lods)
        print(f"Generated {len(lod_meshes)} LOD levels")

        # Calculate grid origin
        grid_origin = (mesh.vertices.min(axis=0) // self.box_size - 1) * self.box_size
        print(f"Grid origin: {grid_origin}")

        # Generate fragments for each LOD
        fragments_by_lod = {}
        for lod, lod_mesh in enumerate(lod_meshes):
            print(f"\nProcessing LOD {lod}")
            print(f"LOD mesh: {len(lod_mesh.vertices)} vertices, {len(lod_mesh.faces)} faces")

            # Apply coordinate offset - mesh vertices need to be in the same coordinate system as the voxel data
            # Offset should be 0 instead of the previous (-1,-1,-1) to align correctly with the image/labels
            lod_mesh.vertices = lod_mesh.vertices - grid_origin
            fragments = self.generate_fragments(lod_mesh, lod)

            if fragments:
                fragments_by_lod[lod] = fragments
                print(f"Generated {len(fragments)} fragments for LOD {lod}")
            else:
                print(f"No fragments generated for LOD {lod}")

        if not fragments_by_lod:
            print(f"Warning: No fragments generated for any LOD level")
            return

        try:
            # Write manifest and fragment data
            self.write_binary_manifest(mesh_id, fragments_by_lod, grid_origin, num_lods)
            self.write_fragment_data(mesh_id, fragments_by_lod)

            # Verify the files exist and are accessible
            mesh_id_str = str(mesh_id)
            index_path = self.output_dir / f"{mesh_id_str}.index"
            data_path = self.output_dir / mesh_id_str

            if index_path.exists() and data_path.exists():
                print(f"Successfully processed mesh {mesh_id}")
                print(f"  Index file: {index_path} ({index_path.stat().st_size} bytes)")
                print(f"  Data file: {data_path} ({data_path.stat().st_size} bytes)")
            else:
                if not index_path.exists():
                    print(f"ERROR: Index file {index_path} does not exist!")
                if not data_path.exists():
                    print(f"ERROR: Data file {data_path} does not exist!")
        except Exception as e:
            print(f"Error processing mesh {mesh_id}: {str(e)}")
            raise

    def _compute_z_order(self, pos: np.ndarray) -> int:
        """Compute Z-order curve index for a 3D position."""
        x, y, z = pos
        answer = 0
        for i in range(21):  # Support up to 21 bits per dimension
            answer |= ((x & (1 << i)) << (2 * i)) | \
                      ((y & (1 << i)) << (2 * i + 1)) | \
                      ((z & (1 << i)) << (2 * i + 2))
        return answer

    def generate_fragments(self, mesh: trimesh.Trimesh, lod: int,
                           enforce_grid_partition: bool = True) -> List[Fragment]:
        """Generate mesh fragments for a given LOD level."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return []

        print(f"LOD {lod}: Processing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        current_box_size = self.box_size * (2 ** lod)
        vertices = mesh.vertices

        # Calculate fragment bounds - ensure we include a margin around the actual mesh to avoid gaps
        start_fragment = np.maximum(
            vertices.min(axis=0) // current_box_size - 1,
            np.array([0, 0, 0])).astype(int)
        end_fragment = (vertices.max(axis=0) // current_box_size + 1).astype(int)

        # Add an overlap factor for better fragment connectivity
        overlap_factor = 0.2 * current_box_size  # 20% overlap between adjacent fragments (increased from 5%)

        fragments = []
        fragment_count = 0
        for x in range(start_fragment[0], end_fragment[0]):
            for y in range(start_fragment[1], end_fragment[1]):
                for z in range(start_fragment[2], end_fragment[2]):
                    pos = np.array([x, y, z])

                    # Extract vertices and faces for this fragment with overlap to reduce gaps
                    bounds_min = pos * current_box_size - overlap_factor
                    bounds_max = bounds_min + current_box_size + (2 * overlap_factor)

                    # Use expanded bounds for selecting vertices to ensure overlap between adjacent fragments
                    mask = np.all((vertices >= bounds_min) & (vertices < bounds_max), axis=1)
                    vertex_indices = np.where(mask)[0]

                    if len(vertex_indices) == 0:
                        continue

                    fragment_vertices = vertices[vertex_indices]

                    # Remap faces to use new vertex indices
                    vertex_map = {old: new for new, old in enumerate(vertex_indices)}
                    face_mask = np.all(np.isin(mesh.faces, vertex_indices), axis=1)
                    fragment_faces = mesh.faces[face_mask]

                    if len(fragment_faces) == 0:
                        continue

                    fragment_faces = np.array([[vertex_map[v] for v in face]
                                               for face in fragment_faces])

                    # Create fragment mesh
                    fragment_mesh = trimesh.Trimesh(
                        vertices=fragment_vertices,
                        faces=fragment_faces
                    )

                    # For LOD > 0, enforce 2x2x2 grid partitioning
                    if enforce_grid_partition and lod > 0:
                        fragment_mesh = self._enforce_grid_partition(
                            fragment_mesh,
                            bounds_min,
                            current_box_size
                        )

                    if len(fragment_mesh.vertices) == 0 or len(fragment_mesh.faces) == 0:
                        continue

                    try:
                        # Encode using Draco
                        draco_bytes = self._encode_mesh_draco(
                            fragment_mesh.vertices,
                            fragment_mesh.faces,
                            bounds_min,
                            current_box_size
                        )

                        if len(draco_bytes) > 12:
                            fragments.append(Fragment(
                                position=pos,
                                draco_bytes=draco_bytes,
                                size=len(draco_bytes),
                                lod=lod
                            ))
                            fragment_count += 1

                    except Exception as e:
                        print(f"Error processing fragment at {pos}: {str(e)}")
                        continue

        print(f"LOD {lod}: Generated {fragment_count} fragments")
        return fragments

    def _encode_mesh_draco(self, vertices: np.ndarray, faces: np.ndarray,
                           bounds_min: np.ndarray, box_size: float) -> bytes:
        """Encode a mesh using Google's Draco encoder with enhanced quality settings."""
        # Normalize vertices to quantization range
        vertices = vertices.copy()

        # Improved quantization to prevent vertex snapping at boundaries
        # Add a slight padding to avoid precision issues at the boundaries
        padding = box_size * 0.001  # 0.1% padding
        bounds_min = bounds_min - padding
        box_size = box_size + 2 * padding

        vertices -= bounds_min
        vertices /= box_size
        vertices *= (2 ** self.vertex_quantization_bits - 1)
        vertices = vertices.astype(np.int32)

        # Create temporary files for the mesh
        with tempfile.TemporaryDirectory() as temp_dir:
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
                "-qp", str(self.vertex_quantization_bits),
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

    def _enforce_grid_partition(self, mesh: trimesh.Trimesh,
                                bounds_min: np.ndarray,
                                box_size: float) -> trimesh.Trimesh:
        """Enforce grid partitioning for better fragment alignment and reduced gaps."""
        """Enforce 2x2x2 grid partitioning for LOD > 0 meshes."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return mesh

        try:
            # Calculate grid planes
            mid_points = bounds_min + np.array([
                [box_size / 2, 0, 0],
                [0, box_size / 2, 0],
                [0, 0, box_size / 2]
            ])

            normals = np.eye(3)

            # Create a slightly expanded version for splitting to avoid gaps
            # We'll use a small expansion factor to ensure overlapping fragments connect properly
            expansion_factor = 0.01  # 1% expansion for better watertight results

            # Split mesh along each plane with enhanced capping
            result_mesh = mesh
            for point, normal in zip(mid_points, normals):
                try:
                    # Use a slightly expanded mesh for slicing to ensure watertight results
                    new_mesh = result_mesh.slice_plane(point, normal, cap=True)
                    if new_mesh is not None and len(new_mesh.vertices) > 0:
                        # Apply post-processing to ensure the mesh is clean after slicing
                        new_mesh.fill_holes()
                        new_mesh.fix_normals()
                        result_mesh = new_mesh
                except ValueError:
                    continue

            if len(result_mesh.vertices) == 0:
                return mesh

            return result_mesh

        except Exception:
            return mesh

    def decimate_mesh(self, mesh: trimesh.Trimesh, target_ratio: float) -> trimesh.Trimesh:
        """Decimate a mesh to a target ratio of original faces."""
        # Skip tiny meshes as they can't be simplified well
        if len(mesh.faces) < 20:
            return mesh

        # Handle meshes with potential issues
        try:
            # Make a copy to avoid modifying the original
            mesh_copy = mesh.copy()

            # Try to fix any potential issues with the mesh before simplification
            if not mesh_copy.is_watertight:
                mesh_copy.fill_holes()

            # Update faces
            mesh_copy.update_faces(mesh_copy.unique_faces())
            mesh_copy.update_faces(mesh_copy.nondegenerate_faces())
            mesh_copy.fix_normals()

            # Set up simplifier
            simplifier = pyfqmr.Simplify()
            simplifier.setMesh(mesh_copy.vertices, mesh_copy.faces)

            # Calculate target face count, ensuring we keep at least 10 faces
            target_count = max(int(len(mesh_copy.faces) * target_ratio), 10)

            # Simplify with more conservative settings for better quality
            simplifier.simplify_mesh(target_count=target_count,
                                     aggressiveness=3,
                                     preserve_border=True,
                                     verbose=False)

            vertices, faces, _ = simplifier.getMesh()

            # Check if we got valid results
            if len(vertices) < 3 or len(faces) < 1:
                print(f"Warning: Simplification produced invalid mesh, using original")
                return mesh

            # Create a new trimesh with the decimated mesh
            result = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Post-process the simplified mesh
            result.update_faces(result.nondegenerate_faces())
            result.fix_normals()

            return result

        except Exception as e:
            print(f"Error during mesh decimation: {str(e)}")
            return mesh  # Return original mesh on error

    def generate_lods(self, mesh: trimesh.Trimesh, num_lods: int) -> List[trimesh.Trimesh]:
        """Generate levels of detail for a mesh."""
        lods = [mesh]  # LOD 0 is the highest detail
        current_mesh = mesh

        for i in range(1, num_lods):
            # Progressive decimation from previous LOD level for smoother transition
            target_ratio = 0.5  # Each level is about half the detail of previous level
            decimated = self.decimate_mesh(current_mesh, target_ratio)

            # Ensure each LOD has at least 10% fewer vertices than the previous
            if len(decimated.vertices) > 0.9 * len(current_mesh.vertices):
                # If decimation didn't reduce enough, force more reduction
                decimated = self.decimate_mesh(current_mesh, 0.3)

            # Add diagnostic info
            print(f"LOD {i}: Original {len(current_mesh.vertices)} vertices → {len(decimated.vertices)} vertices" +
                  f" ({len(decimated.vertices) / len(current_mesh.vertices):.2f}x)")

            lods.append(decimated)
            current_mesh = decimated  # Use this LOD as base for next level

        return lods


def create_ome_zarr_mesh(root_dir, name, mesh_writer):
    """Create an OME-NGFF Zarr store with Neuroglancer-compatible meshes"""
    """
    Create an OME-NGFF Zarr store following the 0.5 specification.

    Args:
        root_dir: Root directory for the Zarr store
        name: Name of the Zarr store 
        blob_data: 3D numpy array of the blob data
        labels_data: Optional segmentation labels
    """
    # Create path
    zarr_path = os.path.join(root_dir, f"{name}.zarr")
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    """ Mesh Writing """

    # Create a directory for mesh data within the zarr store
    mesh_dir = os.path.join(zarr_path, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # Create zarr.json for the mesh group according to RFC-8
    mesh_metadata = {
        "zarr_format": 3,
        "node_type": "external",
        "attributes": {
            "ome": {
                "version": "0.5",
                "mesh": {
                    "version": "0.1",
                    "type": "neuroglancer_multilod_draco",
                    "source": {
                        "image": "../",
                        "labels": "../labels/segmentation"
                    }
                }
            }
        }
    }

    # Write the mesh metadata file
    with open(os.path.join(mesh_dir, "zarr.json"), "w") as f:
        json.dump(mesh_metadata, f, indent=2)

    # If mesh_writer is provided, copy mesh data into the zarr store
    if mesh_writer is not None:
        mesh_source_dir = mesh_writer.output_dir
        print(f"\nCopying mesh files from {mesh_source_dir} to {mesh_dir}")

        # First check what files exist in the source directory
        source_files = os.listdir(mesh_source_dir)
        index_files = [f for f in source_files if f.endswith('.index')]
        data_files = [f for f in source_files if f.isdigit() and os.path.isfile(os.path.join(mesh_source_dir, f))]

        print(f"Source directory contains {len(index_files)} index files and {len(data_files)} data files")
        print(f"Index files: {index_files[:5]}... (showing first 5)" if len(
            index_files) > 5 else f"Index files: {index_files}")
        print(f"Data files: {data_files[:5]}... (showing first 5)" if len(
            data_files) > 5 else f"Data files: {data_files}")

        # Manually copy each index and data file to ensure they're correctly placed
        for item in source_files:
            src_path = os.path.join(mesh_source_dir, item)
            dst_path = os.path.join(mesh_dir, item)

            if not os.path.isfile(src_path):
                continue  # Skip directories

            # Special handling for info file
            if item == "info":
                print(f"Copying info file: {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)
                continue

            # Copy index files
            if item.endswith('.index'):
                print(f"Copying index file: {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)
                # Check if copied successfully
                if os.path.exists(dst_path):
                    print(f"  ✓ Successfully copied {item} ({os.path.getsize(dst_path)} bytes)")
                else:
                    print(f"  ✗ Failed to copy {item}!")
                continue

            # Copy data files (numeric filenames)
            if item.isdigit():
                print(f"Copying data file: {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)
                # Check if copied successfully
                if os.path.exists(dst_path):
                    print(f"  ✓ Successfully copied {item} ({os.path.getsize(dst_path)} bytes)")
                else:
                    print(f"  ✗ Failed to copy {item}!")
                continue

            # Copy any other files
            print(f"Copying other file: {src_path} -> {dst_path}")
            shutil.copy2(src_path, dst_path)

        # Verify all necessary files were copied
        print(f"\nVerifying mesh files in destination directory {mesh_dir}:")
        dest_files = os.listdir(mesh_dir)
        dest_index_files = [f for f in dest_files if f.endswith('.index')]
        dest_data_files = [f for f in dest_files if f.isdigit() and os.path.isfile(os.path.join(mesh_dir, f))]

        print(
            f"Destination directory contains {len(dest_index_files)} index files and {len(dest_data_files)} data files")

        # Check for any missing index files
        for idx_file in index_files:
            if idx_file not in dest_files:
                print(f"WARNING: Index file {idx_file} not copied to destination!")

        # Check for any missing data files
        for data_file in data_files:
            if data_file not in dest_files:
                print(f"WARNING: Data file {data_file} not copied to destination!")

        # Check for paired files
        for idx_file in dest_index_files:
            base_name = idx_file[:-6]  # Remove .index suffix
            if base_name not in dest_data_files:
                print(f"WARNING: Missing data file for index {idx_file}")

        for data_file in dest_data_files:
            if f"{data_file}.index" not in dest_index_files:
                print(f"WARNING: Missing index file for data {data_file}")

    return zarr_path


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

    response = requests.get(ome_xml_path)
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

    temp_mesh_dir = Path("temp_precomputed")
    mesh_writer = NeuroglancerMeshWriter(
        temp_mesh_dir,
        box_size=32,
        vertex_quantization_bits=16,
        transform=[1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0],
        clean_output=True
    )
    mesher = Mesher((1.0, 1.0, 1.0))

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
        mesher.mesh(mask)

        try:
            # v2, f2, n2, values2 = measure.marching_cubes(volume=mask)
            print("Use the Mesher")
            mesh = mesher.get(1, normals=True)
            trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

            # Apply smoothing
            print(f"Smoothing mesh {_key} ({len(trimesh_mesh.vertices)} vertices)...")
            trimesh_mesh = trimesh_mesh.smoothed(lamb=0.75, iterations=smoothing)

            # Fix any mesh issues
            if not trimesh_mesh.is_watertight:
                print(f"Fixing non-watertight mesh {_key}...")
                trimesh_mesh.fill_holes()

            trimesh_mesh.fix_normals()

            # Ensure proper alignment with the image coordinate system
            print(f"Applying proper coordinate alignment for mesh")
            # We need to ensure the mesh coordinates are in the same space as the image/labels data
            # No offset is needed as the vertices are already in the correct coordinate space
            # Explicitly stating this to maintain clarity across renderers
            trimesh_mesh.vertices = trimesh_mesh.vertices.copy()

            # Process the mesh with multiple LODs
            mesh_writer.process_mesh(_key, trimesh_mesh, num_lods=3)
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

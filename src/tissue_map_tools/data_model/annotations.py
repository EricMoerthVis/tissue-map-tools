from typing import Literal, Any, Self, cast

from cloudvolume import CloudVolume
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
    conlist,
    conint,
    confloat,
)
from tissue_map_tools.data_model.sharded import ShardingSpecification
from pathlib import PurePosixPath
import re
import struct
from pathlib import Path
from scipy.spatial import KDTree
import numpy as np
from dask.dataframe import DataFrame as DaskDataFrame
import pandas as pd
from numpy.random import default_rng
import itertools
from numpy.typing import NDArray
import json

RNG = default_rng(42)

PRINT_DEBUG = True
VISUAL_DEBUG = False

ValidNumericNDArray = (
    NDArray[np.uint8]
    | NDArray[np.int8]
    | NDArray[np.uint16]
    | NDArray[np.int16]
    | NDArray[np.uint32]
    | NDArray[np.int32]
    | NDArray[np.float32]
)
# the neuroglancer specs also allow for "rgb" and "rgba", but these are not native
# Python types
SUPPORTED_DTYPES = [
    np.uint32,
    np.int32,
    np.float32,
    np.uint16,
    np.int16,
    np.uint8,
    np.int8,
    "category",
]
# fmt: off
SI_PREFIXES = [
    "Y", "Z", "E", "P", "T", "G", "M", "k", "h", "", "c", "m", "u", "µ", "n", "p", "f",
    "a", "z", "y"
]
# fmt: on
SI_SUFFIXES = ["m", "s"]

si_units = [prefix + suffix for prefix in SI_PREFIXES for suffix in SI_SUFFIXES]


class AnnotationProperty(BaseModel):
    id: str
    type: Literal[
        "rgb", "rgba", "uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"
    ]
    description: str | None = None
    enum_values: list[Any] | None = None
    enum_labels: list[str] | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v):
        regexp = r"^[a-z][a-zA-Z0-9_]*$"
        if not re.match(regexp, v):
            raise ValueError(
                f"Property 'id' must match the regular expression {regexp}."
            )
        return v

    @model_validator(mode="after")
    def _validate_enum_labels_and_values(self) -> Self:
        # Only allow enum_values for numeric types (not rgb/rgba)
        numeric_types = {
            "uint8",
            "int8",
            "uint16",
            "int16",
            "uint32",
            "int32",
            "float32",
        }
        if self.enum_values is not None:
            if self.type not in numeric_types:
                raise ValueError(
                    "'enum_values' is only allowed for numeric types, not for 'rgb' or 'rgba'."
                )
            if self.enum_labels is None:
                raise ValueError(
                    "'enum_labels' must be specified if 'enum_values' is specified."
                )
            if len(self.enum_labels) != len(self.enum_values):
                raise ValueError(
                    "'enum_labels' must have the same length as 'enum_values'."
                )
        if self.enum_labels is not None and self.enum_values is None:
            raise ValueError(
                "'enum_values' must be specified if 'enum_labels' is specified."
            )
        return self


def validate_key_path(key: str) -> str:
    if not isinstance(key, str):
        raise ValueError("key must be a string")
    if key == "":
        raise ValueError("key must not be empty")
    # Disallow empty path components (e.g. 'foo//bar')
    if "//" in key:
        raise ValueError("key must not contain empty path components ('//')")
    path = PurePosixPath(key)
    if path.is_absolute():
        raise ValueError("key must not be an absolute path (must not start with '/')")
    # Disallow '..' as the only component
    if len(path.parts) == 1 and path.parts[0] == "..":
        raise ValueError("key must not be just '..'")
    # Only allow valid path characters (alphanumeric, _, -, ., /)
    if not re.match(r"^[a-zA-Z0-9_\-./]+$", key):
        raise ValueError("key contains invalid characters")
    return key


class AnnotationRelationship(BaseModel):
    id: str
    key: str
    sharding: ShardingSpecification | None = None

    _validate_key = field_validator("key")(validate_key_path)


class AnnotationById(BaseModel):
    key: str
    sharding: ShardingSpecification | None = None

    _validate_key = field_validator("key")(validate_key_path)


class AnnotationSpatialLevel(BaseModel):
    key: str
    sharding: ShardingSpecification | None = None
    # maybe there is a better syntax that doesn't make mypy unhappy; but this works
    grid_shape: conlist(conint(gt=0), min_length=1)  # type: ignore[valid-type]
    chunk_size: conlist(confloat(gt=0), min_length=1)  # type: ignore[valid-type]
    limit: conint(gt=0)  # type: ignore[valid-type]

    _validate_key = field_validator("key")(validate_key_path)


class AnnotationInfo(BaseModel):
    type: Literal["neuroglancer_annotations_v1"] = Field(..., alias="@type")
    dimensions: dict[str, tuple[int | float, str]] = Field(
        ...,
        description="Dimensions of the annotation space, with scale and unit.",
    )
    lower_bound: list[float]
    upper_bound: list[float]
    annotation_type: Literal["POINT", "LINE", "AXIS_ALIGNED_BOUNDING_BOX", "ELLIPSOID"]
    properties: list[AnnotationProperty]
    relationships: list[AnnotationRelationship]
    by_id: AnnotationById
    spatial: list[AnnotationSpatialLevel]

    model_config = ConfigDict(validate_by_name=True)

    @property
    def rank(self) -> int:
        """Return the rank of the annotation space, which is the length of the dimensions."""
        return len(self.dimensions)

    @field_validator("dimensions", mode="before")
    @classmethod
    def _ensure_valid_dimensions(
        cls, values: dict[str, Any]
    ) -> dict[str, tuple[int | float, str]]:
        validated = {}
        for key, value in values.items():
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(
                    f"Dimension '{key}' must be a two-element list [scale, unit]."
                )
            scale, unit = value
            if not (isinstance(scale, (int, float)) and scale > 0):
                raise ValueError(
                    f"Scale for dimension '{key}' must be a positive number."
                )
            if not isinstance(unit, str):
                raise ValueError(f"Unit for dimension '{key}' must be a string.")
            if unit not in si_units:
                raise ValueError(
                    f"Unit '{unit}' for dimension '{key}' is not supported. Only the "
                    f"following combinations of SI prefixes and suffixes are allowed: prefixes: {SI_PREFIXES}; suffixes: {SI_PREFIXES} (e.g. 'um')."
                )
            validated[key] = (scale, unit)
        return validated

    @model_validator(mode="after")
    def _ensure_rank_well_defined(self) -> Self:
        if len(self.lower_bound) != len(self.upper_bound):
            raise ValueError("Lower and upper bounds must have the same length.")
        if len(self.lower_bound) != len(self.dimensions):
            raise ValueError(
                "Length of lower and upper bounds must match the number of dimensions."
            )
        return self

    @field_validator("relationships")
    @classmethod
    def _ensure_unique_relationship_ids(
        cls, relationships: list[AnnotationRelationship]
    ) -> list[AnnotationRelationship]:
        seen_ids = set()
        for relationship in relationships:
            if relationship.id in seen_ids:
                raise ValueError(f"Duplicate relationship id found: {relationship.id}")
            seen_ids.add(relationship.id)
        return relationships

    @model_validator(mode="after")
    def _validate_spatial_rank(self) -> Self:
        rank = self.rank
        for i, spatial in enumerate(self.spatial):
            if len(spatial.grid_shape) != rank:
                raise ValueError(
                    f"spatial[{i}].grid_shape length {len(spatial.grid_shape)} does not match rank {rank}"
                )
            if len(spatial.chunk_size) != rank:
                raise ValueError(
                    f"spatial[{i}].chunk_size length {len(spatial.chunk_size)} does not match rank {rank}"
                )
        return self


def encode_positions_and_properties_via_single_annotation(
    info: AnnotationInfo,
    positions_values: list[float],
    properties_values: dict[str, Any],
) -> bytes:
    """
    Encode positions and properties of a single annotation to binary format.

    Parameters
    ----------
    info
        The annotation schema information.
    positions_values
        The position values to encode.
    properties_values
        The property values to encode.

    Returns
    -------
    bytes
        The encoded binary data representing the annotation.

    Notes
    -----
    This function is used internally to encode the data for the various indexes.
    """
    rank = info.rank
    buf = bytearray()
    # 1. Encode geometry
    if info.annotation_type == "POINT":
        buf += struct.pack(f"<{rank}f", *positions_values)
    elif info.annotation_type in ("LINE", "AXIS_ALIGNED_BOUNDING_BOX", "ELLIPSOID"):
        buf += struct.pack(f"<{2 * rank}f", *positions_values)
    else:
        raise ValueError(f"Unknown annotation_type: {info.annotation_type}")

    # 2. Encode properties in order
    for prop in info.properties:
        value = properties_values[prop.id]
        if prop.type in ("uint32", "int32", "float32"):
            fmt = {"uint32": "<I", "int32": "<i", "float32": "<f"}[prop.type]
            buf += struct.pack(fmt, value)
        elif prop.type in ("uint16", "int16"):
            fmt = {"uint16": "<H", "int16": "<h"}[prop.type]
            buf += struct.pack(fmt, value)
        elif prop.type in ("uint8", "int8"):
            fmt = {"uint8": "<B", "int8": "<b"}[prop.type]
            buf += struct.pack(fmt, value)
        elif prop.type == "rgb":
            buf += struct.pack("<3B", *value)
        elif prop.type == "rgba":
            buf += struct.pack("<4B", *value)
        else:
            raise ValueError(f"Unknown property type: {prop.type}")

    # 3. Pad to 4-byte boundary
    pad = (4 - (len(buf) % 4)) % 4
    buf += b"\x00" * pad
    return bytes(buf)


def decode_positions_and_properties_via_single_annotation(
    info: AnnotationInfo,
    data: bytes,
    offset: int = 0,
) -> tuple[list[float], dict[str, Any], int]:
    """
    Decode positions and properties of a single annotation from binary format.

    Parameters
    ----------
    info
        The annotation schema information.
    data
        The binary data to decode.
    offset
        The starting offset in the data buffer (default is 0).

    Returns
    -------
    positions_values
        The decoded position values.
    properties_values
        The decoded property values.
    offset
        The offset after reading the annotation.
    """
    rank = info.rank
    # 1. Decode geometry
    if info.annotation_type == "POINT":
        n_floats = rank
    elif info.annotation_type in ("LINE", "AXIS_ALIGNED_BOUNDING_BOX", "ELLIPSOID"):
        n_floats = 2 * rank
    else:
        raise ValueError(f"Unknown annotation_type: {info.annotation_type}")
    positions_values = list(struct.unpack_from(f"<{n_floats}f", data, offset))
    offset += 4 * n_floats

    # 2. Decode properties in order
    properties_values = {}
    for prop in info.properties:
        if prop.type == "uint32":
            (value,) = struct.unpack_from("<I", data, offset)
            offset += 4
        elif prop.type == "int32":
            (value,) = struct.unpack_from("<i", data, offset)
            offset += 4
        elif prop.type == "float32":
            (value,) = struct.unpack_from("<f", data, offset)
            offset += 4
        elif prop.type == "uint16":
            (value,) = struct.unpack_from("<H", data, offset)
            offset += 2
        elif prop.type == "int16":
            (value,) = struct.unpack_from("<h", data, offset)
            offset += 2
        elif prop.type == "uint8":
            (value,) = struct.unpack_from("<B", data, offset)
            offset += 1
        elif prop.type == "int8":
            (value,) = struct.unpack_from("<b", data, offset)
            offset += 1
        elif prop.type == "rgb":
            value = list(struct.unpack_from("<3B", data, offset))
            offset += 3
        elif prop.type == "rgba":
            value = list(struct.unpack_from("<4B", data, offset))
            offset += 4
        else:
            raise ValueError(f"Unknown property type: {prop.type}")
        properties_values[prop.id] = value

    # 3. Skip padding to 4-byte boundary
    pad = (4 - (offset % 4)) % 4
    offset += pad
    return positions_values, properties_values, offset


def encode_positions_and_properties_and_relationships_via_single_annotation(
    info: AnnotationInfo,
    positions_values: list[float],
    properties_values: dict[str, Any],
    relationships_values: dict[str, list[int]],
) -> bytes:
    """
    Encode positions, properties and relationships using the single annotation encoding

    Parameters
    ----------
    info
        The annotation schema information.
    positions_values
        The position values to encode.
    properties_values
        The property values to encode.
    relationships_values
        The relationship values to encode.

    Returns
    -------
    bytes
        The encoded binary data representing the annotation, including positions,
        properties, and relationships.

    Notes
    -----
    This encoding is used for the annotation ID index
    """
    buf = bytearray(
        encode_positions_and_properties_via_single_annotation(
            info=info,
            positions_values=positions_values,
            properties_values=properties_values,
        )
    )

    # Encode relationships
    for rel in info.relationships:
        rel_ids = relationships_values.get(rel.id, [])
        buf += struct.pack("<I", len(rel_ids))
        for rid in rel_ids:
            buf += struct.pack("<Q", rid)

    return bytes(buf)


def decode_positions_and_properties_and_relationships_via_single_annotation(
    info: AnnotationInfo,
    data: bytes,
) -> tuple[list[float], dict[str, Any], dict[str, list[int]]]:
    """
    Decode positions, properties and relationships from binary data (single annotation)

    Parameters
    ----------
    info
       The annotation schema information.
    data
       The binary data to decode.

    Returns
    -------
    positions_values
        The decoded position values.
    properties_values
        The decoded property values.
    relationships_values
        The decoded relationship values.

    Notes
    -----
    This decoding is used for the annotation ID index.
    """
    positions_values, properties_values, offset = (
        decode_positions_and_properties_via_single_annotation(data=data, info=info)
    )

    # 4. Decode relationships
    relationships_values = {}
    for rel in info.relationships:
        (num_ids,) = struct.unpack_from("<I", data, offset)
        offset += 4
        ids = []
        for _ in range(num_ids):
            (rid,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            ids.append(rid)
        relationships_values[rel.id] = ids

    return positions_values, properties_values, relationships_values


def encode_positions_and_properties_via_multiple_annotation(
    info: AnnotationInfo,
    annotations: list[tuple[int, list[float], dict[str, Any]]],
) -> bytes:
    """
    Encode via positions and properties via the multiple annotation encoding.

    Parameters
    ----------
    info
        The annotation schema information.
    annotations
        A list of tuples, each containing the annotation ID, position values,
        and property values.

    Returns
    -------
    bytes
        The encoded binary data representing all annotations for a related object,
        including positions, properties, and annotation IDs.

    Notes
    -----
    This encoding is used for the related object ID index and for the spatial index.
    Note that relationships are not encoded in this function.
    """
    buf = bytearray()
    count = len(annotations)
    buf += struct.pack("<Q", count)

    # Encode positions and properties for all annotations
    for _, positions_values, properties_values in annotations:
        buf += encode_positions_and_properties_via_single_annotation(
            info=info,
            positions_values=positions_values,
            properties_values=properties_values,
        )

    # Encode annotation ids for all annotations
    for ann_id, _, _ in annotations:
        buf += struct.pack("<Q", ann_id)

    return bytes(buf)


def decode_positions_and_properties_via_multiple_annotation(
    info: AnnotationInfo,
    data: bytes,
) -> list[tuple[int, list[float], dict[str, Any]]]:
    """
    Decode positions and properties from binary data (multiple annotation encoding)

    Parameters
    ----------
    info
        The annotation schema information.
    data
        The binary data to decode.

    Returns
    -------
    A list of tuples, each containing the annotation ID, position values, and
    property values. Relationships are not decoded in this function.

    Notes
    -----
    This decoding is used for the related object ID index and for the spatial index.
    Note that relationships are not decoded in this function.
    """
    (count,) = struct.unpack_from("<Q", data)
    offset = 8

    decoded_annotations_data = []
    # First pass: decode positions and properties
    for _ in range(count):
        positions_values, properties_values, offset = (
            decode_positions_and_properties_via_single_annotation(
                data=data,
                info=info,
                offset=offset,
            )
        )
        decoded_annotations_data.append((positions_values, properties_values))

    # Second pass: decode annotation ids
    decoded_annotations = []
    for i in range(count):
        (ann_id,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        positions_values, properties_values = decoded_annotations_data[i]
        decoded_annotations.append((ann_id, positions_values, properties_values))

    return decoded_annotations


def write_annotation_id_index(
    info: AnnotationInfo,
    root_path: Path,
    annotations: dict[int, tuple[list[float], dict[str, Any], dict[str, list[int]]]],
):
    """
    Write the annotation ID index to disk (unsharded format).

    Parameters
    ----------
    info
        The annotation schema information.
    root_path
        The root directory where the annotation index will be written.
    annotations
        A dictionary mapping annotation IDs to tuples containing position values,
        property values, and relationship values.

    Raises
    ------
    NotImplementedError
        If sharding is specified in the annotation info.
    """
    if info.by_id.sharding is not None:
        raise NotImplementedError(
            "Sharded annotation ID index writing is not implemented."
        )

    index_dir = root_path / info.by_id.key
    index_dir.mkdir(parents=True, exist_ok=True)

    for annotation_id, (positions, properties, relationships) in annotations.items():
        encoded_data = (
            encode_positions_and_properties_and_relationships_via_single_annotation(
                info=info,
                positions_values=positions,
                properties_values=properties,
                relationships_values=relationships,
            )
        )
        with open(index_dir / str(annotation_id), "wb") as f:
            f.write(encoded_data)


def read_annotation_id_index(
    info: AnnotationInfo,
    root_path: Path,
) -> dict[int, tuple[list[float], dict[str, Any], dict[str, list[int]]]]:
    """
    Read the annotation ID index from disk (unsharded format).

    Parameters
    ----------
    info
        The annotation schema information.
    root_path
        The root directory where the annotation index is stored.

    Returns
    -------
    A dictionary mapping annotation IDs to tuples containing position values,
    property values, and relationship values.

    Raises
    ------
    NotImplementedError
        If sharding is specified in the annotation info.
    FileNotFoundError
        If the annotation ID index directory does not exist.
    """
    if info.by_id.sharding is not None:
        raise NotImplementedError(
            "Sharded annotation ID index reading is not implemented."
        )

    index_dir = root_path / info.by_id.key
    if not index_dir.is_dir():
        raise FileNotFoundError(
            f"Annotation ID index directory '{index_dir}' does not exist."
        )

    annotations = {}
    for fpath in index_dir.iterdir():
        if fpath.is_file():
            if re.match(r"^\d+$", fpath.name) is None:
                # Ignore files that are not valid uint64 ids
                continue
            annotation_id = int(fpath.name)

            with open(fpath, "rb") as f:
                encoded_data = f.read()

            decoded_data = (
                decode_positions_and_properties_and_relationships_via_single_annotation(
                    data=encoded_data, info=info
                )
            )
            annotations[annotation_id] = decoded_data
    return annotations


def write_related_object_id_index(
    info: AnnotationInfo,
    root_path: Path,
    annotations_by_object_id: dict[
        str,
        dict[int, list[tuple[int, list[float], dict[str, Any]]]],
    ],
):
    """
    Write the related object ID index to disk (unsharded format).

    Parameters
    ----------
    info
        The annotation schema information.
    root_path
        The root directory where the related object ID index will be written.
    annotations_by_object_id
        A dictionary mapping relationship id to a dictionary mapping object id to a list
        of annotations. Each annotation is a tuple of (annotation id, position values,
        property values).

    Raises
    ------
    NotImplementedError
        If sharding is specified for any relationship in the annotation info.
    """
    for rel in info.relationships:
        if rel.sharding is not None:
            raise NotImplementedError(
                f"Sharded related object ID index writing for relationship '{rel.id}' is not implemented."
            )

        index_dir = root_path / rel.key
        index_dir.mkdir(parents=True, exist_ok=True)

        if rel.id in annotations_by_object_id:
            for (
                object_id,
                annotations,
            ) in annotations_by_object_id[rel.id].items():
                encoded_data = encode_positions_and_properties_via_multiple_annotation(
                    info=info,
                    annotations=annotations,
                )
                with open(index_dir / str(object_id), "wb") as f:
                    f.write(encoded_data)


def read_related_object_id_index(
    info: AnnotationInfo,
    root_path: Path,
) -> dict[str, dict[int, list[tuple[int, list[float], dict[str, Any]]]]]:
    """
    Read the related object ID index from disk (unsharded format).

    Parameters
    ----------
    info
        The annotation schema information.
    root_path
        The root directory where the related object ID index is stored.

    Returns
    -------
    A dictionary mapping relationship id to a dictionary mapping object id to a list
    of annotations. Each annotation is a tuple of (annotation id, position values,
    property values).

    Raises
    ------
    NotImplementedError
        If sharding is specified for any relationship in the annotation info.
    FileNotFoundError
        If the related object ID index directory does not exist.
    """
    all_relationships_data = {}
    for rel in info.relationships:
        if rel.sharding is not None:
            raise NotImplementedError(
                f"Sharded related object ID index reading for relationship '{rel.id}' is not implemented."
            )

        index_dir = root_path / rel.key
        if not index_dir.is_dir():
            raise FileNotFoundError(
                f"Related object ID index directory '{index_dir}' does not exist."
            )

        relationship_data = {}
        for fpath in index_dir.iterdir():
            if fpath.is_file():
                if re.match(r"^\d+$", fpath.name) is None:
                    continue
                object_id = int(fpath.name)

                with open(fpath, "rb") as f:
                    encoded_data = f.read()

                decoded_data = decode_positions_and_properties_via_multiple_annotation(
                    data=encoded_data, info=info
                )
                relationship_data[object_id] = decoded_data
        all_relationships_data[rel.id] = relationship_data
    return all_relationships_data


class GridLevel:
    def __init__(
        self,
        level: int,
        grid_shape: tuple[int, ...],
        mins: NDArray[np.float64],
        maxs: NDArray[np.float64],
        limit: int,
        parent_cells: list[tuple[int, ...]],
        parent_grid_shape: tuple[int, ...],
    ) -> None:
        self.level: int = level
        self.grid_shape: tuple[int, ...] = grid_shape
        self.mins: NDArray[np.float64] = mins
        self.maxs: NDArray[np.float64] = maxs
        self.limit: int = limit

        # quantities derived in this function call
        self.sizes: NDArray[np.float64] = np.array(maxs) - np.array(mins)
        self.chunk_size: NDArray[np.float64] = self.sizes / np.array(self.grid_shape)
        self.cells: list[tuple[int, ...]] = []

        # quantities set later
        self.populated_cells: dict[tuple[int, ...], NDArray[np.float64]] = {}

        if len(self.mins) != 3 or len(self.maxs) != 3:
            raise NotImplementedError("GridLevel only supports 3D grids at the moment.")

        for parent_cell in parent_cells:
            new_cells_by_dim: dict[int, tuple[int, ...]] = {}
            for dim in range(3):
                index = parent_cell[dim]
                factor = grid_shape[dim] // parent_grid_shape[dim]
                if factor == 1:
                    new_cells_by_dim[dim] = (index,)
                else:
                    new_cells_by_dim[dim] = (index * factor, index * factor + 1)
            new_cells = cast(
                list[tuple[int, ...]],
                itertools.product(*new_cells_by_dim.values()),
            )
            self.cells.extend(new_cells)

    def iter_cells(self):
        for i, j, k in self.cells:
            yield (i, j, k)

    def centroid(self, cell: tuple[int, ...]) -> NDArray[np.float64]:
        """Calculate the centroid of the grid cell."""
        return np.array(cell) * self.chunk_size + self.chunk_size / 2 + self.mins

    def get_next_grid_shape(self) -> tuple[int, ...]:
        """Get the shape of the next grid level so that we get isotropic chunks.

        Notes
        -----
        The specs say: "each component of chunk_size of each successively level
        should be either equal to, or half of, the corresponding component of the
        prior level chunk_size, whichever results in a more spatially isotropic
        chunk."

        We implement this as follows: if a chunk length for a given axis is half
        (or less) than the size of any other chunk, then we leave this axis as is,
        otherwise we divide the chunk size by 2 (i.e. we multiply the grid shape by 2).
        """
        next_grid_shape = list(self.grid_shape)
        for i in range(3):
            if any(
                [
                    self.chunk_size[i] * 2 <= self.chunk_size[j].item()
                    for j in range(3)
                    if j != i
                ]
            ):
                continue
            else:
                next_grid_shape[i] *= 2
        return tuple(next_grid_shape)


def get_coordinates_and_kd_tree(
    points: pd.DataFrame | DaskDataFrame,
) -> tuple[ValidNumericNDArray, KDTree]:
    """
    Extract xyz coordinates and create a KDTree.
    """
    # TODO: we can generalize to 2D points. 1D points do not make much sense. If we
    #  only hjave 1D or 2D point the following line will throw an error.
    if isinstance(points, DaskDataFrame):
        xyz = points[["x", "y", "z"]].compute().values
    else:
        xyz = points[["x", "y", "z"]].values
    if xyz.dtype not in SUPPORTED_DTYPES:
        # TODO: make a test for this
        raise TypeError(
            f"Unsupported type for xyz coordinates: {type(xyz).__name__}; supported "
            f"types are {ValidNumericNDArray.__args__}."
        )
    kd_tree = KDTree(xyz)
    return xyz, kd_tree


def compute_spatial_index(
    xyz: ValidNumericNDArray,
    kd_tree: KDTree | None = None,
    limit: int = 1000,
    starting_grid_shape: tuple[int, ...] | None = None,
) -> dict[int, GridLevel]:
    # TODO: only points are supported at the moment, not lines, axis-aligned bounding
    #  boxes and ellipsoids
    if starting_grid_shape is None:
        starting_grid_shape = (1, 1, 1)

    if kd_tree is None:
        kd_tree = KDTree(xyz)

    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)

    remaining_indices = set(range(len(xyz)))

    # the default case is that starting_grid_shape is (1, 1, 1), which means that
    # parent_cells is [(0, 0, 0)].
    parent_cells = list(itertools.product(*[range(s) for s in starting_grid_shape]))

    grid: dict[int, GridLevel] = {}
    grid_level = GridLevel(
        level=0,
        grid_shape=starting_grid_shape,
        mins=mins,
        maxs=maxs,
        limit=limit,
        parent_cells=parent_cells,
        parent_grid_shape=starting_grid_shape,
    )
    # to avoid the risk of points in the boundary of the grid not being included
    eps = 1e-6
    len_previous_remaining_indices = len(remaining_indices)

    while len(remaining_indices) > 0:
        # initialization
        grid[grid_level.level] = grid_level
        if PRINT_DEBUG:
            print(
                f"Processing grid level {grid_level.level} with shape {grid_level.grid_shape} "
                f"and chunk size {grid_level.chunk_size}. Remaining points: {len(remaining_indices)}"
            )

        # main logic
        # if PRINT_DEBUG:
        #     print("Active cells: ", grid_level.cells)
        for i, j, k in grid_level.iter_cells():
            # calculate the centroid of the grid cell
            centroid = grid_level.centroid((i, j, k))

            # find points in the grid cell
            # this filter points by a radius r, but we have different values per axis,
            # so we proceed with manual filtering on the result from the kDTree query
            indices = kd_tree.query_ball_point(
                centroid, r=grid_level.chunk_size.max().item() / 2 + eps, p=np.inf
            )
            filtered = xyz[indices]
            mask = (
                (centroid[0] - grid_level.chunk_size[0] - eps <= filtered[:, 0])
                & (filtered[:, 0] <= centroid[0] + grid_level.chunk_size[0] + eps)
                & (centroid[1] - grid_level.chunk_size[1] - eps <= filtered[:, 1])
                & (filtered[:, 1] <= centroid[1] + grid_level.chunk_size[1] + eps)
                & (centroid[2] - grid_level.chunk_size[2] - eps <= filtered[:, 2])
                & (filtered[:, 2] <= centroid[2] + grid_level.chunk_size[2] + eps)
            )
            discarded = np.sum(~mask).item()
            if discarded > 0:
                # TODO: **possible bug!** This message is not printed while I would
                #  expect that the kDTree query would sometimes return more points than
                #  the mask would allow (this should happend when chunk_size has
                #  different dimensions)
                # let's keep this print active until we address the comment above
                if PRINT_DEBUG or True:
                    print(
                        f"-----------------> {discarded} points where filtered out of"
                        f" {len(indices)}"
                    )
            indices = np.array(indices)[mask].tolist()

            # filter out points that are not in the grid cell because they have been
            # previously emitted
            indices = [i for i in indices if i in remaining_indices]

            if len(indices) > 0:
                if len(indices) <= limit:
                    emitted = indices
                    RNG.shuffle(emitted)
                else:
                    # BIG BUG! see here:
                    # https://github.com/google/neuroglancer/issues/227#issuecomment-916384909
                    emitted = RNG.choice(indices, size=limit, replace=False)
                if PRINT_DEBUG:
                    print(
                        f"Emitting {len(emitted)} points for grid cell ({i}, {j}, {k})"
                    )
                grid_level.populated_cells[(i, j, k)] = emitted
                remaining_indices.difference_update(emitted)

        if VISUAL_DEBUG:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            chunk_size = grid_level.chunk_size
            lines_x = np.arange(
                grid_level.mins[0],
                grid_level.maxs[0] + chunk_size[0],
                chunk_size[0] + eps,
            )
            lines_y = np.arange(
                grid_level.mins[1],
                grid_level.maxs[1] + chunk_size[1],
                chunk_size[1] + eps,
            )
            for x in lines_x:
                plt.plot(
                    [x, x],
                    [grid_level.mins[1], grid_level.maxs[1]],
                    color="red",
                    linewidth=0.5,
                )
            for y in lines_y:
                plt.plot(
                    [grid_level.mins[0], grid_level.maxs[0]],
                    [y, y],
                    color="red",
                    linewidth=0.5,
                )

            remaining_xyz = xyz[list(remaining_indices)]
            plt.scatter(
                remaining_xyz[:, 0],
                remaining_xyz[:, 1],
                s=100,
                c=remaining_xyz[:, 2],
            )
            if len(remaining_xyz) > 0:
                cbar = plt.colorbar()
                cbar.set_ticks([remaining_xyz[:, 2].min(), remaining_xyz[:, 2].max()])
                cbar.set_ticklabels(
                    [f"{remaining_xyz[:, 2].min()}", f"{remaining_xyz[:, 2].max()}"]
                )
            plt.show()

        # prepare for the next level
        grid_level = GridLevel(
            level=grid_level.level + 1,
            grid_shape=grid_level.get_next_grid_shape(),
            mins=grid_level.mins,
            maxs=grid_level.maxs,
            limit=limit,
            parent_cells=list(grid_level.populated_cells.keys()),
            parent_grid_shape=grid_level.grid_shape,
        )

        # sanity check
        if len(remaining_indices) == len_previous_remaining_indices:
            raise ValueError(
                "No points were emitted in this grid level. This is likely due to the "
                "grid size being too small. To fix, try increasing the `limit` "
                "parameter."
            )
        len_previous_remaining_indices = len(remaining_indices)
    return grid


def write_spatial_index(
    info: AnnotationInfo,
    root_path: Path,
    spatial_key: str,
    annotations_by_spatial_chunk: dict[
        str, list[tuple[int, list[float], dict[str, Any]]]
    ],
):
    """
    Writes a spatial index for annotations, grouping them by spatial chunk.

    Parameters
    ----------
    info : AnnotationInfo
        The annotation schema information.
    root_path : Path
        The root directory where the spatial index will be written.
    spatial_key : str
        Key indicating the level of teh spatial index
    annotations_by_spatial_chunk : dict of str to list of tuple
        Dictionary mapping spatial chunk identifiers (e.g., '0_0_0') to lists of
        annotation records. Each annotation record is a tuple:
            (id, [x, y, z], properties)
        where:
            id : int
                Unique identifier for the annotation.
            [x, y, z] : list of float
                Coordinates of the annotation in 3D space.
            properties : dict
                Additional properties or attributes associated with the annotation
                (e.g., {'x': 4153.0, 'y': 326.0, 'z': 110.1, 'gene': 21}).

    Example
    -------
    annotations_by_spatial_chunk = {
        '0_0_0': [
            (6635, [4153.0, 326.0, 110.15], {'x': 4153.0, 'y': 326.0, 'z': 110.15, 'gene': 21}),
            (6980, [4556.0, 5170.0, 55.07], {'x': 4556.0, 'y': 5170.0, 'z': 55.07, 'gene': 126}),
            ...
        ],
        ...
    }

    Returns
    -------
    None
        Writes spatial index files to disk.
    """
    annotation_spatial_level: AnnotationSpatialLevel | None = None
    for spatial in info.spatial:
        if spatial.key == spatial_key:
            annotation_spatial_level = spatial
            break
    if annotation_spatial_level is None:
        raise ValueError(
            f"Spatial level with key '{spatial_key}' not found in annotation info."
        )
    if annotation_spatial_level.sharding is not None:
        raise NotImplementedError(
            f"Sharded spatial index writing for relationship '{spatial_key}' is not "
            f"implemented."
        )

    index_dir = root_path / annotation_spatial_level.key
    index_dir.mkdir(parents=True, exist_ok=True)

    for cell_name, annotations in annotations_by_spatial_chunk.items():
        encoded_data = encode_positions_and_properties_via_multiple_annotation(
            info=info,
            annotations=annotations,
        )
        with open(index_dir / str(cell_name), "wb") as f:
            f.write(encoded_data)


def read_spatial_index(
    info: AnnotationInfo,
    root_path: Path,
    spatial_key: str,
) -> dict[str, list[tuple[int, list[float], dict[str, Any]]]]:
    """ """
    pass
    return {}
    # all_relationships_data = {}
    # for rel in info.relationships:
    #     if rel.sharding is not None:
    #         raise NotImplementedError(
    #             f"Sharded related object ID index reading for relationship '{rel.id}' is not implemented."
    #         )
    #
    #     index_dir = root_path / rel.key
    #     if not index_dir.is_dir():
    #         raise FileNotFoundError(
    #             f"Related object ID index directory '{index_dir}' does not exist."
    #         )
    #
    #     relationship_data = {}
    #     for fpath in index_dir.iterdir():
    #         if fpath.is_file():
    #             if re.match(r"^\d+$", fpath.name) is None:
    #                 continue
    #             object_id = int(fpath.name)
    #
    #             with open(fpath, "rb") as f:
    #                 encoded_data = f.read()
    #
    #             decoded_data = decode_related_object_id_index(
    #                 data=encoded_data, info=info
    #             )
    #             relationship_data[object_id] = decoded_data
    #     all_relationships_data[rel.id] = relationship_data
    # return all_relationships_data


def find_annotations_from_cloud_volume(cv: CloudVolume) -> list[str]:
    """
    Find all annotations in a cloud volume.

    Annotations do not appear to be listed in the root info file, so we need to
    manually explore each info file for each folder in the cloud volume path.
    """
    annotations_names = []
    cv_path = Path(cv.meta.path.basepath) / cv.meta.path.layer
    for subpath in cv_path.iterdir():
        subpath = Path(subpath)
        if not subpath.is_dir():
            continue

        # read the info file for this annotation
        info_file = subpath / "info"
        if not info_file.is_file():
            continue

        with open(info_file, "r") as f:
            info = json.load(f)
            type_ = info.get("@type", str(subpath))
            if type_ == "neuroglancer_annotations_v1":
                annotations_names.append(subpath.name)
    return annotations_names

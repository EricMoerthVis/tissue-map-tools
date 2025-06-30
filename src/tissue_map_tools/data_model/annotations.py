from typing import Literal, Any, Self, cast
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
import matplotlib.pyplot as plt
import numpy as np
from dask.dataframe import DataFrame as DaskDataFrame
import pandas as pd
from numpy.random import default_rng
import itertools
from numpy.typing import NDArray

RNG = default_rng(42)

PRINT_DEBUG = False


class AnnotationProperty(BaseModel):
    id: str
    type: Literal[
        "rgb", "rgba", "uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"
    ]
    description: str | None = None
    enum_values: list[Any] | None = None
    enum_labels: list[str] | None = None

    model_config = ConfigDict(validate_by_name=True, extra="allow")

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v):
        import re

        if not re.match(r"^[a-z][a-zA-Z0-9_]*$", v):
            raise ValueError(
                "Property 'id' must match the regular expression ^[a-z][a-zA-Z0-9_]*$."
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

    @field_validator("key")
    @classmethod
    def _validate_key(cls, v):
        return validate_key_path(v)


class AnnotationSpatialLevel(BaseModel):
    key: str
    sharding: ShardingSpecification | None = None
    # maybe there is a better syntax that doesn't make mypy unhappy; but this works
    grid_shape: conlist(conint(gt=0), min_length=1)  # type: ignore[valid-type]
    chunk_size: conlist(confloat(gt=0), min_length=1)  # type: ignore[valid-type]
    limit: conint(gt=0)  # type: ignore[valid-type]

    @field_validator("key")
    @classmethod
    def _validate_key(cls, v):
        return validate_key_path(v)


class AnnotationInfo(BaseModel):
    type: Literal["neuroglancer_annotations_v1"] = Field(..., alias="@type")
    dimensions: dict[str, tuple[int | float, str]] = Field(
        ...,
        description="Dimensions of the annotation space, with scale and unit (m and s are supported).",
    )
    lower_bound: list[float]
    upper_bound: list[float]
    annotation_type: Literal["POINT", "LINE", "AXIS_ALIGNED_BOUNDING_BOX", "ELLIPSOID"]
    properties: list[AnnotationProperty]
    relationships: list[AnnotationRelationship]
    by_id: AnnotationById
    spatial: list[AnnotationSpatialLevel]

    model_config = ConfigDict(validate_by_name=True, extra="allow")

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
            if unit not in ["m", "s"]:
                raise ValueError(
                    f"Unit '{unit}' for dimension '{key}' is not supported. Only 'm' and 's' are allowed."
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


def encode_positions_and_properties(
    info: AnnotationInfo,
    positions_values: list[float],
    properties_values: dict[str, Any],
) -> bytes:
    """
    Encode positions and properties of a single annotation to binary format.
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


def decode_positions_and_properties(
    data: bytes,
    info: AnnotationInfo,
    offset: int = 0,
) -> tuple[list[float], dict[str, Any], int]:
    """
    Decode positions and properties of a single annotation from binary format.
    Returns (positions_values, properties_values, offset)
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


def encode_annotation_id_index(
    info: AnnotationInfo,
    positions_values: list[float],
    properties_values: dict[str, Any],
    relationships_values: dict[str, list[int]],
) -> bytes:
    """
    Encode a single annotation of the annotation ID index to binary format.

    Notes
    -------
    This encoding makes use of the "single annotation encoding" approach, i.e. it
    encodes a single annotation with its positions, properties, and relationships.
    """
    buf = bytearray(
        encode_positions_and_properties(
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


def decode_annotation_id_index(
    data: bytes,
    info: AnnotationInfo,
) -> tuple[list[float], dict[str, Any], dict[str, list[int]]]:
    positions_values, properties_values, offset = decode_positions_and_properties(
        data=data, info=info
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


def encode_related_object_id_index(
    info: AnnotationInfo,
    annotations: list[tuple[int, list[float], dict[str, Any]]],
) -> bytes:
    """
    Encode a list of annotations for the related object ID index.
    This should only encode positions and properties, not relationships.
    """
    buf = bytearray()
    count = len(annotations)
    buf += struct.pack("<Q", count)

    # Encode positions and properties for all annotations
    for _, positions_values, properties_values in annotations:
        buf += encode_positions_and_properties(
            info=info,
            positions_values=positions_values,
            properties_values=properties_values,
        )

    # Encode annotation ids for all annotations
    for ann_id, _, _ in annotations:
        buf += struct.pack("<Q", ann_id)

    return bytes(buf)


def decode_related_object_id_index(
    data: bytes,
    info: AnnotationInfo,
) -> list[tuple[int, list[float], dict[str, Any]]]:
    """
    Decode a list of annotations for the related object ID index.
    This should only decode positions and properties, not relationships.
    """
    (count,) = struct.unpack_from("<Q", data)
    offset = 8

    decoded_annotations_data = []
    # First pass: decode positions and properties
    for _ in range(count):
        positions_values, properties_values, offset = decode_positions_and_properties(
            data=data,
            info=info,
            offset=offset,
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
    root_path: Path,
    annotations: dict[int, tuple[list[float], dict[str, Any], dict[str, list[int]]]],
    info: AnnotationInfo,
):
    """
    Write the annotation ID index to disk (unsharded format).
    """
    if info.by_id.sharding is not None:
        raise NotImplementedError(
            "Sharded annotation ID index writing is not implemented."
        )

    index_dir = root_path / info.by_id.key
    index_dir.mkdir(parents=True, exist_ok=True)

    for annotation_id, (positions, properties, relationships) in annotations.items():
        encoded_data = encode_annotation_id_index(
            info=info,
            positions_values=positions,
            properties_values=properties,
            relationships_values=relationships,
        )
        with open(index_dir / str(annotation_id), "wb") as f:
            f.write(encoded_data)


def read_annotation_id_index(
    root_path: Path,
    info: AnnotationInfo,
) -> dict[int, tuple[list[float], dict[str, Any], dict[str, list[int]]]]:
    """
    Read the annotation ID index from disk (unsharded format).
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

            decoded_data = decode_annotation_id_index(data=encoded_data, info=info)
            annotations[annotation_id] = decoded_data
    return annotations


def write_related_object_id_index(
    root_path: Path,
    info: AnnotationInfo,
    annotations_by_object_id: dict[
        str,
        dict[int, list[tuple[int, list[float], dict[str, Any]]]],
    ],
):
    """
    Write the related object ID index to disk (unsharded format).
    `annotations_by_object_id` is a dict mapping relationship id to a dict mapping object id to a list of annotations.
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
                encoded_data = encode_related_object_id_index(
                    info=info,
                    annotations=annotations,
                )
                with open(index_dir / str(object_id), "wb") as f:
                    f.write(encoded_data)


def read_related_object_id_index(
    root_path: Path,
    info: AnnotationInfo,
) -> dict[str, dict[int, list[tuple[int, list[float], dict[str, Any]]]]]:
    """
    Read the related object ID index from disk (unsharded format).
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

                decoded_data = decode_related_object_id_index(
                    data=encoded_data, info=info
                )
                relationship_data[object_id] = decoded_data
        all_relationships_data[rel.id] = relationship_data
    return all_relationships_data


class GridLevel:
    level: int
    grid_shape: list[int]
    chunk_size: NDArray[np.float64]
    limit: int

    def __init__(
        self,
        level: int,
        grid_shape: list[int],
        mins: NDArray[np.float64],
        maxs: NDArray[np.float64],
        limit: int,
        parent_cells: list[tuple[int, int, int]],
        parent_grid_shape: list[int],
    ) -> None:
        self.level = level
        self.grid_shape = grid_shape
        self.mins = mins
        self.maxs = maxs
        self.limit = limit

        # derived quantities
        self.sizes = np.array(maxs) - np.array(mins)
        self.chunk_size = self.sizes / np.array(self.grid_shape)
        self.cells: list[tuple[int, int, int]] = []

        # quantities set later
        self.populated_cells: dict[tuple[int, int, int], NDArray[np.float64]] = {}

        for parent_cell in parent_cells:
            new_cells_by_dim: dict[int, list[int]] = {}
            for dim in range(3):
                index = parent_cell[dim]
                factor = grid_shape[dim] // parent_grid_shape[dim]
                if factor == 1:
                    new_cells_by_dim[dim] = [index]
                else:
                    new_cells_by_dim[dim] = [index * factor, index * factor + 1]
            new_cells = cast(
                list[tuple[int, int, int]],
                itertools.product(*new_cells_by_dim.values()),
            )
            self.cells.extend(new_cells)

    def iter_full_grid(self):
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                for k in range(self.grid_shape[2]):
                    yield (i, j, k)

    def iter_cells(self):
        for i, j, k in self.cells:
            yield (i, j, k)

    def centroid(self, index: tuple[int, int, int]) -> NDArray[np.float64]:
        """Calculate the centroid of the grid cell."""
        return np.array(index) * self.chunk_size + self.chunk_size / 2 + self.mins

    def get_next_grid_shape(self) -> list[int]:
        """Get the shape of the next grid level so that we get isotropic chunks.

        Notes
        -----
        The specs say: "each component of chunk_size of each successively level
        should be either equal to, or half of, the corresponding component of the
        prior level chunk_size, whichever results in a more spatially isotropic
        chunk."

        We implement this as follows: if a chunk lenght for a given axis is half
        (or less) than the size of any other chunk, then we leave this axis as is,
        otherwise we divide the chunk size by 2.
        """
        next_grid_shape = self.grid_shape.copy()
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
        return next_grid_shape


def compute_spatial_index(
    points: DaskDataFrame | pd.DataFrame,
    limit: int = 1000,
    starting_grid_shape: list[int] | None = None,
) -> dict[int, GridLevel]:
    # TODO: only points are supported at the moment, not lines, axis-aligned bounding
    #  boxes and ellipsoids
    if starting_grid_shape is None:
        starting_grid_shape = [1, 1, 1]

    # TODO: we can generalize to 2D points. 1D points do not make much sense
    xyz = points[["x", "y", "z"]].values
    tree = KDTree(xyz)

    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)

    remaining_indices = set(range(len(xyz)))

    grid: dict[int, GridLevel] = {}
    grid_level = GridLevel(
        level=0,
        grid_shape=starting_grid_shape,
        mins=mins,
        maxs=maxs,
        limit=limit,
        parent_cells=[(0, 0, 0)],
        parent_grid_shape=starting_grid_shape,
    )
    # to avoid the risk of points in the boundary of the grid not being included
    eps = 1e-6
    previous_remaining_indices = len(remaining_indices)

    while len(remaining_indices) > 0:
        # initialization
        grid[grid_level.level] = grid_level
        if PRINT_DEBUG or True:
            print(
                f"Processing grid level {grid_level.level} with shape {grid_level.grid_shape} "
                f"and chunk size {grid_level.chunk_size}. Remaining points: {len(remaining_indices)}"
            )

        # main logic
        if PRINT_DEBUG:
            print("Active cells: ", grid_level.cells)
        for i, j, k in grid_level.iter_cells():
            # calculate the centroid of the grid cell
            centroid = grid_level.centroid((i, j, k))

            # find points in the grid cell
            # this filter points by a radius r, but we have different values per axis
            indices = tree.query_ball_point(
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
                # TODO: possible bug! This message is not printed while I would
                #  expect that the kDTree query would return more points than the mask
                #  would allow (this should happend when chunk_size has different
                #  dimensions
                if PRINT_DEBUG or True:
                    print(
                        f"-----------------> {discarded} points where filtered out of"
                        f" {len(indices)}"
                    )
            indices = np.array(indices)[mask].tolist()

            # filter out points that are not in the grid cell
            indices = [i for i in indices if i in remaining_indices]

            if len(indices) > 0:
                if len(indices) <= limit:
                    emitted = indices
                else:
                    emitted = RNG.choice(indices, size=limit, replace=False)
                if PRINT_DEBUG:
                    print(
                        f"Emitting {len(emitted)} points for grid cell ({i}, {j}, {k})"
                    )
                grid_level.populated_cells[(i, j, k)] = xyz[emitted]
                remaining_indices.difference_update(emitted)

                # create a new layer for this grid cell
                # layer_name = f"level_{grid_level.level}_cell_{i}_{j}_{k}"

                # here we would save the points to the precomputed format
                # e.g., save_points_to_precomputed(points[indices], precomputed_path, layer_name)

        # np.take(xyz, indices, axis=0)
        remaining_xyz = xyz[list(remaining_indices)]

        # visual debug
        VISUAL_DEBUG = False
        if VISUAL_DEBUG:
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
        if len(remaining_indices) == previous_remaining_indices:
            raise ValueError(
                "No points were emitted in this grid level. This is likely due to the "
                "grid size being too small."
            )
        previous_remaining_indices = len(remaining_indices)

    print("spatial index computed, now saving to precomputed format")
    return grid

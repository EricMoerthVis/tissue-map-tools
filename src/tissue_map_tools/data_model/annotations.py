from typing import Literal, Any, Self
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from tissue_map_tools.data_model.sharded import ShardingSpecification
from pathlib import PurePosixPath
import re


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

    _validate_key = field_validator("key")(validate_key_path)


class AnnotationSpatialLevel(BaseModel):
    key: str
    sharding: ShardingSpecification | None = None
    grid_shape: list[int]
    chunk_size: list[float]
    limit: int

    _validate_key = field_validator("key")(validate_key_path)


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

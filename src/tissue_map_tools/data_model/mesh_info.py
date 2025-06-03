from typing import Literal, Optional, List
from pydantic import BaseModel, Field, ConfigDict


class ShardingSpecification(BaseModel):
    """
    Pydantic model for the sharding specification.
    Based on the example: {"@type": "neuroglancer_uint64_sharded_v1", "preshift_bits": 0, "hash": "murmurhash3_x86_128", "minishard_bits": 0, "shard_bits": 0, "minishard_index_encoding": "gzip", "data_encoding": "raw"}
    And the reference to ./sharded.md in specs.md
    """

    type: str = Field(..., alias="@type")
    preshift_bits: Optional[int] = None
    hash_function: Optional[str] = Field(
        None, alias="hash"
    )  # 'hash' is a built-in, alias if needed
    minishard_bits: Optional[int] = None
    shard_bits: Optional[int] = None
    minishard_index_encoding: Optional[str] = None  # e.g., "gzip"
    data_encoding: Optional[str] = None  # e.g., "raw", "gzip"

    model_config = ConfigDict(validate_by_name=True, extra="allow")


class MultilodDracoInfo(BaseModel):
    """
    Pydantic model for the multi-resolution mesh info JSON file format.
    """

    type: Literal["neuroglancer_multilod_draco"] = Field(..., alias="@type")
    vertex_quantization_bits: Literal[10, 16]
    transform: List[float] = Field(..., min_length=12, max_length=12)
    lod_scale_multiplier: float
    sharding: Optional[ShardingSpecification] = None
    # as explained in the specs, if specified, the segment properties are used from here
    # only if the meshes is the data source. If the data source is the volume that
    # contains the meshes, the the segment properties should be specified in the volume
    # info file.
    segment_properties: Optional[str] = None

    model_config = ConfigDict(validate_by_name=True, extra="allow")

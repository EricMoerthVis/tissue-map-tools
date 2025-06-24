from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class ShardingSpecification(BaseModel):
    """
    Pydantic model for the sharding specification.

    See the full specification at: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md#sharding-specification
    """

    type: Literal["neuroglancer_uint64_sharded_v1"] = Field(..., alias="@type")
    preshift_bits: int
    # using a alias because 'hash' is a Python built-in
    hash_function: Literal["identity", "murmurhash3_x86_128"] = Field(..., alias="hash")
    minishard_bits: int
    shard_bits: int
    # when not specified, the default is 'raw'
    minishard_index_encoding: Optional[Literal["raw", "gzip"]] = None
    # same for data_encoding, default is 'raw'; in the case of multiscale meshes,
    # this encoding applies to the manifests but not to the mesh fragment data (which
    # uses Draco compression)
    data_encoding: Optional[Literal["raw", "gzip"]] = None

    model_config = ConfigDict(validate_by_name=True, extra="allow")

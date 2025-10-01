from typing import Literal, Optional, List
from pydantic import BaseModel, Field, ConfigDict

from tissue_map_tools.data_model.sharded import ShardingSpecification


class MultilodDracoInfo(BaseModel):
    """
    Pydantic model for the multi-resolution mesh format.

    See the full specification at: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md
    """

    type: Literal["neuroglancer_multilod_draco"] = Field(..., alias="@type")
    vertex_quantization_bits: Literal[10, 16]
    transform: List[float] = Field(..., min_length=12, max_length=12)
    lod_scale_multiplier: float
    sharding: Optional[ShardingSpecification] = None
    # as explained in the specs, if specified, the segment properties are used from here
    # only if the meshes is the data source. If the data source is the volume that
    # contains the meshes, then the segment properties should be specified in the volume
    # info file.
    segment_properties: Optional[str] = None

    model_config = ConfigDict(validate_by_name=True, extra="allow")

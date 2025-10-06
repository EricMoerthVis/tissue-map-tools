import pytest
from pydantic import ValidationError
from tissue_map_tools.data_model.mesh import MultilodDracoInfo

# Valid data examples
unsharded_data = {
    "@type": "neuroglancer_multilod_draco",
    "vertex_quantization_bits": 16,
    "transform": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "lod_scale_multiplier": 1.0,
}

sharded_data = {
    "@type": "neuroglancer_multilod_draco",
    "vertex_quantization_bits": 16,
    "transform": [897.0, 0, 0, 0, 0, 897.0, 0, 0, 0, 0, 465.0, 0],
    "lod_scale_multiplier": 1.0,
    "sharding": {
        "@type": "neuroglancer_uint64_sharded_v1",
        "preshift_bits": 0,
        "hash": "murmurhash3_x86_128",
        "minishard_bits": 10,
        "shard_bits": 10,
        "minishard_index_encoding": "gzip",
        "data_encoding": "raw",
    },
}

# Test data with extra fields (not in spec but should be allowed)
data_with_extra_fields = {
    "@type": "neuroglancer_multilod_draco",
    "vertex_quantization_bits": 16,
    "transform": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "lod_scale_multiplier": 1.0,
    "mip": 0,  # Extra field
    "chunk_size": [448, 448, 448],  # Extra field
    "spatial_index": {
        "resolution": [897.0, 897.0, 465.0],
        "chunk_size": [401856.0, 401856.0, 208320.0],
    },  # Extra field
    "unknown_field": "some_value",  # Extra field
}

# Invalid data examples
invalid_type_data = {
    "@type": "wrong_type",  # Invalid type
    "vertex_quantization_bits": 16,
    "transform": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "lod_scale_multiplier": 1.0,
}

invalid_vq_bits_data = {
    "@type": "neuroglancer_multilod_draco",
    "vertex_quantization_bits": 12,  # Invalid value
    "transform": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "lod_scale_multiplier": 1.0,
}

invalid_transform_data = {
    "@type": "neuroglancer_multilod_draco",
    "vertex_quantization_bits": 16,
    "transform": [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],  # 11 elements instead of 12
    "lod_scale_multiplier": 1.0,
}


def test_parse_unsharded_info():
    info = MultilodDracoInfo(**unsharded_data)
    assert info.type == "neuroglancer_multilod_draco"
    assert info.vertex_quantization_bits == 16
    assert len(info.transform) == 12
    assert info.lod_scale_multiplier == 1.0
    assert info.sharding is None
    assert info.segment_properties is None


def test_parse_sharded_info():
    info = MultilodDracoInfo(**sharded_data)
    assert info.type == "neuroglancer_multilod_draco"
    assert info.sharding is not None
    assert info.sharding.type == "neuroglancer_uint64_sharded_v1"
    assert info.sharding.preshift_bits == 0
    assert info.sharding.hash_function == "murmurhash3_x86_128"


def test_extra_fields_allowed():
    """Test that extra fields (not in spec) are allowed and don't cause errors."""
    info = MultilodDracoInfo(**data_with_extra_fields)
    assert info.type == "neuroglancer_multilod_draco"
    assert info.vertex_quantization_bits == 16
    assert len(info.transform) == 12
    assert info.lod_scale_multiplier == 1.0
    # Extra fields should be accessible via __pydantic_extra__
    assert hasattr(info, "__pydantic_extra__")
    extra_fields = info.__pydantic_extra__
    if extra_fields is not None:
        assert "mip" in extra_fields
        assert "chunk_size" in extra_fields
        assert "spatial_index" in extra_fields
        assert "unknown_field" in extra_fields


def test_invalid_type():
    with pytest.raises(ValidationError) as excinfo:
        MultilodDracoInfo(**invalid_type_data)
    assert "Input should be 'neuroglancer_multilod_draco'" in str(excinfo.value)


def test_invalid_vertex_quantization_bits():
    with pytest.raises(ValidationError) as excinfo:
        MultilodDracoInfo(**invalid_vq_bits_data)
    assert "Input should be 10 or 16" in str(excinfo.value)


def test_invalid_transform_length():
    with pytest.raises(ValidationError) as excinfo:
        MultilodDracoInfo(**invalid_transform_data)
    # Check for validation error about list length
    assert "List should have at least 12 items" in str(
        excinfo.value
    ) or "ensure this value has at least 12 items" in str(excinfo.value)

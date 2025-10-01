import pytest
from pydantic import ValidationError
from tissue_map_tools.data_model.sharded import ShardingSpecification

sharding_specification = {
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

minimal_sharding_specification = {
    "sharding": {
        "@type": "neuroglancer_uint64_sharded_v1",
        "preshift_bits": 0,
        "hash": "murmurhash3_x86_128",
        "minishard_bits": 10,
        "shard_bits": 10,
    },
}


def test_parse_full_sharding_specification():
    spec = ShardingSpecification(**sharding_specification["sharding"])
    assert spec.type == "neuroglancer_uint64_sharded_v1"
    assert spec.preshift_bits == 0
    assert spec.hash_function == "murmurhash3_x86_128"
    assert spec.minishard_bits == 10
    assert spec.shard_bits == 10
    assert spec.minishard_index_encoding == "gzip"
    assert spec.data_encoding == "raw"


def test_parse_minimal_sharding_specification():
    spec = ShardingSpecification(**minimal_sharding_specification["sharding"])
    assert spec.type == "neuroglancer_uint64_sharded_v1"
    assert spec.preshift_bits == 0
    assert spec.hash_function == "murmurhash3_x86_128"
    assert spec.minishard_bits == 10
    assert spec.shard_bits == 10
    assert spec.minishard_index_encoding is None
    assert spec.data_encoding is None


def test_invalid_hash_function():
    invalid = dict(minimal_sharding_specification["sharding"], hash="invalid_hash")
    with pytest.raises(ValidationError) as excinfo:
        ShardingSpecification(**invalid)
    assert "Input should be 'identity' or 'murmurhash3_x86_128'" in str(excinfo.value)


def test_invalid_type():
    invalid = dict(
        minimal_sharding_specification["sharding"], **{"@type": "wrong_type"}
    )
    with pytest.raises(ValidationError) as excinfo:
        ShardingSpecification(**invalid)
    assert "Input should be 'neuroglancer_uint64_sharded_v1'" in str(excinfo.value)

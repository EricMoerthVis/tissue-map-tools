import pytest
from pydantic import ValidationError
from tissue_map_tools.data_model.annotations import AnnotationInfo


def example_sharding():
    return {
        "@type": "neuroglancer_uint64_sharded_v1",
        "preshift_bits": 0,
        "hash": "murmurhash3_x86_128",
        "minishard_bits": 10,
        "shard_bits": 10,
        "minishard_index_encoding": "gzip",
        "data_encoding": "raw",
    }


def example_info():
    return {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"x": [1.0, "m"], "y": [1.0, "m"], "z": [40.0, "m"]},
        "lower_bound": [0, 0, 0],
        "upper_bound": [100, 100, 10],
        "annotation_type": "POINT",
        "properties": [
            {"id": "color", "type": "rgb"},
            {"id": "confidence", "type": "float32", "description": "Score"},
        ],
        "relationships": [
            {"id": "segment", "key": "segments", "sharding": example_sharding()}
        ],
        "by_id": {"key": "by_id", "sharding": example_sharding()},
        "spatial": [
            {
                "key": "spatial0",
                "sharding": example_sharding(),
                "grid_shape": [1, 1, 1],
                "chunk_size": [100.0, 100.0, 10.0],
                "limit": 1000,
            }
        ],
    }


def test_annotation_info_valid():
    info = AnnotationInfo(**example_info())
    assert info.type == "neuroglancer_annotations_v1"
    assert info.annotation_type == "POINT"
    assert info.dimensions["x"] == (1.0, "m")
    assert info.properties[0].id == "color"
    assert info.by_id.sharding is not None
    assert info.spatial[0].grid_shape == [1, 1, 1]


def test_missing_required_field():
    data = example_info()
    del data["annotation_type"]
    with pytest.raises(ValidationError):
        AnnotationInfo(**data)


def test_invalid_property_type():
    data = example_info()
    data["properties"][0]["type"] = "invalid_type"
    with pytest.raises(ValidationError):
        AnnotationInfo(**data)


def test_enum_labels_and_values():
    data = example_info()
    data["properties"][1]["enum_values"] = [0, 1]
    data["properties"][1]["enum_labels"] = ["low", "high"]
    info = AnnotationInfo(**data)
    assert info.properties[1].enum_labels == ["low", "high"]
    assert info.properties[1].enum_values == [0, 1]


def test_enum_values_only_for_numeric_types():
    data = example_info()
    data["properties"][0]["type"] = "rgb"
    data["properties"][0]["enum_values"] = [1, 2]
    data["properties"][0]["enum_labels"] = ["a", "b"]
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "'enum_values' is only allowed for numeric types" in str(excinfo.value)


def test_enum_labels_required_with_enum_values():
    data = example_info()
    data["properties"][1]["enum_values"] = [1, 2]
    data["properties"][1].pop("enum_labels", None)
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "'enum_labels' must be specified if 'enum_values' is specified" in str(
        excinfo.value
    )


def test_enum_labels_and_values_length_must_match():
    data = example_info()
    data["properties"][1]["enum_values"] = [1, 2]
    data["properties"][1]["enum_labels"] = ["a"]
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "'enum_labels' must have the same length as 'enum_values'" in str(
        excinfo.value
    )


def test_enum_values_required_with_enum_labels():
    data = example_info()
    data["properties"][1]["enum_labels"] = ["a", "b"]
    data["properties"][1].pop("enum_values", None)
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "'enum_values' must be specified if 'enum_labels' is specified" in str(
        excinfo.value
    )


def test_enum_labels_and_values_valid():
    data = example_info()
    data["properties"][1]["type"] = "uint8"
    data["properties"][1]["enum_values"] = [1, 2]
    data["properties"][1]["enum_labels"] = ["a", "b"]
    info = AnnotationInfo(**data)
    assert info.properties[1].enum_values == [1, 2]
    assert info.properties[1].enum_labels == ["a", "b"]


def test_invalid_dimension_scale():
    data = example_info()
    data["dimensions"]["x"] = [0, "m"]  # scale is not positive
    with pytest.raises(ValidationError):
        AnnotationInfo(**data)


def test_invalid_dimension_unit():
    data = example_info()
    data["dimensions"]["x"] = [1.0, 123]  # unit is not a string
    with pytest.raises(ValidationError):
        AnnotationInfo(**data)


def test_invalid_dimension_length():
    data = example_info()
    data["dimensions"]["x"] = [1.0]  # not a two-element list
    with pytest.raises(ValidationError):
        AnnotationInfo(**data)


def test_unsupported_dimension_unit():
    data = example_info()
    data["dimensions"]["x"] = [1.0, "foo"]  # unsupported unit
    with pytest.raises(ValidationError):
        AnnotationInfo(**data)


def test_valid_dimensions_xyz():
    data = example_info()
    data["dimensions"] = {"x": [1.0, "m"], "y": [2.0, "m"], "z": [3.0, "m"]}
    info = AnnotationInfo(**data)
    assert list(info.dimensions.keys()) == ["x", "y", "z"]
    assert info.dimensions["x"] == (1.0, "m")
    assert info.dimensions["y"] == (2.0, "m")
    assert info.dimensions["z"] == (3.0, "m")


def test_rank_mismatch_lower_upper_bound():
    data = example_info()
    data["lower_bound"] = [0, 0]
    data["upper_bound"] = [100, 100, 10]
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "Lower and upper bounds must have the same length." in str(excinfo.value)


def test_rank_mismatch_bounds_dimensions():
    data = example_info()
    data["lower_bound"] = [0, 0, 0, 0]
    data["upper_bound"] = [100, 100, 10, 10]
    # dimensions has only 3 keys
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert (
        "Length of lower and upper bounds must match the number of dimensions."
        in str(excinfo.value)
    )


def test_rank_property():
    data = example_info()
    info = AnnotationInfo(**data)
    assert info.rank == 3


def test_invalid_property_id_regexp():
    data = example_info()
    data["properties"][0]["id"] = "1invalid"  # does not start with a lowercase letter
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "Property 'id' must match the regular expression" in str(excinfo.value)

    data = example_info()
    data["properties"][0]["id"] = "InvalidUpper"  # starts with uppercase
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "Property 'id' must match the regular expression" in str(excinfo.value)

    data = example_info()
    data["properties"][0]["id"] = "valid_id1"  # valid
    info = AnnotationInfo(**data)
    assert info.properties[0].id == "valid_id1"


def test_key_absolute_path():
    data = example_info()
    data["relationships"][0]["key"] = "/absolute/path"
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "must not be an absolute path" in str(excinfo.value)


def test_key_empty():
    data = example_info()
    data["relationships"][0]["key"] = ""
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "must not be empty" in str(excinfo.value)


def test_key_empty_component():
    data = example_info()
    data["relationships"][0]["key"] = "foo//bar"
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "must not contain empty path components" in str(excinfo.value)


def test_key_just_dotdot():
    data = example_info()
    data["relationships"][0]["key"] = ".."
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "must not be just '..'" in str(excinfo.value)


def test_key_invalid_characters():
    data = example_info()
    data["relationships"][0]["key"] = "foo/bar$"
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "contains invalid characters" in str(excinfo.value)


def test_key_valid_relative():
    data = example_info()
    data["relationships"][0]["key"] = "foo/bar"
    info = AnnotationInfo(**data)
    assert info.relationships[0].key == "foo/bar"


def test_key_valid_dotdot():
    data = example_info()
    data["relationships"][0]["key"] = "foo/../bar"
    info = AnnotationInfo(**data)
    assert info.relationships[0].key == "foo/../bar"


def test_duplicate_relationship_ids():
    data = example_info()
    # Add a duplicate relationship id
    data["relationships"].append(
        {"id": "segment", "key": "other_segments", "sharding": example_sharding()}
    )
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "Duplicate relationship id found: segment" in str(excinfo.value)


def test_spatial_grid_shape_chunk_size_length_mismatch():
    data = example_info()
    # grid_shape too short
    data["spatial"][0]["grid_shape"] = [1, 1]
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "grid_shape length" in str(excinfo.value)

    data = example_info()
    # chunk_size too long
    data["spatial"][0]["chunk_size"] = [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "chunk_size length" in str(excinfo.value)


def test_spatial_grid_shape_chunk_size_positive():
    data = example_info()
    data["spatial"][0]["grid_shape"] = [1, 0, 1]  # zero is not positive
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "greater than 0" in str(excinfo.value)

    data = example_info()
    data["spatial"][0]["chunk_size"] = [1.0, -1.0, 1.0]  # negative is not positive
    with pytest.raises(ValidationError) as excinfo:
        AnnotationInfo(**data)
    assert "greater than 0" in str(excinfo.value)

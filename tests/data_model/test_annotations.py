import pytest
from typing import Any
import numpy as np
import tempfile
from pathlib import Path
from pydantic import ValidationError
from tissue_map_tools.data_model.annotations import (
    AnnotationInfo,
    decode_positions_and_properties_and_relationships_via_single_annotation,
    encode_positions_and_properties_and_relationships_via_single_annotation,
    write_annotation_id_index,
    read_annotation_id_index,
    write_related_object_id_index,
    read_related_object_id_index,
    write_spatial_index,
    read_spatial_index,
)


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
        "dimensions": {"x": [100.0, "m"], "y": [100.0, "m"], "z": [40.0, "m"]},
        "lower_bound": [10, 0, -10],
        "upper_bound": [110, 100, 30],
        "annotation_type": "POINT",
        "properties": [
            {"id": "color", "type": "rgb"},
            {"id": "confidence", "type": "float32", "description": "Score"},
            {
                "id": "cell_type",
                "type": "uint8",
                "description": "Cell type",
                "enum_values": [0, 1, 2],
                "enum_labels": ["A", "B", "C"],
            },
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
                "chunk_size": [100.0, 100.0, 40.0],
                "limit": 3,
            },
            {
                "key": "spatial1",
                "sharding": example_sharding(),
                "grid_shape": [2, 2, 1],
                "chunk_size": [50.0, 50.0, 40.0],
                "limit": 3,
            },
        ],
    }


def test_annotation_info_valid():
    info = AnnotationInfo(**example_info())
    assert info.type == "neuroglancer_annotations_v1"
    assert info.annotation_type == "POINT"
    assert info.dimensions["x"] == (100.0, "m")
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


def test_roundtrip_point():
    """Test roundtrip for a POINT annotation."""
    info = AnnotationInfo(**example_info())

    positions_values = [10.0, 20.0, 30.0]
    properties_values = {"color": [255, 128, 0], "confidence": 0.95, "cell_type": 1}
    relationships_values = {"segment": [1001, 1002]}

    encoded = encode_positions_and_properties_and_relationships_via_single_annotation(
        info=info,
        positions_values=positions_values,
        properties_values=properties_values,
        relationships_values=relationships_values,
    )
    decoded_pos, decoded_props, decoded_rels = (
        decode_positions_and_properties_and_relationships_via_single_annotation(
            info=info, data=encoded
        )
    )

    assert np.allclose(positions_values, decoded_pos)
    assert properties_values["color"] == decoded_props["color"]
    assert np.allclose(properties_values["confidence"], decoded_props["confidence"])
    assert properties_values["cell_type"] == decoded_props["cell_type"]
    assert relationships_values == decoded_rels


def test_roundtrip_line_missing_relationship():
    """Test roundtrip for a LINE annotation with a missing relationship."""
    line_info_dict = example_info()
    line_info_dict["annotation_type"] = "LINE"
    info = AnnotationInfo(**line_info_dict)

    positions_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    properties_values = {"color": [255, 128, 0], "confidence": 0.95, "cell_type": 2}
    relationships_values = {}  # Missing 'segment' relationship

    encoded = encode_positions_and_properties_and_relationships_via_single_annotation(
        info=info,
        positions_values=positions_values,
        properties_values=properties_values,
        relationships_values=relationships_values,
    )
    decoded_pos, decoded_props, decoded_rels = (
        decode_positions_and_properties_and_relationships_via_single_annotation(
            info=info, data=encoded
        )
    )

    assert np.allclose(positions_values, decoded_pos)
    assert properties_values["color"] == decoded_props["color"]
    assert np.allclose(properties_values["confidence"], decoded_props["confidence"])
    assert properties_values["cell_type"] == decoded_props["cell_type"]
    assert {"segment": []} == decoded_rels


def test_roundtrip_decode_encode():
    """Test that decoding and re-encoding gives the same result."""
    info = AnnotationInfo(**example_info())
    positions_values = [15.0, 25.0, 35.0]
    properties_values = {"color": [10, 20, 30], "confidence": 0.5, "cell_type": 0}
    relationships_values = {"segment": [2001]}

    original_encoded = (
        encode_positions_and_properties_and_relationships_via_single_annotation(
            info=info,
            positions_values=positions_values,
            properties_values=properties_values,
            relationships_values=relationships_values,
        )
    )

    decoded_pos, decoded_props, decoded_rels = (
        decode_positions_and_properties_and_relationships_via_single_annotation(
            info=info, data=original_encoded
        )
    )
    re_encoded = (
        encode_positions_and_properties_and_relationships_via_single_annotation(
            info=info,
            positions_values=decoded_pos,
            properties_values=decoded_props,
            relationships_values=decoded_rels,
        )
    )

    assert original_encoded == re_encoded


def test_write_read_annotation_id_index():
    """Test writing and reading the annotation ID index."""
    info = AnnotationInfo(**example_info())
    info.by_id.sharding = None

    annotations = {
        1: (
            [10.0, 20.0, 30.0],
            {"color": [255, 0, 0], "confidence": 0.9, "cell_type": 1},
            {"segment": [101, 102]},
        ),
        2: (
            [40.0, 50.0, 60.0],
            {"color": [0, 255, 0], "confidence": 0.8, "cell_type": 2},
            {"segment": [103]},
        ),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)

        # Test writing
        write_annotation_id_index(
            root_path=root_path, annotations=annotations, info=info
        )

        # Check that files were created
        index_dir = root_path / info.by_id.key
        assert (index_dir / "1").is_file()
        assert (index_dir / "2").is_file()

        # Test reading
        read_annotations = read_annotation_id_index(info=info, root_path=root_path)

        # Test consistency
        assert len(annotations) == len(read_annotations)
        for ann_id, original_data in annotations.items():
            read_data = read_annotations[ann_id]
            original_pos, original_props, original_rels = original_data
            read_pos, read_props, read_rels = read_data

            assert original_pos == read_pos
            assert original_props["color"] == read_props["color"]
            assert np.allclose(original_props["confidence"], read_props["confidence"])
            assert original_props["cell_type"] == read_props["cell_type"]
            assert original_rels == read_rels


def assert_dict_of_ids_positions_properties_equal(
    a: dict[str | int, list[tuple[int, list[float], dict[str, Any]]]],
    b: dict[str | int, list[tuple[int, list[float], dict[str, Any]]]],
) -> None:
    assert set(a.keys()) == set(b.keys())
    for key in a.keys():
        a_list = a[key]
        b_list = b[key]
        assert len(a_list) == len(b_list)
        for i in range(len(a_list)):
            a_id, a_pos, a_props = a_list[i]
            b_id, b_pos, b_props = b_list[i]
            assert a_id == b_id
            assert np.allclose(a_pos, b_pos)
            assert a_props.keys() == b_props.keys()
            for prop_key in a_props.keys():
                a_val = a_props[prop_key]
                b_val = b_props[prop_key]
                if isinstance(a_val, list):
                    assert a_val == b_val
                else:
                    assert np.allclose(a_val, b_val)


def test_write_read_related_object_id_index():
    """Test writing and reading the related object ID index."""
    info = AnnotationInfo(**example_info())
    # Test unsharded case
    for rel in info.relationships:
        rel.sharding = None

    annotations_by_object_id = {
        "segment": {
            101: [
                (
                    1,  # annotation_id
                    [10.0, 20.0, 30.0],  # positions
                    {
                        "color": [255, 0, 0],
                        "confidence": 0.9,
                        "cell_type": 1,
                    },  # properties
                )
            ],
            102: [
                (
                    1,
                    [10.0, 20.0, 30.0],
                    {"color": [255, 0, 0], "confidence": 0.9, "cell_type": 1},
                ),
                (
                    2,
                    [40.0, 50.0, 60.0],
                    {"color": [0, 255, 0], "confidence": 0.8, "cell_type": 2},
                ),
            ],
        }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)

        write_related_object_id_index(
            root_path=root_path,
            info=info,
            annotations_by_object_id=annotations_by_object_id,
        )

        rel_key = info.relationships[0].key
        index_dir = root_path / rel_key
        assert (index_dir / "101").is_file()
        assert (index_dir / "102").is_file()

        read_data = read_related_object_id_index(info=info, root_path=root_path)

        assert annotations_by_object_id.keys() == read_data.keys()
        for rel_id, original_rel_data in annotations_by_object_id.items():
            read_rel_data = read_data[rel_id]
            assert_dict_of_ids_positions_properties_equal(
                original_rel_data, read_rel_data
            )


def test_enum_in_example_info():
    """Test that enum properties are correctly parsed from the example info."""
    info = AnnotationInfo(**example_info())
    prop = next((p for p in info.properties if p.id == "cell_type"), None)
    assert prop is not None
    assert prop.enum_values == [0, 1, 2]
    assert prop.enum_labels == ["A", "B", "C"]


def test_write_read_spatial_index(tmp_path):
    """Test writing and reading the spatial index."""
    info = AnnotationInfo(**example_info())
    # Unsharded for this test
    for spatial in info.spatial:
        spatial.sharding = None

    # Helper to keep properties consistent with schema
    def props(color: list[int], confidence: float, cell_type: int) -> dict[str, Any]:
        return {
            "color": color,
            "confidence": confidence,
            "cell_type": cell_type,
        }

    # Construct annotations matching the schema and spatial chunking
    annotations = {
        "spatial0": {
            # single chunk covering the whole range
            "1_1_1": [
                (
                    1,
                    [11.0, 1.0, 0.0],
                    props([255, 0, 0], 0.9, 0),
                ),
                (
                    2,
                    [59.9, 49.9, 10.0],
                    props([0, 255, 0], 0.8, 1),
                ),
                (
                    3,
                    [109.9, 99.9, 29.9],
                    props([0, 0, 255], 0.7, 2),
                ),
            ]
        },
        "spatial1": {
            # left (x), bottom (y): x in [10,60), y in [0,50)
            "1_1_1": [
                (
                    4,
                    [20.0, 10.0, -5.0],
                    props([255, 128, 0], 0.85, 1),
                ),
                (
                    5,
                    [30.0, 40.0, 20.0],
                    props([128, 0, 255], 0.65, 0),
                ),
            ],
            # left (x), top (y): x in [10,60), y in [50,100)
            "1_2_1": [
                (
                    6,
                    [15.0, 60.0, 0.0],
                    props([0, 200, 200], 0.95, 2),
                ),
                (
                    7,
                    [45.0, 70.0, 10.0],
                    props([200, 0, 200], 0.55, 0),
                ),
                (
                    8,
                    [59.9, 99.0, 25.0],
                    props([50, 100, 150], 0.75, 1),
                ),
            ],
            # right (x), bottom (y): x in [60,110), y in [0,50)
            "2_1_1": [
                (
                    9,
                    [70.0, 10.0, -1.0],
                    props([20, 220, 60], 0.6, 2),
                ),
                (
                    10,
                    [80.0, 25.0, 5.0],
                    props([220, 20, 60], 0.7, 1),
                ),
                (
                    11,
                    [109.0, 49.9, 15.0],
                    props([60, 60, 220], 0.8, 0),
                ),
            ],
            # right (x), top (y): empty
            "2_2_1": [],
        },
    }

    # Write all spatial levels
    for spatial_level in info.spatial:
        write_spatial_index(
            info=info,
            root_path=tmp_path,
            spatial_key=spatial_level.key,
            annotations_by_spatial_chunk=annotations[spatial_level.key],
        )

    # Verify files exist for each chunk written
    for spatial_level in info.spatial:
        index_dir = tmp_path / spatial_level.key
        assert index_dir.is_dir()
        for chunk_name, ann_list in annotations[spatial_level.key].items():
            # Always create a file, even for empty chunk lists
            assert (index_dir / chunk_name).is_file()
            # Optional: ensure empty files only contain the number 0 as an uint64le
            if len(ann_list) == 0:
                with open(index_dir / chunk_name, "rb") as f:
                    data = f.read()
                    assert data == (0).to_bytes(8, byteorder="little")

    # Read back and verify content integrity
    for spatial_level in info.spatial:
        read_data_for_level = read_spatial_index(
            info=info,
            root_path=tmp_path,
            spatial_key=spatial_level.key,
        )

        original_level_data = annotations[spatial_level.key]
        assert_dict_of_ids_positions_properties_equal(
            original_level_data, read_data_for_level
        )

"""Lock the byte and hash conventions used by scientific evidence."""

import pytest

from identity_invariant_fas.data.serialization import digest, json_bytes


def test_canonical_json_bytes():
    assert json_bytes({"z": [None, True, 1.5], "a": "é"}) == (
        b'{\n  "a": "\\u00e9",\n  "z": [\n    null,\n    true,\n    1.5\n  ]\n}\n'
    )
    assert json_bytes({"z": 1, "a": 2}) == json_bytes({"a": 2, "z": 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(value):
    with pytest.raises(ValueError):
        json_bytes({"value": value})


def test_sha256_known_vector():
    assert digest(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

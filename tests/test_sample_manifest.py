"""Sample metadata validation using synthetic CSVs, without raw datasets."""

import csv
from dataclasses import asdict
from pathlib import PurePosixPath

import pytest

from identity_invariant_fas.constants import ATTACK_LABEL, BONA_FIDE_LABEL
from identity_invariant_fas.data.sample_manifest import (
    REQUIRED_COLUMNS,
    UNASSIGNED_SPLIT,
    SampleRecord,
    load_sample_manifest,
)


def make_row(**changes):
    row = dict(
        sample_id="sample-001",
        filepath="ClientFace/0001/image.jpg",
        subject_id="0001",
        class_label="0",
        attack_type="",
        split="train",
        preprocessing_version="",
    )
    row.update(changes)
    return row


def write_csv(tmp_path, rows, columns=REQUIRED_COLUMNS, encoding="utf-8"):
    path = tmp_path / "samples.csv"
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_valid_manifest_preserves_source_metadata_and_order(tmp_path, encoding):
    rows = [
        make_row(sample_id="z", extra="ignored"),
        make_row(
            sample_id="a", filepath="ImposterFace/0001/other.jpg",
            class_label="1", split="test", preprocessing_version="producer-v1",
        ),
        make_row(sample_id="u", split=UNASSIGNED_SPLIT, attack_type="source-value"),
    ]
    path = write_csv(tmp_path, rows, (*reversed(REQUIRED_COLUMNS), "extra"), encoding)
    records = load_sample_manifest(path)
    assert all(isinstance(record, SampleRecord) for record in records)
    assert [record.sample_id for record in records] == ["z", "a", "u"]
    assert [record.class_label for record in records] == [BONA_FIDE_LABEL, ATTACK_LABEL, 0]
    assert records[0].subject_id == "0001"
    assert records[0].attack_type == ""
    assert records[0].preprocessing_version == ""
    assert records[1].preprocessing_version == "producer-v1"
    assert [record.split for record in records] == ["train", "test", "unassigned"]
    assert asdict(records[2])["attack_type"] == "source-value"


@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_missing_required_columns(tmp_path, missing):
    columns = [name for name in REQUIRED_COLUMNS if name != missing]
    path = write_csv(tmp_path, [], columns)
    with pytest.raises(ValueError, match=f"Missing required columns: {missing}"):
        load_sample_manifest(path)


def test_duplicate_sample_ids_across_source_splits(tmp_path):
    path = write_csv(tmp_path, [make_row(), make_row(split="test", subject_id="0002")])
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_sample_manifest(path)


@pytest.mark.parametrize("label", ["", "-1", "2", "real", "attack", "0.0", "True", "01"])
def test_invalid_class_labels(tmp_path, label):
    path = write_csv(tmp_path, [make_row(class_label=label)])
    with pytest.raises(ValueError, match="invalid class_label"):
        load_sample_manifest(path)


@pytest.mark.parametrize("field", ["sample_id", "filepath", "subject_id", "split"])
@pytest.mark.parametrize("empty", ["", " \t "])
def test_empty_required_values(tmp_path, field, empty):
    path = write_csv(tmp_path, [make_row(**{field: empty})])
    with pytest.raises(ValueError, match=f"{field} must be non-empty"):
        load_sample_manifest(path)


@pytest.mark.parametrize("field", ["sample_id", "subject_id", "split"])
@pytest.mark.parametrize("padding", [" {}", "{} ", "\t{}", "{}\t"])
def test_identifiers_reject_surrounding_whitespace(tmp_path, field, padding):
    value = padding.format(make_row()[field])
    path = write_csv(tmp_path, [make_row(**{field: value})])
    with pytest.raises(ValueError, match=f"{field} must not have leading or trailing whitespace"):
        load_sample_manifest(path)


def test_filepath_spaces_are_preserved(tmp_path):
    filepath = " ClientFace/0001/image.jpg "
    path = write_csv(tmp_path, [make_row(filepath=filepath)])
    assert load_sample_manifest(path)[0].filepath == filepath


@pytest.mark.parametrize("filepath", [r"ClientFace\0001\image.jpg", "ClientFace/0001/image.jpg"])
def test_paths_are_portable_and_do_not_depend_on_working_directory(tmp_path, monkeypatch, filepath):
    path = write_csv(tmp_path, [make_row(filepath=filepath)])
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    record = load_sample_manifest(path)[0]
    assert record.filepath == "ClientFace/0001/image.jpg"
    assert tmp_path.joinpath(*PurePosixPath(record.filepath).parts) == (
        tmp_path / "ClientFace" / "0001" / "image.jpg"
    )


@pytest.mark.parametrize("filepath", [
    "/image.jpg", "C:/image.jpg", r"C:image.jpg", r"\\server\share\image.jpg",
    "../image.jpg", r"folder\..\image.jpg", ".", "./", "a:b.jpg", "a\x00.jpg",
])
def test_nonportable_paths_are_rejected(tmp_path, filepath):
    path = write_csv(tmp_path, [make_row(filepath=filepath)])
    with pytest.raises(ValueError, match="portable dataset-relative path"):
        load_sample_manifest(path)


def test_header_only_manifest(tmp_path):
    assert load_sample_manifest(write_csv(tmp_path, [])) == []


def test_duplicate_headers(tmp_path):
    path = write_csv(tmp_path, [], (*REQUIRED_COLUMNS, "sample_id"))
    with pytest.raises(ValueError, match="Duplicate column names"):
        load_sample_manifest(path)


@pytest.mark.parametrize("line", ["a,b,c,0", "a,b,c,0,,train,,extra"])
def test_malformed_row_width(tmp_path, line):
    path = write_csv(tmp_path, [])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with pytest.raises(ValueError, match="row length does not match header"):
        load_sample_manifest(path)

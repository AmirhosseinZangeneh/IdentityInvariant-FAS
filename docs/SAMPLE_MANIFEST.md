# Sample metadata manifests

`identity_invariant_fas.data.sample_manifest.load_sample_manifest` reads a
UTF-8 CSV (optional BOM) into an ordered list of immutable `SampleRecord`
objects. This is sample-level source metadata for future protocol generation
and PyTorch adapters. It does not load images, run preprocessing, assign
training identity indices, verify identity provenance, or establish a valid
evaluation protocol. Canonical subject-disjoint fold generation and final
train/validation/test assignment will be implemented separately.

All seven columns must be present; column order is flexible. Extra columns
are ignored. Duplicate headers, malformed row widths, duplicate sample IDs,
and missing required columns are rejected. A header-only manifest is allowed.

| Field | Meaning |
| --- | --- |
| `sample_id` | Non-empty, unique opaque string supplied by the producer; no ID generation is performed. |
| `filepath` | Non-empty path relative to an explicitly chosen dataset root, independent of CSV location or working directory. |
| `subject_id` | Non-empty opaque source identity string; leading zeros are preserved. |
| `class_label` | Exactly `0` (bona fide) or `1` (attack), from the existing shared constants. Returned as an integer. |
| `attack_type` | Source value verbatim; empty means unspecified, as in the current NUAA loader. No attack category is inferred from class labels. |
| `split` | Existing dataset/source split only, or literal `unassigned` if none exists. Empty values are rejected. For NUAA, preserve official `train`/`test` provenance when known. |
| `preprocessing_version` | Producer-supplied preprocessing identifier; empty means undocumented/unknown. No version is inferred or preprocessing performed. |

Strings are preserved (apart from path normalization); whitespace-only IDs
and paths are rejected. `sample_id`, `subject_id`, and `split` also reject
leading or trailing whitespace; valid values are preserved without stripping.
This restriction does not apply to filepath spaces.
Filepaths accept either slash style and normalize to
forward slashes without filesystem access. Absolute paths, drive-qualified
paths, parent traversal (`..`), colons, NULs, and paths resolving lexically to
`.` are rejected. Producers should use portable dataset-relative names.
The loader preserves row order and does not require image files to exist.

```python
from pathlib import Path, PurePosixPath
from identity_invariant_fas.data.sample_manifest import load_sample_manifest

records = load_sample_manifest("metadata/nuaa_samples.csv")
dataset_root = Path("datasets/NUAA")
image_path = dataset_root.joinpath(*PurePosixPath(records[0].filepath).parts)
```

The caller chooses the root explicitly. Future dataset adapters can index
the returned list and consume record attributes, or use `dataclasses.asdict`.
The parser preserves source split strings without imposing a dataset-specific
vocabulary; producers are responsible for their provenance. `split` must not
be populated with the future canonical fold assignment.

## Compatibility and unresolved NUAA semantics

The existing `data.manifest` API (`FASSample`, `read_manifest`, `write_manifest`,
`ManifestDataset`) and `load_nuaa_samples` remain unchanged. Their CSV schema
is different and is not automatically converted or detected. Existing
experiments continue using those APIs.

In the NUAA loader, `FASSample.subject` is the listed path's parent-folder
name. The new `subject_id` denotes that source identity concept, **not** the
legacy integer `FASSample.subject_id` assigned for training. The loader does
not establish whether matching folder identifiers across ClientFace and
ImposterFace denote the same person. Identity provenance needs verification
before canonical splitting. NUAA `attack_type` is currently empty for both
classes, and no preprocessing version is recorded; neither is guessed here.

Existing k-fold configs use the official test partition as their source pool;
the official entry point uses the official train/test lists, while
`scripts/train_nuaa.py` pools both. Source `split` therefore does not describe
an experiment's assigned fold. Dataset root conventions also differ:
several configs use `datasets/NUAA`, while the official config and older
documentation use `datasets`. This layer makes no choice between them and
does not generate a NUAA manifest or touch raw data.

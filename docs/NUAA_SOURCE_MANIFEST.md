# NUAA source manifest and provenance audit

This preparation step creates the canonical seven-column sample metadata CSV
from all four official source lists. It does **not** create canonical folds,
establish human-subject correspondence, or integrate with training. The
existing NUAA loader, split utilities, and experiments remain unchanged.

## Generation

From the repository root, in an environment with the project installed:

```powershell
python -m identity_invariant_fas.data.nuaa_manifest --root datasets/NUAA --manifest manifests/generated/nuaa_source.csv --audit docs/audit/nuaa_source_audit.json
```

The equivalent API is:

```python
from identity_invariant_fas.data.nuaa_manifest import (
    build_nuaa_source_manifest,
    generate_nuaa_source_manifest,
    validate_nuaa_source_manifest,
)

records, audit = build_nuaa_source_manifest("datasets/NUAA")
validate_nuaa_source_manifest(records, "datasets/NUAA")
generate_nuaa_source_manifest(
    "datasets/NUAA",
    "manifests/generated/nuaa_source.csv",
    "docs/audit/nuaa_source_audit.json",
)
```

In the local development environment, `env_cuda/Scripts/python.exe` was used.
The CSV is generated in the existing ignored directory; the compact JSON audit
is reviewable in Git. Outputs must be distinct and outside the raw dataset root.
Generation writes only these two outputs and does not decode or modify images.

## Deterministic metadata contract

- Reuses the legacy loader's source-list specifications, label constants,
  annotation parser, and folder extraction. Every nonblank source-list entry
  must have an unambiguous `subject/filename` path and an existing file.
- `filepath` is `ClientFace/<folder>/<filename>` or
  `ImposterFace/<folder>/<filename>`, relative to the supplied root, with `/`
  separators. No absolute directory is stored.
- `sample_id` is exactly `NUAA:` followed by that filepath. This injective
  naming rule distinguishes identical filenames in different class folders,
  uses no random state or row number, and is independent of root location.
  Renaming a relative path changes the sample ID. IDs are case-sensitive.
- `subject_id` preserves the raw folder token, including leading zeros.
  **It is not a verified cross-class human identity.** Matching tokens are
  recorded as an observed naming overlap only; no mapping is manufactured.
- `class_label` retains `0` for ClientFace/bona fide and `1` for
  ImposterFace/attack. `split` retains the official source `train` or `test`.
  No source assignment is missing for these four lists.
- `attack_type` and `preprocessing_version` remain empty. The readme describes
  face-detector output, but supplies no preprocessing version identifier or
  per-sample attack subtype. Landmark coordinates are parsed like the loader;
  this schema neither stores them nor applies them to images.
- Rows are sorted lexicographically by relative filepath. CSV and JSON use
  UTF-8 and LF newlines, with no timestamps or absolute paths in the payloads.

Duplicate source paths (including repeats across source splits) fail rather
than being deduplicated or assigned an invented split. Manifest validation
requires unique sample IDs and filepaths, exact source coverage, and exact
metadata agreement, including labels and splits. Generation repeats indexing
and auditing, checks equality, and round-trips the CSV through the existing
canonical parser. Synthetic tests also compare repeated output bytes and
output bytes after relocating the dataset.

The audit includes SHA-256 digests of the four list files, the local readme
when present, the CSV, and sorted image-path inventories. List hashes change
if list bytes change, even when the sorted CSV remains identical. Inventory
hashes cover names, **not image contents**. File existence does not establish
image validity, content uniqueness, or dataset authenticity. Unlisted folders
and images (`.jpg`, `.jpeg`, `.png`, `.bmp`, case-insensitive extensions) are
reported rather than silently added to the manifest.

## Local provenance findings

Observed on 2026-09-21; exact source fingerprints and enumerated IDs are in
[the generated audit](audit/nuaa_source_audit.json).

| Official source | Bona fide | Attack | Total |
| --- | ---: | ---: | ---: |
| Train | 1,743 | 1,748 | 3,491 |
| Test | 3,362 | 5,761 | 9,123 |
| Total | 5,105 | 7,509 | 12,614 |

All 12,614 listed paths exist and map uniquely to manifest records. On this
copy, the image inventory exactly matches the lists: no unlisted images or
folders were observed. Full repeated generation and metadata validation passed.

- ClientFace: 15 folder IDs, `0001` through `0015`.
- ImposterFace: 15 folder IDs, `0001` through `0012`, plus `0014`, `0015`, `0016`.
- Intersection: 14 tokens (`0001` through `0012`, `0014`, `0015`).
- Client-only: `0013`. Imposter-only: `0016`. Union: 16 tokens.
- Within ClientFace, source train/test folder overlap is `0004`, `0006`, `0007`.
  Within ImposterFace it is `0001` through `0009`. Official source assignment
  therefore is not even folder-disjoint and must not be relabeled as such.

The bundled `datasets/NUAA/readme.txt` states totals of 5,105 client and 7,509
imposter images, matching the observed lists and inventory. It says each
subject's images are stored in a separate directory and describes filename
IDs in the range `0001~0016`. The observed 16-token union is consistent with
that range, but neither class has 16 folders. The readme does not explicitly
state cross-class identity equivalence or explain the asymmetric IDs.

The repository's `configs/nuaa_identity_invariant.yaml` sets `num_subjects: 16`;
that agrees numerically with the token union but is a model configuration,
not provenance evidence. `docs/audit/pre_protocol_lock_state.md` explicitly
leaves exact protocol correspondence, reproducibility, and preprocessing
consistency for verification and supplies no independently verified count.
The existing loader and split utilities group raw tokens without verifying
human correspondence. Their behavior is not independent identity evidence.

## Unresolved before human-subject folds

The available local metadata supports folder-derived identifiers but does
**not sufficiently establish a human mapping across bona fide and attack
folders**. Equal folder IDs have not been verified to represent the same
person. Neither `0013` nor `0016` can be remapped, merged, or excluded based
on this evidence. The reason for the mismatch remains unknown. The union
count must not be reported as a verified count of 16 human subjects.

A documented cross-class mapping and explanation of these discrepancies are
needed before canonical human-subject-disjoint fold generation. Attack subtype
and exact preprocessing provenance/version also remain undocumented. This
audit is confined to the locally available lists, readme, directory names,
and repository documentation; no external provenance source was consulted.

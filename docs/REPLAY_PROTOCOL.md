# Replay-Attack metadata and protocol preparation

This layer prepares explicit source metadata for later experiments. It does
not train models, extract frames, run identity probes, or change existing
experiment configurations. No raw directory scanner or filename-based client
parser is provided: the repository has no verified raw naming/mapping artifact.

## Official evidence and scope

Idiap describes 50 clients in disjoint PAD cohorts: train (15), devel (15),
and test (20). Train is for PAD fitting, devel for threshold selection, and
test for reporting. Enrollment is separate. The biometric protocol uses
enrollment to build client models and separate real accesses for evaluation,
without mixing development and test cohorts. Conditions include controlled
and adverse lighting and fixed or hand-held attack support.
[Source: Idiap Replay-Attack documentation](https://www.idiap.ch/fr/recherche/donnees/replayattack/index_html?set_language=fr).

The official Bob API exposes explicit client IDs/cohorts, unique file IDs and
paths, real-access purpose (`authenticate`/`enroll`), take, light, and attack
device/support/sample metadata. These supply a semantic reference, not a
locally verified recording inventory.
[Source: bob.db.replay 3.0.9 API](https://www.idiap.ch/software/bob/docs/bob/bob.db.replay/v3.0.9/py_api.html).

The adapter targets the original Replay-Attack PAD cohorts with all attack
types. It does not implement the family of attack-specific or support-specific
benchmark subsets or the later mask extension. No client roster is inferred
from the published cardinalities. A cardinality match alone is not evidence
that a supplied mapping is authentic or its video inventory complete.

## Explicit metadata contract

`identity_invariant_fas.data.replay_protocol` provides `ReplayClient`,
`ReplayRecording`, and `ReplayProtocol`. `load_replay_metadata` reads our
normalized UTF-8 JSON format, not an unmodified Bob database. An exporter or
reviewed transcription from the licensed official catalog is required before
real-data use. Unknown JSON fields and duplicate keys are rejected.

Top-level keys are exactly:

| Key | Meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `dataset` | Exactly `Replay-Attack`. |
| `source` | Logical source-catalog name/version or URL; use no absolute local paths. |
| `source_sha256` | Lowercase SHA-256 of the original mapping/catalog artifact. |
| `clients` | Explicit client roster: objects with `client_id` and `pad_cohort`. |
| `recordings` | One object per original source recording, with the fields below. |

Client IDs are canonical strings exported from official client metadata,
never guessed from filename similarity. Preserve the official mapping
consistently; e.g. a Bob integer client key should have one decimal string
representation throughout the export. The client roster's `pad_cohort` is
one of `train`, `devel`, `test`, even for clients' enrollment recordings.

| Recording field | Contract |
| --- | --- |
| `source_recording_id` | Required canonical string key from the source catalog. |
| `filepath` | Required dataset-relative video path from source metadata; slash styles normalize to `/`. No absolute paths, `..`, colons, or NULs. |
| `client_id` | Required explicit reference to the client roster. |
| `class_label` | Required integer: existing `0` bona fide, `1` attack. |
| `purpose` | Required repository enum `pad` or `enroll`. |
| `pad_partition` | Required key: `train`/`devel`/`test` for PAD; JSON `null` for enrollment. |
| `light` | Optional `controlled` or `adverse`. |
| `attack_device` | Optional presentation medium: `print`, `mobile`, `highdef`. |
| `attack_support` | Optional `fixed` or `hand`. |
| `sample_type` | Optional original attack material: `photo` or `video`. |
| `sample_device` | Optional original attack sampling device: `mobile` or `highdef`; not the presentation device. |
| `take` | Optional source take number as a nonnegative integer; not a session key. |
| `session_id`, `session_source` | Optional paired fields for a separately verified recording-session key and its evidence. Both default to `null`. |

All optional fields default to `null`, never inferred categories. Bona-fide
recordings cannot carry attack-specific fields. No preprocessing version is
invented; this is original recording metadata, not a processed frame manifest.
Camera/lighting/medium/take/purpose are not used to manufacture sessions.

Exporting from official semantic metadata should map real-access
`purpose=enroll` to our `purpose=enroll`, class 0, and null PAD partition.
Real-access `purpose=authenticate` and attack objects map to `purpose=pad`,
classes 0 and 1 respectively, and the explicit client's official cohort.
These are field mappings, not filename rules. A producer must reconcile
every recording against the actual authoritative catalog; no automatic Bob
dependency, exporter, or raw-name parser is installed in this change.

## PAD validation and RQ3

`validate_replay_protocol` checks provenance fields, the explicit roster,
client mappings, role/class/partition consistency, and recording uniqueness.
Duplicate source IDs or relative paths fail even if metadata otherwise agree.
An overlap or conflicting client assignment fails rather than being repaired.

Full validation is the default: expected client counts must be 15/15/20, and
each roster client must have both bona-fide and attack PAD recordings. These
constants belong only to the Replay-specific module. This checks cohort and
class coverage, not the full official recording count or all attack subtypes.
`require_complete=False` / CLI `--allow-partial` supports synthetic tests and
partial inspection; it never disables overlap, duplicate, or conflict checks,
and its report explicitly says full cardinalities were not checked.

`replay_pad_partitions(protocol)` returns only PAD recordings under exactly
`train`, `devel`, and `test`. Enrollment is excluded by construction. This is
the future subject-disjoint RQ3 preparation path after validating the input
against official metadata and files. No new folds, thresholds, hyperparameters,
or training integration are introduced.

## Enrollment and RQ1

`prepare_replay_identity_probe(protocol, cohort)` selects one complete client
cohort. Probe training contains only its enrollment recordings; probe
evaluation contains only its separate bona-fide PAD real-access recordings.
Both identity-class sets must equal the cohort's entire roster. Missing
enrollment or real-access coverage fails; attacks can never be enrollment.
The returned `ReplayIdentityPreparation` contains aligned recording references,
`ProbeObservation` objects, and a validated `ProbeSplit`, with no features.

Group IDs are `Replay-Attack:recording:` followed by the percent-encoded
official source recording ID. Encoding is injective and independent of local
root paths, list order, and filename spelling. Video-level sample IDs append
`:video`. For future frames, pass `frame_indices={video_id: [actual_indices]}`;
the mapping must cover exactly the selected enrollment/access recordings.
Sample IDs append `:frame:<index>`, while all frames retain their source group.
No frame files are generated or decoded, and frame existence/indices must be
verified by the later extraction stage. Duplicate/empty frame selections fail.

`identity_id` is the supplied verified Replay client key. No identity is
estimated from pixels or names. `session_id` remains null unless an explicit
session key and evidence were provided. Requesting session-disjoint mode with
missing metadata or overlapping sessions fails through the existing
[group-aware probe validator](IDENTITY_PROBING.md). Enrollment/access separation
is not itself a claim of session-disjointness.

A PAD encoder may never have seen devel/test clients, while the closed-set
identity classifier must train on those same clients' enrollment observations.
This proposed RQ1 classification preparation is distinct from Idiap's biometric
verification scoring protocol. It does not silently substitute a new published
benchmark, mix cohorts, or fall back to frame-level cross-validation.

## Inspection commands and APIs

After producing an audited metadata export (the paths below are placeholders):

```powershell
python -m identity_invariant_fas.data.replay_protocol --metadata metadata/replay_official.json
python -m identity_invariant_fas.data.replay_protocol --metadata metadata/replay_official.json --root datasets/ReplayAttack --source-artifact metadata/original_official_catalog
```

The first command checks metadata only. The second additionally checks every
referenced file and the catalog fingerprint, without decoding or modifying
media. JSON is printed to stdout; no output files are written automatically.
The report contains sorted cohort sets, class counts, enrollment coverage,
validation flags, and a normalized metadata digest, without root paths or
timestamps. Repeated input and relocated raw roots give identical reports.

```python
from identity_invariant_fas.data.replay_protocol import (
    load_replay_metadata, inspect_replay_protocol,
    replay_pad_partitions, prepare_replay_identity_probe,
)

protocol = load_replay_metadata("metadata/replay_official.json")
report = inspect_replay_protocol(protocol, root="datasets/ReplayAttack")
pad = replay_pad_partitions(protocol)
plan = prepare_replay_identity_probe(protocol, "test")
```

The old `load_replay_attack_manifest` generic CSV API remains unchanged for
compatibility. It does not provide these guarantees and is not silently
converted. `FASSample` and `SampleRecord` are also unchanged; neither can
adequately distinguish all of these roles and conditions on its own.

## Limits and next experiment gate

The source reference and digest are producer assertions; matching an artifact
hash does not authenticate its origin or prove the transcription correct.
The importer checks declared metadata, not identity truth. Referenced-file
checks detect missing paths and resolved path aliases; they do not establish
media validity, content-level duplication, or inventory completeness. Raw-file
inventory reconciliation, official mapping verification, and session evidence
remain external audit requirements.

The local Replay directory is empty; see the
[availability record](audit/replay_metadata_availability.md). No real manifest
or dataset audit was fabricated. Before the first controlled RQ1 experiment,
obtain licensed recordings and the official catalog, implement/review its
explicit export, reconcile all client/recording assignments and enrollment
coverage, verify source hashes and files, then define reproducible frame
sampling/preprocessing with source-group IDs. Only after the chosen cohort's
enrollment/access plan passes validation should encoder provenance and frozen
feature extraction be approved as a separate experiment task.

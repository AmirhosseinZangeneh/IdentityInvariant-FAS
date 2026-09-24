# MSU-MFSD human review — response closure (not preprocessing approval)

Reviewer: Amirhossein Zangeneh
Scope: 64 unique reviewed cases (56 training domain comparisons, 8 historical original crops).
Source response SHA-256: `c495547a2db8d58e0225e3cc1c72f311bc3b173f05ac6dfb32d9041d1d7535e7`
Final response SHA-256: `385fe217e56e95d172d0b91d7f0237e1a9462a77073ae3297f34a01adf89afa4`

## Reviewer-authorized corrections

| Case | Field | Previous | Final |
|---|---|---|---|
| `MSU-MFSD:scene01/real/real_client008_android_SD_scene01.mp4:annotation:298` | `half_open_endpoint_plausible` | `no` | `uncertain` |
| `MSU-MFSD:scene01/real/real_client008_android_SD_scene01.mp4:annotation:298` | `orientation_acceptable` | `no` | `yes` |
| `original-crop:MSU-MFSD:scene01/real/real_client002_android_SD_scene01.mp4:frame:0:prep:9e3335e82b760e18aec7daa048e74cc401e611f4c8c75871d0216b563d153b25` | `face_box_acceptable` | `None` | `uncertain` |

## Visual review field counts

| Field | Yes | Uncertain | No | Unset |
|---|---:|---:|---:|---:|
| `orientation_acceptable` | 57 | 7 | 0 | 0 |
| `eye_placement_acceptable` | 64 | 0 | 0 | 0 |
| `face_box_acceptable` | 50 | 13 | 1 | 0 |
| `half_open_endpoint_plausible` | 57 | 7 | 0 | 0 |

## Temporal preference — comparison cases only

| Preference | Count |
|---|---:|
| `decframes_export_i` | 30 |
| `native_i` | 6 |
| `native_i_minus_1` | 6 |
| `native_i_plus_1` | 7 |
| `indistinguishable` | 7 |

Temporal responses entered for the eight historical original-crop forms remain untouched in the raw response JSON, but are **not applicable** to four-domain comparisons and are excluded from these totals.
These are reviewer choices, not proof that any selected frame domain generated the PittPatt annotations. A selected domain can also share identical pixels with another candidate.

## Preserved limitations and blockers

- At least one case (real client 008 Android annotation 298) retains `face_box_acceptable=no` because the chin is truncated; other cases retain uncertain geometry. Do not silently average them away.
- `half_open_endpoint_plausible` describes a visual check; it does not prove a coordinate endpoint convention.
- The PittPatt annotation-generation stream/timebase and universal annotation-to-native/export correspondence remain unresolved (P4).
- Eight previously identified codec-error recordings still require a documented handling policy.
- Final JSON preserves `fidelity_status=manual_review_pending` and `experiment_ready=false`; no training, official-test-driven policy choice or preprocessing authorization is given.

## Ingestion and provenance validation

The final response bytes were independently hashed and match the final SHA-256
above. The earlier source digest and reviewer-authorized correction table are
retained as supplied historical context; this ingestion received and validates
only the final response, and does not reapply or independently certify those
earlier edits. No response, comment or source JSON byte was changed. A targeted
Git attribute disables line-ending conversion for this source file so future
checkouts preserve its exact-byte digest.

The deterministic [derived audit](audit/msu_human_review_validation.json) binds
the final response to the review pack, blank template, historical crop metadata,
raw protocol lock, policy decision and prior validation from commit `cee0d0d`.
It records committed-blob SHA-256 and canonical-JSON SHA-256 separately; working
evidence must match the committed files, allowing Git LF/CRLF checkout conversion
only for those existing evidence files. Source-response hashing never normalizes
bytes. No decoding, exports, image generation or raw-media access is required.

All 64 unique case IDs exactly match the committed template and pack: 56 training
comparisons and eight historical crops, with no official-test cases. All required
geometry fields are populated with valid values. Counts above were calculated
from the JSON, not substituted as validation inputs. The audit preserves each
raw case observation/comment, identifies the one negative face-box case, and
lists all **14 cases with at least one uncertain geometry field** (including that
negative face-box case). Original-crop temporal values remain in observations;
derived temporal analysis marks them `not_applicable` and excludes all eight.

`human_observation_status=recorded_and_validated` describes completion of this
ingestion. It is distinct from fidelity approval: the response's
`fidelity_status=manual_review_pending` is retained without implying the human
never performed the review. The historical policy artifact is unchanged;
`policy_state=P4`, `frozen_policy=null`, `experiment_ready=false`. No preference
winner or frame-domain correction is inferred. These 64 cases do not establish
release-wide fidelity, resolve the eight codec errors, or prove PittPatt
correspondence. Geometry concerns and missing source correspondence evidence
remain explicit blockers before any preprocessing approval or experiment.

Reproduce ingestion into a **new** output file (existing outputs are rejected):

```powershell
python -m identity_invariant_fas.data.msu_human_review --response docs/audit/msu_human_review_responses.json --expected-sha256 385fe217e56e95d172d0b91d7f0237e1a9462a77073ae3297f34a01adf89afa4 --evidence-ref cee0d0d --output docs/audit/msu_human_review_validation.json
```

The read-only `ingest_human_review(repository, response_path,
expected_sha256=..., evidence_ref=...)` API returns the same audit without writing
a file. The lower-level `validate_human_review(...)` accepts already-trusted
metadata for synthetic tests; repository ingestion must use the commit-binding
API. Missing provenance, digest conflicts, duplicate/missing/unexpected cases,
invalid/unfilled fields, identity changes or readiness promotion fail explicitly.

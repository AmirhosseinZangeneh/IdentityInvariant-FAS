# MSU-MFSD public-release provenance and protocol lock

This is a metadata/inventory lock, not an experiment result. The local licensed
release was inspected without decoding, renaming, regenerating, or modifying
raw files. Models, preprocessing, training, metrics, and configurations remain
unchanged. The deterministic [audit artifact](audit/msu_protocol_lock.json)
contains the exact source fingerprints, inventories, client sets, and readiness
checks. The full generated recording table is in the ignored
`manifests/generated/msu_recordings.json`.

## Identity evidence and normalization

The bundled `datasets/MSU/README.txt` explicitly defines the real and attack
filename grammars, their `clientID` field, the 35-client public release, and
the supplied train/test subject lists. Real and attack recordings name the
same documented client identifier. This supports a human-client mapping via
release metadata, rather than filename similarity or visual identity inference.
Actual face content has not been independently verified.

The lists use numeric tokens such as `02`; filenames use `client002`.
The supplied `DecFrames.m`, function `insublist`, parses the three filename
digits as an integer and compares against the numeric subject list. The new
`read_msu_subjects` follows that evidence: numeric ID 2 becomes canonical
string `002`. Duplicates, including padding aliases such as `02` and `002`
within one list, fail. Train/test overlap after normalization also fails.
The 001–055 filename range is not interpreted as 55 released subjects.

The local README, MATLAB script, and two subject lists are fingerprinted in
the lock. `structure.txt` is a local directory-tree snapshot, not an independent
official recording manifest; completeness is checked against the documented
eight acquisition slots per listed subject.

## Observed local inventory

| Official partition | Clients | Bona fide | Attack | Videos |
| --- | ---: | ---: | ---: | ---: |
| Train | 15 | 30 | 90 | 120 |
| Test | 20 | 40 | 120 | 160 |
| Total | 35 | 70 | 210 | 280 |

Canonical train clients:
`002, 003, 005, 006, 007, 008, 009, 011, 012, 021, 022, 034, 053, 054, 055`.

Canonical test clients:
`001, 013, 014, 023, 024, 026, 028, 029, 030, 032, 033, 035, 036, 037, 039, 042, 048, 049, 050, 051`.

These sets are disjoint and their union exactly equals the discovered client
set. Each client has eight videos: two bona-fide and six attacks. Each of
`ipad_video`, `iphone_video`, and `printed_photo` has 70 videos overall
(30 train / 40 test), one per capture device per client.

The capture-device tokens identify Google Nexus 5 front-facing (`android`)
and MacBook Air built-in (`laptop`) cameras: 140 videos each, respectively
`.mp4` and `.mov`. Each has 60 train and 80 test videos. All 280 names carry
`SD`; this is a source metadata token, not a decoded resolution measurement.
Presentation media are separately represented as iPad Air, iPhone 5S, and
printed paper. The README mentions cameras used to create attack material,
but the source camera cannot be mapped per recording from the filename.

There are 280 matching PittPatt annotation files. The audit found no missing
expected acquisition slots, missing annotations, orphan annotations, unlisted
recordings, unexpected clients, duplicate paths/recording IDs, or physical
file aliases. Total video size is 11,099,241,100 bytes. The per-client audit
lists all bona-fide video groups and counts; every client's counts agree with
the documented 2 + 6 pattern. Byte-identical copies under unrelated paths were
not tested because video contents were not hashed.

## Canonical representation and API

`identity_invariant_fas.data.msu_protocol` supplies:

- `load_msu_protocol(root)`: read and validate the entire public-release
  inventory with an anchored filename grammar and numeric subject mapping.
- `validate_msu_protocol(protocol)`: reject duplicate IDs/paths/acquisition
  slots, missing clients/slots, wrong cardinalities, overlap, and conflicting
  partition/class/device/annotation metadata.
- `audit_msu_protocol(protocol)`: deterministic inventory and readiness report.
- `write_msu_protocol_lock(root, metadata_path, audit_path)`: repeat indexing
  to check stable input, then write canonical metadata and audit outside raw data.
- `verify_msu_protocol_lock(root, audit_path)`: compare a fresh audit with the
  reviewed lock, rejecting drift before an experiment.

`MSUProtocol` holds official train/test client sets, source fingerprints, and
`MSURecording` entries. Each entry distinguishes relative filepath, canonical
client, PAD partition, PAD class (`0` bona fide, `1` attack), capture device
and model, attack type, presentation device, resolution token, scenario,
annotation path/digest, and media byte size. Bona-fide attack metadata is null.
`session_id` is always null: `scene01` is a scenario, not a verified session.
The generic `FASSample` and seven-column `SampleRecord` schemas are unchanged.

`record.video_id` is exactly `MSU-MFSD:` followed by its portable relative
filepath. It includes class, source filename, and extension, and is independent
of absolute dataset location. Future frames use
`record.probe_observation(frame_index)` to obtain distinct sample IDs and the
same source `group_id`, with `identity_id=client_id`. Calling without an index
describes one video-level observation. This method does not extract frames,
check actual frame indices, or construct probe splits.

Generate or explicitly update the reviewed artifacts:

```powershell
python -m identity_invariant_fas.data.msu_protocol --root datasets/MSU --metadata manifests/generated/msu_recordings.json --audit docs/audit/msu_protocol_lock.json
```

Check the lock without writing anything:

```powershell
python -m identity_invariant_fas.data.msu_protocol --root datasets/MSU --verify-lock docs/audit/msu_protocol_lock.json
```

The local run used `env_cuda/Scripts/python.exe`. Outputs use sorted metadata,
UTF-8/LF JSON, no timestamps, and no absolute roots. The normalized metadata
digest covers the complete recording table, video sizes, annotation hashes,
client assignments, and source fingerprints. It detects changes to names,
sizes, lists, and annotation bytes; it does **not** detect same-size changes
to video content or establish video decodability. Do not automatically
regenerate a lock to bypass a mismatch. Failed validation produces an error
and does not publish new lock outputs. Unknown files with recognized video
extensions and orphan `.face` files are rejected, including outside `scene01`;
non-media tools/documentation are not treated as recordings.

## RQ1 and RQ3 readiness

For either official cohort (train: 15 classes; test: 20 classes), every client
has two separate bona-fide source-video groups. Metadata therefore supports
a closed-set video-group-disjoint probe with the same clients on both sides.
An Android-to-laptop or laptop-to-Android design is possible for the entire
cohort, and would be a **project-defined camera-separated** probe. No such
split or experiment is executed here.

Only one bona-fide video per client per camera is available. Bona-fide-only
group CV cannot have more than two folds; within-camera video-disjoint probing
of those clients is not possible. Additional frames do not add independent
acquisitions. Device separation is not session separation: session-disjoint
probing is unsupported by verified local metadata, and temporal independence
of the two camera acquisitions is not established. MSU supplies no separate
official enrollment partition in this release.

For RQ3, the validated official 15/20 separation supports unseen-human PAD
evaluation when the encoder is trained only on the official training cohort.
It does not certify the training provenance of any existing checkpoint.
**MSU-MFSD has no official development partition.** Any future validation or
model-selection split must use only official training subjects and be labeled
project-defined. Test identities must not guide tuning or threshold selection.
No development split, custom fold, threshold, or training migration is added.

## Existing preprocessing fidelity audit

`scripts/preprocess_msu.py` consumes `index_msu_videos` and original videos plus
sibling `.face` files. It selects up to 30 annotated frame indices by uniform
spacing by default, decodes with OpenCV, attempts orientation auto-handling,
expands/clips face boxes, and writes JPEG crops and a legacy CSV.

There is a confirmed blocker before decoding: the legacy loader compares
two-digit list strings (`02`) directly with three-digit filename strings
(`002`) and rejects valid clients. The synthetic regression test reproduces
this mismatch. This task documents it and leaves the loader/preprocessor
untouched; the new canonical loader handles the documented numeric convention.

Where the legacy preprocessing succeeds on suitable metadata, each output
frame inherits `subject`, `split`, `label`, `video_id`, `camera`, and
`attack_type`; grouping is therefore preserved through the source filename
stem. These stems are unique in the locked inventory, but are a different
identifier convention from the new namespaced relative-path group ID.

The frame manifest does not retain an explicit original video path, source
fingerprints, a structured frame-index field (only its filename), preprocessing
version/settings, original annotation/eye coordinates, crop box, verified
session, or separate presentation-device field. Malformed annotation rows
are skipped, and duplicate frame indices overwrite earlier entries. The
bundled README notes rotated Android footage and boxes relative to rotated
frames; decoder orientation, index alignment, crop validity, frame coverage,
and write success remain untested. No preprocessing was run in this task.

## Gate before the first controlled experiment

Review and accept this metadata lock, then undertake a separate minimal
preprocessing integration/verification task: resolve the legacy normalization
blocker by using or explicitly adapting to the canonical protocol, retain the
canonical video ID and explicit source/frame/preprocessing provenance, and
verify orientation and crop/index fidelity on controlled samples. Define the
training-only model-selection rule and the exact RQ1 cohort/camera direction
before extracting frozen features or training models. Session claims require
new source evidence, not inferred camera or directory labels.

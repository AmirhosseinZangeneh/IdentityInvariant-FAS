# Dataset Protocols

The separate [sample metadata manifest](SAMPLE_MANIFEST.md) infrastructure is
available for future protocol reconstruction. Existing loaders and experiments
continue to use their current representations; canonical fold assignment is
not implemented by the new manifest layer.

## NUAA

For deterministic source metadata generation and the unresolved identity
provenance findings, see [NUAA source manifest](NUAA_SOURCE_MANIFEST.md).
This preparation layer does not generate subject-disjoint folds.

The project uses the NUAA face-detector-output format. Place these items under
`datasets/`:

- `ClientFace/`
- `ImposterFace/`
- `client_train_face.txt`
- `imposter_train_face.txt`
- `client_test_face.txt`
- `imposter_test_face.txt`

The legacy identity-generalization experiment used the official **test**
partition as a source pool and created subject-disjoint five-fold splits.
This custom protocol is reproduced by the supplied configs for continuity.

For a paper comparison against published NUAA official-protocol numbers, add a
separate official train/test experiment rather than mixing the two protocols.

## MSU-MFSD

The local 35-client, 280-video release has a validated
[official protocol lock and provenance audit](MSU_PROTOCOL.md). It has 15
training and 20 test clients, with no official development partition. Future
model-selection subsets must use only official training subjects and be
labeled project-defined.

Expected original structure:

```text
MSU/
├── scene01/
│   ├── real/
│   └── attack/
├── train_sub_list.txt
└── test_sub_list.txt
```

The database includes PittPatt `.face` files. The legacy preprocessing command is:

```bash
python scripts/preprocess_msu.py --root datasets/MSU
```

The script samples a fixed number of annotated frames per video and creates
`data_processed/MSU-MFSD/manifest.csv`. It is currently blocked on the local
release by the legacy loader's `02` versus `client002` normalization mismatch.
It remains unchanged; consult the protocol audit before future preprocessing.

Frame-level predictions should be aggregated to video-level scores for the
official MSU evaluation. That aggregation is intentionally left as an explicit
experiment step rather than silently mixing frame and video metrics.

## Replay-Attack

The new [Replay protocol layer](REPLAY_PROTOCOL.md) uses an explicit official
client roster and recording metadata, separating PAD from enrollment and
validating client-disjoint train/devel/test cohorts. Local raw data are absent;
only synthetic validation has been performed. Use this layer for future
controlled RQ1/RQ3 preparation. The generic CSV interface below is retained
for compatibility and does not validate official protocol semantics.

The database is restricted by its EULA and is not included. After official
access is granted, prepare face frames and a manifest with:

```text
path,label,subject,dataset,split,video_id,attack_type,camera
```

using 0 for bona fide and 1 for attack.

# Dataset Protocols

## NUAA

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

Expected original structure:

```text
MSU/
├── scene01/
│   ├── real/
│   └── attack/
├── train_sub_list.txt
└── test_sub_list.txt
```

The database includes PittPatt `.face` files. Use:

```bash
python scripts/preprocess_msu.py --root datasets/MSU
```

The script samples a fixed number of annotated frames per video and creates
`data_processed/MSU-MFSD/manifest.csv`.

Frame-level predictions should be aggregated to video-level scores for the
official MSU evaluation. That aggregation is intentionally left as an explicit
experiment step rather than silently mixing frame and video metrics.

## Replay-Attack

The database is restricted by its EULA and is not included. After official
access is granted, prepare face frames and a manifest with:

```text
path,label,subject,dataset,split,video_id,attack_type,camera
```

using 0 for bona fide and 1 for attack.

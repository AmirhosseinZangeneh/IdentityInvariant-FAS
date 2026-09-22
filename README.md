# IdentityInvariant-FAS

Identity-invariant representation learning for face presentation attack
detection (PAD).

This repository investigates whether a face anti-spoofing encoder relies on
subject-specific shortcuts and whether adversarial subject suppression can
reduce that identity leakage while preserving spoof discrimination.

## Research idea

The proposed II-ECNN uses a shared ECNN encoder with two branches:

```text
                         +--> Spoof classifier --> bona fide / attack
Input --> ECNN encoder --|
                         +--> GRL --> Subject classifier
```

The gradient reversal layer (GRL) leaves features unchanged in the forward
pass and reverses the subject-classification gradient during backpropagation.
The intended effect is to retain spoof-relevant evidence while discouraging
identity-discriminative features.

## Models

- `PaperECNNClassifier`: baseline ECNN classifier.
- `IdentityInvariantECNN`: ECNN + adversarial subject branch.
- `AblationECNN`: architecture-matched two-head model **without** GRL.

The ablation is intentionally architecture-matched so that GRL is the primary
controlled difference.

## Scientific status

The original workspace produced promising NUAA development results, including
reduced linear identity-probe accuracy for an intermediate GRL strength.
However, the refactor uncovered several issues that must be corrected before
those values are treated as final paper evidence:

- the historical ablation was not architecture-matched;
- some representation analyses used 64x64 inputs while training used 160x160;
- the historical GRL sweep did not use the same warm-up schedule as the main
  II-ECNN experiment;
- one sweep script swapped APCER/BPCER names;
- the NUAA subject-k-fold experiment is a custom protocol, not the official
  NUAA train/test protocol.

See [`docs/SCIENTIFIC_AUDIT.md`](docs/SCIENTIFIC_AUDIT.md) before using the
legacy results in a manuscript.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[video,plot,dev]"
```

## NUAA subject-disjoint experiment

Place NUAA under `datasets/` as described in
[`docs/DATASETS.md`](docs/DATASETS.md).

Baseline:

```bash
python experiments/train_nuaa_kfold.py --config configs/nuaa_baseline.yaml
```

II-ECNN:

```bash
python experiments/train_nuaa_kfold.py --config configs/nuaa_iecnn.yaml
```

Controlled ablation:

```bash
python experiments/train_nuaa_kfold.py --config configs/nuaa_ablation.yaml
```

GRL sensitivity:

```bash
python experiments/grl_sweep.py --config configs/grl_sweep.yaml
```

## Identity leakage

Use the [group-aware closed-set probe API](docs/IDENTITY_PROBING.md) for
publication protocols, with explicit sample, identity, and group metadata.
The historical NUAA command below is exploratory only: it lacks verified
group/session metadata and now requires explicit opt-in. Use the same
evaluation image size as the training configuration:

```bash
python experiments/identity_leakage_nuaa.py ^
  --allow-legacy-sample-cv ^
  --model ii_ecnn ^
  --checkpoint outputs/nuaa_subject_kfold/ii_ecnn_lambda_0.05/fold_1/best_model.pt ^
  --num-subjects 12 ^
  --image-size 160 ^
  --output results/generated/identity_leakage_fold1.json
```

Both probe paths require known identity classes for the identity classifier.
The legacy command does not prevent frames from the same video/session from
crossing folds and must not support publication claims of independent-group
identity recoverability.

## MSU-MFSD

MSU-MFSD is video-based. Its [canonical protocol audit](docs/MSU_PROTOCOL.md)
verifies the local 15/20 client separation and 280-video inventory. The legacy
frame-extraction command below remains blocked by the documented subject-ID
normalization mismatch; preprocessing integration is a separate next step:

```bash
python scripts/preprocess_msu.py --root datasets/MSU
```

See [`docs/DATASETS.md`](docs/DATASETS.md) for protocol notes.

## Replay-Attack

Replay-Attack is license-restricted and is not distributed with this project.
The [Replay metadata/protocol API](docs/REPLAY_PROTOCOL.md) validates explicit
client cohorts and separates PAD recordings from enrollment for future RQ1/RQ3
preparation. Official metadata and raw-data verification are required before
experiments. The existing generic manifest adapter remains available.

## Metrics

The project uses a single label convention:

- `0`: bona fide / real
- `1`: attack / spoof

Accordingly:

- APCER = FN / (FN + TP)
- BPCER = FP / (FP + TN)
- ACER = (APCER + BPCER) / 2

AUC and EER use the model's attack score.

## Repository structure

```text
IdentityInvariant-FAS/
├── configs/
├── datasets/
├── docs/
├── experiments/
├── scripts/
├── src/identity_invariant_fas/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── archive/
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Data and checkpoints

Raw datasets, trained weights, generated frames, predictions, and experiment
outputs are excluded from Git. Respect the license/EULA of every benchmark.

## Citation

A `CITATION.cff` file is provided. Update it with the final paper title, DOI,
and repository URL after publication.


## NUAA official-protocol baseline

For numerical comparison with work that reports the official NUAA train/test
lists, keep this experiment separate from the custom subject-k-fold protocol:

```bash
python experiments/train_nuaa_official.py --config configs/nuaa_official_baseline.yaml
```

## Cross-dataset manifest evaluation

After MSU-MFSD frames have been prepared, evaluate a NUAA-trained checkpoint
without fine-tuning:

```bash
python experiments/evaluate_manifest.py ^
  --manifest data_processed/MSU-MFSD/manifest.csv ^
  --split test ^
  --model ii_ecnn ^
  --checkpoint outputs/nuaa_subject_kfold/ii_ecnn_lambda_0.05/fold_1/best_model.pt ^
  --video-level ^
  --output results/generated/nuaa_to_msu_fold1.json
```

For a final cross-dataset table, define in advance whether fold checkpoints are
ensembled or each fold is evaluated separately and then summarized. Do not
choose the aggregation rule after inspecting target-dataset test performance.

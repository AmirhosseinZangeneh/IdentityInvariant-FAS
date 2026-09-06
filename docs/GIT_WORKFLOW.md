# Git Workflow

Recommended branch policy for this research project:

- `main`: reproducible, reviewed research releases only.
- `develop`: integration branch for validated experiment changes.
- `feature/<name>`: one scientific or engineering change per branch.

Suggested next branches:

```text
feature/msu-cross-dataset
feature/replay-attack
feature/official-nuaa-reproduction
feature/representation-analysis
```

A useful commit sequence for the first public push:

```text
chore: initialize reproducible research package
feat: add ECNN baseline and adversarial identity model
feat: add architecture-matched GRL ablation
feat: add subject-disjoint NUAA protocol
feat: add identity leakage evaluation
feat: add MSU-MFSD preprocessing and manifest support
docs: add scientific audit and reproducibility notes
test: add model, GRL, metric, and split validation
```

Do not fabricate historical commit dates. If the repository is initialized now,
make honest commits from the current state onward.

Before merging a feature branch:

```bash
python -m compileall src experiments scripts tests
pytest -q
```

Never commit raw datasets, restricted media, trained weights, or private
credentials.

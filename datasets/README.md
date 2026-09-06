# Datasets

Dataset files are deliberately excluded from this repository.

Supported/target datasets:

- **NUAA**: the current experiments use the face-detector-output format.
- **MSU-MFSD**: use the original videos plus official `.face` annotations;
  `scripts/preprocess_msu.py` creates a frame manifest.
- **Replay-Attack**: access is license-restricted. Prepare an equivalent
  manifest after receiving official access.

Never commit restricted dataset media to the repository.

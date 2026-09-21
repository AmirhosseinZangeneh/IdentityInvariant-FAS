# Replay-Attack local availability — 2026-09-21

This is a repository availability check, **not a real-data protocol audit**.

- `datasets/ReplayAttack/` exists and is empty, including hidden-file inspection.
- The configured `data_processed/Replay-Attack/manifest.csv` is absent.
- No Replay client roster, official recording catalog, or enrollment mapping
  was found in the inspected repository artifacts.
- The existing generic CSV loader only checks the dataset tag. Its config
  supplies no client or enrollment protocol evidence.

Official online documentation was consulted for protocol semantics and
explicit metadata fields; references are in [Replay protocol](../REPLAY_PROTOCOL.md).
No raw data, licensed catalog, or official ID mapping was downloaded.
No actual client IDs, video counts, overlaps, missing-file statistics, or
enrollment coverage are reported because they cannot be measured locally.

Implemented validation is exercised exclusively by synthetic fixtures,
including an explicitly synthetic 15/15/20 roster. Those fixtures are not
evidence about dataset contents.

Blocked real-data checks: catalog authenticity/transcription, discovered
client sets and partition disjointness, actual class/video counts, complete
enrollment/access coverage, missing/duplicate recordings, file validity,
inventory completeness, and any verified recording-session mapping.

Next gate: obtain licensed raw data and official client/recording metadata;
prepare and review a provenance-bearing export, reconcile it with the raw
inventory, and validate the selected enrollment/access cohort before any RQ1
feature extraction or identity-classifier fitting.

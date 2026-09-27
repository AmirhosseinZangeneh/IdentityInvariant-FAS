# NUAA scientific role: prospective decision

Decision at repository baseline `272f2c836aa3bf6596ee633b97cbf59da7f5f3da`:
**NUAA_ROLE_FROZEN = true**, with **identity_provenance_status = unresolved**.
This freezes permitted interpretation, not human identity provenance or
authorization to run experiments. No new scientific result is produced.

## Evidence and limits

The committed [source-manifest report](NUAA_SOURCE_MANIFEST.md) and
[source audit](audit/nuaa_source_audit.json) record the 2026-09-21 local
inspection of four source lists, directory inventories and the bundled readme.
This decision uses those committed observations; it does not re-inspect raw
images or assert that today's raw copy has been revalidated.

| Source partition | ClientFace (bona fide) | ImposterFace (attack) | Total |
| --- | ---: | ---: | ---: |
| Train | 1,743 | 1,748 | 3,491 |
| Test | 3,362 | 5,761 | 9,123 |
| All | 5,105 | 7,509 | 12,614 |

The audit records unique listed paths, matching labels/source splits, existing
files, no unlisted image paths/folders, repeatable generation, and hashes of
the lists, readme, manifest and path inventories. Inventory hashes cover
paths, not image bytes; these checks do not prove authenticity, content
uniqueness, or human identity. Attack subtype and preprocessing version are
also unresolved. The generated CSV is private/ignored, not a committed
independent identity mapping.

| Folder-token evidence | Exact observed set |
| --- | --- |
| ClientFace: 15 | `0001`–`0015` |
| ImposterFace: 15 | `0001`–`0012`, `0014`, `0015`, `0016` |
| Intersection: 14 | `0001`–`0012`, `0014`, `0015` |
| Client-only | `0013` |
| Imposter-only | `0016` |
| Union: 16 tokens | `0001`–`0016` |

**Equal cross-class tokens denote the same human: UNPROVEN.** The committed
report quotes the readme's separate-directory-per-subject description and
`0001~0016` filename range, but no explicit cross-class mapping or explanation
of the asymmetric IDs. Neither counts nor matching strings establish one.
Do not pair, merge, remap, exclude, or visually identify people to resolve it.

The loader already assigns the raw parent-folder token to `FASSample.subject`.
`build_subject_mapping`, k-fold and validation splitting group that string
across both class directories. Thus equal tokens already share an auxiliary
class and split group. This existing computational grouping is preserved and
disclosed as a **proxy**, not endorsed as a human pairing. Class-prefixed
tokens would also lack verified human semantics and would change the protocol;
this decision does not introduce them. `subject_id` in the source manifest
likewise retains the raw token. Internal API/CLI names remain for compatibility.

## Allowed roles and wording

1. **Official-source-list PAD evaluation:** retain train/test assignments for
   baseline reproduction, controlled backbone evaluation and implementation
   verification. The current official runner reserves project-defined
   folder-token-disjoint validation from official training lists; disclose
   that choice when comparing published baselines. Do not tune on the official
   test set. Official train/test is not folder-token-disjoint: within-class
   overlaps are ClientFace `0004,0006,0007` and ImposterFace `0001`–`0009`.
2. **Custom folder-token-disjoint generalization experiment:** the current
   k-fold configuration repartitions the official **test** pool and reserves
   validation by token within the outer training pool. Report the source pool,
   seed, folds and validation rule. Keep it separate from official-protocol
   results; no folder-token overlap does not certify absence of human overlap.
   The all-list union and current test pool each have 16 tokens; the official
   training pool has 9. These are token counts, not verified human counts.
3. **Exploratory proxy intervention/representation analysis:** NUAA remains
   useful for testing gradient routing and effects on folder-token predictability
   or PAD performance. NUAA alone cannot establish the central human-identity
   suppression claim on the available evidence.

Use **folder token**, **folder ID**, **folder-token-disjoint**, and
**folder-token proxy predictability/suppression**. Do not claim verified
human/person/subject identity, a verified human count, generalization to unseen
human identities, or identity-invariant human-subject-disjoint evaluation.
Historical output names containing `subject` are compatibility labels; their
files remain untouched and do not acquire stronger semantics retrospectively.
The historical `audit/pre_protocol_lock_state.md` wording likewise supplies
no identity mapping; this prospective decision supersedes its NUAA claim labels
without rewriting that audit snapshot.

## Controlled ablation and probes

The three-arm implementation is **implementation-valid**: its shared model,
beta, warm-up, fixed/DANN schedules and signed gradients remain unchanged.
It is **not yet scientifically human-identity-grounded on NUAA**. The subject
classifier predicts token classes: the positive arm encourages token
discrimination and the adversarial arm opposes it. Both inherit uncertain
cross-class grouping. Tokens occurring in only one PAD class also encode a
class association; suppression of such a proxy need not be suppression of
human identity. Representation shifts and t-SNE separation cannot resolve
that ambiguity. Spoof-only remains a valid PAD comparator.

An identity probe trained on these labels measures token recoverability.
Low probe accuracy is not proof of low human-identity information. NUAA also
lacks verified recording/session grouping for independent-group probing.
The legacy sample-CV opt-in remains exploratory with possible correlated-sample
leakage; the group-aware API cannot certify identity provenance by itself.
See [identity probing](IDENTITY_PROBING.md). Existing k-fold results cannot
support verified unseen-human claims, even if their numerical computation is
otherwise correct; earlier schedule/ablation limitations also remain in force.

Prospective runner metadata fixes `identity_semantics=folder_token_unverified`,
`identity_provenance_status=unresolved`, and `human_identity_verified=false`.
Custom k-fold uses `split_semantics=folder_token_disjoint`; official evaluation
uses `official_source_lists`; legacy probing uses `sample_level_cv`. Both PAD
runners label validation `project_defined_folder_token_disjoint`. These labels
describe current algorithms, not a new dataset audit. They are not configurable
certification switches. Checkpoints/results receive them without changing
optimization, metrics or splits.

## Dataset for the final human-identity claim and upgrade conditions

**MSU-MFSD is the best-supported planned carrier in the current repository.**
The [committed MSU protocol audit](MSU_PROTOCOL.md) ties real/attack `clientID`
to release documentation and numeric list matching, with 35 clients in
disjoint 15/20 training/test cohorts and two bona-fide source videos per client.
This is documentary human-client evidence, not independent visual verification.
Before experiments, revalidate the source lock, preserve source/video/client
mapping through approved preprocessing, satisfy existing execution gates, and
freeze training-only selection and video-group-disjoint probe cohorts. Camera
labels are not verified session IDs. This decision neither changes MSU gates
nor reopens its frame-correspondence/timing work.

**Replay-Attack is a conditional second carrier.** The committed
[protocol design](REPLAY_PROTOCOL.md) describes explicit official client,
recording, PAD cohort and enrollment/access metadata. The
[availability audit](audit/replay_metadata_availability.md) found no licensed
raw inventory or verified roster/catalog; synthetic fixtures are not evidence
of real identities. Obtain and fingerprint official mapping artifacts, review
their export, reconcile recordings with raw files, verify cohort separation
and enrollment/access coverage, and freeze source-group-disjoint evaluation
before making human-identity claims. Session claims require separate evidence.

Upgrading NUAA requires an authoritative, reviewable sample/folder-to-human
mapping across and within classes, an explanation of `0013`/`0016` and token
aliases, and a versioned audit tying that mapping to exact source fingerprints.
Then separately approve new human-based folds, auxiliary labels and probe
groups; validate no human/group leakage, freeze selection rules and rerun
prospectively. Do not upgrade legacy results by relabeling them. A verified
mapping enables such experiments; it does not itself demonstrate suppression.

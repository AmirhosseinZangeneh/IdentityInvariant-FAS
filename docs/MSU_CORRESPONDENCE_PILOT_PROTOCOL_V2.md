# MSU correspondence pilot protocol v2: resilient timing evidence

Design dated 2026-09-26 against frozen v1 tooling commit
`8ef9a86cebbb67a5f3159d91ab84010779fcc5e2` on `research-reconstruction`.
**Design/documentation only: not implemented, not execution authorization,
new-run execution status `not_executed`.** The provider/trust profile below
must be resolved and frozen before a new Session 1. No external anchor has
been requested, published, or purchased by this design task.

## 1. Scope, inherited science, and Run 1 exclusion

Read this document together with
[protocol v1](MSU_CORRESPONDENCE_PILOT_PROTOCOL.md), exact-byte SHA-256
`dfbd88ad785bf32b947e8f2c54aafc3a593d5474112dae6b4eb864cf439b39a4`.
V1 remains unchanged. Its scientific sections are incorporated by reference:
question/hypotheses, six cases, 24 associations, 16 pixel groups, frozen clean
assets and fingerprints, measurement definition, coordinates, reviewer
procedure, uncertainty, ties, comparisons, missingness and interpretation.
Only timing evidence, the associated output fields and new-run isolation are
amended here. This document takes precedence only for those matters.

Retain the strict rule:

`valid locked new Session 1 -> >=172800 real elapsed seconds -> new Session 2`

The 48-hour interval is not shortened, rounded up, or converted to two calendar
dates. Retain the 60-minute session cap, mandatory break after eight groups,
synthetic-grid prerequisite, 1x/2x/4x nearest-neighbor display, strict image-left
x < image-right x for two measurable eyes, uncertainty rules, one measurement
per unique group/session, both coordinate conventions and strict 1-pixel
comparison margin. No threshold fitting, new frame selection, TEST data or PAD
training/evaluation is introduced. Geometry and codec gates remain separate.

[Run 1 is operationally aborted](MSU_CORRESPONDENCE_PILOT_RUN1_RESTART_DECISION.md).
Its locked Session 1 is ineligible for this protocol. Its private root must
never be opened by the new execution tool as an input, overwritten or reused.
Prior participation is additional reviewer exposure: disclose it without
reopening old marks/order. One actual reviewer can perform the new pair;
neither two sessions nor the restart create independent reviewers or an
untouched confirmation. Geometric agreement still cannot prove historical
PittPatt input identity.

Preserve `policy_state=P4`, `frozen_policy=null`,
`fidelity_status=manual_review_pending`, `experiment_ready=false`.

## 2. Timing option comparison

The comparisons below are design judgments under a threat model of accidental
or simple manipulation, not a malicious machine owner rewriting every artifact
or a compromised trusted time authority.

| Dimension | A: signed RFC 3161 authority tokens | B: GitHub/server-side hash anchors | C: local wall clock plus reboot/event evidence |
| --- | --- | --- | --- |
| Scientific strength | Independent time assertions tied to exact commitments; can establish a conservative interval with declared uncertainty and correct event ordering. Depends on authority time quality and workflow integrity. | Independent server observations can corroborate ordering. Ordinary API records lack a signed, bounded-accuracy time guarantee suitable for the primary 48-hour proof. | Cannot independently establish off-machine elapsed time; editable clock and logs share the same local trust domain. |
| Power-loss tolerance | Signed evidence survives offline; no continuously running process between sessions. Network needed only at evidence checkpoints. | Server records survive local outages; network/account access needed at checkpoints and later retrieval. | Reboot logs can survive but may be missing or discontinuous; cannot reconstruct trustworthy off-time from clock values alone. |
| Tamper evidence | Signature, imprint, nonce, certificate and local ledger bindings detect altered or mismatched tokens. | Server resource IDs and retained responses aid checking, but comments can be edited/deleted; content must not be accepted solely with an old creation time. | Logs and local timestamps can be changed, cleared, rolled over or affected by clock correction. |
| Auditability | Archive original tokens and verification material for later verification without a live original process. Long-term trust/revocation evidence must also be retained. | Usually requires continuing server access; a saved JSON response is not a portable server signature. A purpose-built signed service would require its own frozen trust design. | Useful diagnostic history, not a reproducible independent lower bound. |
| Dependencies | Qualified TSA availability, PKI/trust material, validator and possibly service fees/account. No provider is selected yet. | GitHub or another service, permissions, retention and API stability; a custom signed service adds operation/security duties. | Windows logging and local clock configuration; minimal external dependency but insufficient evidence. |
| Privacy | Send only salted commitment digests and a random request nonce. TSA can observe network/account/timing metadata. | Only salted hashes should leave the host; account/repository/event timing can still reveal activity. Public posting adds unnecessary exposure. | Data stay local; no external disclosure, but weaker assurance. |
| Complexity | Moderate/high: binary token parsing, certificate and policy validation, bounded time arithmetic, crash-safe state transitions. | Low/moderate for posting, much higher for defensible immutability, accuracy and archived authentication. | Low to collect; high and still insufficient to infer reliable time from incomplete logs. |
| Failure modes | Outage, wrong imprint/policy/nonce, untrusted or revoked signer, unbounded accuracy, truncated files, lost local evidence. Block progression rather than fall back. | Edited body attached to old `created_at`, deleted record, stale response, wrong server/account, unavailable API, unknown clock error. | Manual/NTP correction, battery clock reset, timezone changes, event gaps and log manipulation. Absence of an event proves no continuity. |

RFC 3161 defines a signed time token for a hash imprint, with time, policy and
optional accuracy/nonce fields; accuracy can instead be specified by the
authority's policy. Missing accuracy is not evidence of zero uncertainty.
[RFC 3161, sections 2.2-2.4.2](https://www.rfc-editor.org/rfc/rfc3161.html).
GitHub documents creation/update timestamps and update/delete operations for
issue comments; an editable comment is therefore not an immutable timestamp
of its current content. [GitHub REST issue comments](https://docs.github.com/en/rest/issues/comments).
Windows event 4616 records system-time changes, but such diagnostics are not
an independent time authority. [Microsoft event 4616](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/event-4616).

**Recommendation: A is mandatory primary evidence.** Local UTC, monotonic
session durations and reboot/time-change diagnostics are secondary context
only. Option B is optional corroboration if separately authorized and selected
before preparation; it never replaces a missing/invalid primary token or
supplies a permissive alternative threshold. No GitHub posting is authorized
by this document. Disagreement must be preserved and investigated; an
unresolved contradiction in primary evidence blocks progression. Network
delay in an optional receipt is not automatically a clock contradiction.

## 3. Freeze the timing profile before the new Session 1

Prepare a fresh private root, provisionally
`data_processed/recovery/msu-correspondence-pilot-v2/`, exclusively. Existing
content means refuse preparation; no overwrite or adoption. Its resolved path
must differ from Run 1 and be Git-ignored. The lifecycle is entirely new:
`prepare -> Session 1 -> >=48h -> Session 2 -> analysis`.

Freeze a unique cryptographically random run ID, new session IDs, new orders
and secret opaque mappings, actual reviewer identity/disclosure, authorization,
asset/evidence hashes, v1 and v2 document hashes, this restart-decision hash,
the future v2 tooling commit and exact executable/source/dependency hashes.
V1's tooling commit is a historical reference, not a claim that v2 code exists.
Retain exclusive plan/seal creation and full preflight. No self-referential
hash is required: finalize the documents/tool first, then hash them into the
plan; the plan digest is used by subsequent records.

The sealed timing profile must additionally name the TSA endpoint and policy
OID, approved trust roots/signer policy, algorithms, accepted time-accuracy
bounds, revocation-validation rules and archive requirements. Pin the chosen
validator/version and a precise UTC-to-elapsed-time conversion including leap
second handling. Unbounded or ambiguous time uncertainty is disqualifying.
No provider switching, trust-root substitution or accuracy relaxation after
Session 1. An unplanned rollover/outage leaves the run pending or requires a
separate restart decision, not an improvised alternate authority.

Before any real presentation, test the selected endpoint with synthetic
commitments only, verify its trust/accuracy/privacy properties, and freeze
the completed profile. Missing decisions in section 8 prevent preparation.

## 4. Bound commitments without exporting private content

Use SHA-256 and a frozen deterministic UTF-8 serialization, with a versioned
domain-separation tag. Exact serialization and test vectors must be fixed in
the implementation specification. Preserve the actual hashed bytes locally;
do not rely on regenerating them with another serializer later.

A private commitment envelope contains phase, random run/session IDs, plan
and protocol/tool/profile digests, a new 256-bit random privacy salt and the
relevant ledger/lock/token hashes. The TSA receives **only its digest**, the
hash algorithm identifier, requested policy, a distinct random nonce, and
necessary protocol options. It never receives the envelope, names, paths,
measurements, candidate labels, ordering/mappings, images or raw lock file.
The random salt also avoids exposing an easily enumerable unsalted identity
digest. Hash-only disclosure is not anonymity: service logs can retain IP,
account and request times. Archive credentials separately from evidence.

Use exclusive immutable attempt records with fresh nonces; archive request
bytes, response bytes, token, envelope, token hash, verification result and
selected-attempt binding. Retain failed/retried attempts. T1 is the first valid
RFC 3161 timestamp obtained after the valid Session-1 lock; retain it durably
and preserve that choice. T2 is the first qualifying valid pre-Session-2
timestamp whose conservative lower-bound interval satisfies the frozen
>=172800-second rule. Later T2 attempts to meet that interval are allowed only
because earlier valid attempts may legitimately be too early. Once the first
qualifying T2 exists, it is the authoritative T2 for that run; no favorable-token
cherry-picking is allowed. An interrupted write
cannot be treated as valid. Recovery may add a new attempt but cannot edit the
old one or manufacture an absent response.

## 5. Causal sequence and the 48-hour proof

The following sequence is a proposed application design, not a property that
a timestamp token establishes by itself. A token proves a commitment existed
by an authority time; it does not observe a reviewer, an image presentation,
or a session-lock operation. The audited state machine must enforce the order.

1. **New Session 1:** complete the unchanged measurement procedure and write
   its completion event and immutable lock durably. Verify their chain and
   plan binding without displaying marks. Only then construct `S1_POST_LOCK`,
   binding the exact lock bytes, events digest and all run identities. Obtain,
   verify and durably retain token T1. The token is a conservative upper bound
   on the earlier completion/lock time. If the network is unavailable, keep
   the session locked and wait; a later T1 is acceptable but starts a later,
   more conservative waiting bound. Do not backdate a token using local UTC.
2. **Offline interval:** the workstation may shut down, lose power, reboot,
   remain off/asleep, or terminate every local process. No witness heartbeat
   is required. Reopen only the new run's records, replay seals/locks/evidence,
   and verify software/assets when execution is requested. Reset local
   monotonic counters are never subtracted across boots. Session 1 remains
   locked and cannot be presented again or edited.
3. **Session 2 pre-start gate:** before its start event, synthetic grid or
   candidate display, construct `S2_PRE_START` with the fixed Session 2 ID,
   fresh attempt nonce/salt, T1 hash, Session-1 lock hash and all run bindings.
   Obtain and validate T2. Evaluate the lower-bound rule below. If too early,
   preserve the attempt and wait; a later attempt is permitted before any
   Session 2 starts. If unavailable or inconsistent, do not start Session 2.
4. **Consume eligibility:** after a qualifying T2 is durably stored, write a
   write-once eligibility receipt identifying the exact T1/T2 pair and computed
   bounds. The Session-2 start event must include that receipt and T2 hashes,
   then follow the usual synthetic-grid and measurement procedure. A token
   acquired after the start cannot retroactively make that start eligible.
   Reboot before start is harmless if all evidence survives: revalidate and
   use the recorded eligibility receipt once; later start only increases the
   interval. No second start is allowed after a start event exists.
5. **Analysis:** only after Session 2 also completes and locks, replay both
   sessions, plan/seal, timing tokens, trust evidence, receipt and causal
   bindings through one authoritative verifier. Both CLI and public API must
   use that verifier; summaries/calculations cannot accept raw unverified
   session lists. No analysis during waiting or after an invalid session.

Let authority issuance times have conservatively established intervals
`[L1,U1]` for T1 and `[L2,U2]` for T2, in real elapsed seconds on the frozen
time scale. Require:

```text
gap_lower_bound_seconds = L2 - U1
eligible iff gap_lower_bound_seconds >= 172800
```

For a valid symmetric error bound e, use `L = t-e`, `U = t+e`, so the bound is
`(t2-e2) - (t1+e1)`. Use exact integer/rational units, not rounded displayed
hours. Example only: if each bound is one second, a token-center separation
of exactly 48 hours is insufficient; 48 hours plus two seconds meets the
bound. One second is illustrative, not a chosen TSA accuracy or a scientific
threshold. Freeze real provider-specific bounds before preparation.

The reasoning is: actual Session-1 lock <= U1, and actual Session-2 start
>= L2 by enforced ordering. Therefore their separation is at least L2-U1.
Token delivery and offline delays are conservative; they do not subtract from
the 48 hours. Neither an S1 token taken before completion nor an S2 token taken
after start supplies these inequalities. Do not claim an exact physical gap
or upper bound on it from these two tokens alone.

Use authenticated accuracy claims and a frozen conservative bound; reconcile
any token/policy discrepancy by refusal, not by choosing the smaller error.
If accuracy is omitted, a specifically approved, archived policy bound is
required. Handle leap seconds explicitly; a POSIX/calendar subtraction with
unknown leap/smear behavior is not sufficient evidence of 172800 real seconds.
If the frozen conversion evidence cannot cover the interval, refuse the gate.

## 6. Verification, interruption and failure rules

The future verifier must check token signature and expected signer, certificate
chain/EKU and validity at the relevant trusted time, revocation evidence under
the frozen policy, hash algorithm/imprint, nonce, policy, response status,
time format/accuracy, phase and exact run/session/protocol/tool/lock bindings.
Reject unknown critical extensions, altered/truncated encodings, wrong-phase
tokens, cross-run substitution and inconsistent issuance order. Store original
binary material and independent verification diagnostics. Never accept a
human-read timestamp string, screenshot or HTTP Date header as a substitute.
OpenSSL provides request/response verification and chain-verification options;
choosing a precise invocation/library and archiving all validation inputs is
still implementation work. [OpenSSL timestamp tooling](https://docs.openssl.org/3.5/man1/openssl-ts/).
Account for the certificate-identification update where relevant.
[RFC 5816](https://www.rfc-editor.org/rfc/rfc5816.html).

Missing service access means **pending**, without invalidating an otherwise
completed session or authorizing progression. A machine restart between
sessions does not require a new Session 1 if its new-run lock and evidence
remain intact. Lost/corrupt measurements or an unverifiable lock still fail
closed; resilience to downtime is not permission to reconstruct records.
Archive certificates, policy snapshots and applicable signed revocation data
so later verification need not assume today's local clock or today's trust
store. If later verification cannot establish trust at issuance, do not claim
successful verification or release analysis on an exception.

Maintain monotonic duration checks within each active session for its existing
60-minute cap, stable display configuration and event ordering. Interruptions
during measurement still leave an incomplete/invalid, non-resumable session;
this amendment permits off-time **between** sessions, not splicing marking
across processes or boots. A completed lock awaiting network evidence is not
an interrupted measurement session. Local UTC changes between sessions are
recorded as diagnostics, never used to shorten the gate or invalidate valid
external evidence merely because a battery clock reset. Altered ledger bytes
remain an integrity failure.

No v1 `clock/failed.json` semantics are imported into the new run. Missing
failure markers never prove success. Derive state from positive validated
records and exact bindings, not absence of a stop file. No fallback to B, C,
manual attestation or a changed threshold is permitted when A fails.

## 7. Private timing outputs and future acceptance tests

Keep v1 scientific result arrays/rules. Extend private timing provenance with
run/session IDs; protocol/tool/profile/plan hashes; phase and attempt IDs;
envelope/request/response/token hashes and relative locations; signer/chain
and policy identifiers; validation evidence; normalized time bounds; selected
receipt; chronology checks; and status/reasons. None is an executed result in
this design document. All actual values remain absent or `not_executed`.

Session 1 has `gap_hours=null`. For Session 2, report `gap_hours` as
`gap_lower_bound_seconds / 3600`, with
`gap_hours_kind=external_conservative_lower_bound`, not as an exact gap or
v1's cross-session monotonic delta. Retain token-center separation separately
and local UTC as non-authoritative diagnostics. This descriptive field is
computed only after the single authoritative timing verifier succeeds; it
does not decide eligibility. Record the schema/version difference explicitly.

Before execution, require synthetic tests for exact/just-below threshold with
accuracy, fraction/UTC/leap handling, forward/backward local clock changes,
power loss at every persistence boundary, reboot between sessions, unavailable
service, delayed post-lock anchoring, early pre-start attempts, token replay,
wrong run/session/tool/protocol/phase/lock, signatures/certificates/revocation,
unknown accuracy, truncated tokens, wrong nonce/imprint, cross-run directory
rejection, premature CLI/API analysis, and no network disclosure of private
content. Exercise all six existing integrity fixes, local session limits,
masking, coordinate validation and unchanged scientific gates. Verify a second
auditor can reproduce token checks from the archived package. Do not use real
Run 1 measurements or images as test fixtures.

## 8. Decisions required before implementation and execution

Before implementation, settle a concrete TSA/service and accessible endpoint,
policy and accuracy evidence; costs/availability/privacy terms; accepted trust
roots and signer lifecycle; revocation/archive rules; verifier/dependencies;
serialization and time-scale rules; crash-safe record/receipt schema; and
whether optional server corroboration is worth its dependencies. A provider
without an auditable bounded time claim cannot implement this proposal as-is.
Also specify private backup/access controls, new output-root identity, and
network disclosure authorization. No endpoint has been approved by this draft.

Before execution, freeze the resolved profile, finalized v2 protocol and
reviewed implementation hashes in the new plan; finish synthetic/adversarial
tests; confirm the actual reviewer's participation and additional exposure;
and obtain separate execution authorization. V1 test passes do not certify a
future v2 implementation. Until these gates are complete, v2 is a prospective
design only and Run 1 remains operationally aborted and untouched.

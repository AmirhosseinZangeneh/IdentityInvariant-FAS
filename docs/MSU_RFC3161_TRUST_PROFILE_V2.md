# MSU pilot V2: RFC 3161 trust and verification profile

Profile identifier: `msu-rfc3161-v2.1`. Design dated 2026-09-26 against
`research-reconstruction`, committed HEAD
`d1a424327b05284e5bd7edc128e652dce99a43f2`.

**Documentation only; not implemented; Run 2 `not_executed`.** The normative
choices below are prospective requirements. Provider approval remains blocked
pending qualification on the specific evidence identified in section 12; this is not a claim
that a deployment-ready provider configuration has already been frozen.

Read with [protocol V2](MSU_CORRESPONDENCE_PILOT_PROTOCOL_V2.md) and the
[Run 1 decision](MSU_CORRESPONDENCE_PILOT_RUN1_RESTART_DECISION.md).
Run 1 remains permanently `operationally_aborted`; no Run 1 record is an input.
Start a completely fresh, ignored private root, provisionally
`data_processed/recovery/msu-correspondence-pilot-v2/`.
Retain `prepare -> Session 1 -> >=172800 real seconds -> Session 2 -> analysis`.
No scientific measurement, comparison, interpretation or reviewer rule changes.

Preserve exactly `policy_state=P4`, `frozen_policy=null`,
`fidelity_status=manual_review_pending`, `experiment_ready=false`.
The threat model covers accidental/simple manipulation, not replacement of all
local artifacts by a malicious owner, compromised TSA keys or a dishonest TSA.

## 1. Provider recommendation and limits of available evidence

**Recommend DigiCert as the candidate for qualification, not yet approved for
Run 2.** Its official documentation identifies `http://timestamp.digicert.com`
as its RFC 3161 service and supplies timestamp chain downloads. This establishes
a documented service, not verified availability from this workstation today.
[DigiCert RFC 3161 service documentation](https://knowledge.digicert.com/general-information/rfc3161-compliant-time-stamp-authority-server).

| Item | Decision or evidence limit |
| --- | --- |
| Candidate endpoint | Exactly `http://timestamp.digicert.com`, HTTP POST to `/`; no query parameters or alternate hosts. |
| Access/authentication | Official usage instructions show no credential parameter. They do not establish an unrestricted research-use entitlement, price, SLA or all access restrictions. Confirm those points with current provider terms/documentation before approval. Do not assume anonymous access means permission or perpetual free service. |
| HTTP/HTTPS | HTTP is explicitly documented. HTTPS support/equivalence has not been established here; do not silently rewrite the scheme. Signed tokens, not HTTP Date or transport security, authenticate timing. HTTP discloses the digest, nonce and network metadata and permits denial/delay; it cannot authorize an altered token. |
| Transport policy | No redirects, automatic provider fallback or hidden retries. Each retry is a new archived attempt. HTTP 200 and `application/timestamp-reply` are necessary, not sufficient. Reject truncated or excess trailing data. |
| Availability | No live request, heartbeat, timestamp or purchase was made. Geographic/network restrictions, usage permission, uptime and revocation availability require qualification. |

DigiCert also documents port 80 for this service. Its troubleshooting guidance
is not an accuracy guarantee or a service contract.
[Official troubleshooting guidance](https://knowledge.digicert.com/solution/troubleshooting-timestamping-problems).
Do not import a DigiCert Europe/qualified-service policy into this public
endpoint without explicit provider evidence linking them.

The endpoint-specific TSA policy OID is **unresolved** from the official
sources reviewed. No guessed OID, wildcard policy, or trust-on-first-response
is permitted. Obtain an authoritative service-to-policy binding and archive
it before implementation is finalized. Section 12 makes this a blocking item.

## 2. Explicit trust anchor and rotation policy

For the recommended DigiCert configuration, the sole permitted anchor is
**DigiCert Trusted Root G4**, whose certificate DER SHA-256 must be:

```text
552f7bdcf1a7af9e6ce672017f4f12abf77240c78e761ac203d1d9d20ac89988
```

Obtain it through DigiCert's official root repository and compare the actual
DER fingerprint with this value; a matching name is insufficient. Archive the
certificate and source-page snapshot. The published fingerprint is evidence
for this design; downloading and verifying the actual bundle remains a
qualification step.
[DigiCert official root repository](https://knowledge.digicert.com/general-information/digicert-trusted-root-authority-certificates).

The verifier's explicit trust store contains only that anchor. Windows/OS
trust, default OpenSSL CA paths/stores, certificates returned by the server,
and a partial chain ending at an intermediate cannot add trust. Intermediates
are untrusted chain-building inputs; archive the actual selected chain and
its DER hashes. Provider/AIA retrieval is allowed only as a bounded fetch of
public certificates, followed by full verification; it cannot install roots.

Before preparation, freeze a bundle manifest containing sorted certificate
roles, relative paths, exact file SHA-256 and DER SHA-256, source URLs and the
approved endpoint/policy binding. Hash its exact UTF-8 bytes and seal that
hash into the new plan. Archive the bundle itself privately. No implicit
online trust-store updates or changes to the frozen anchor are allowed.

Responder/intermediate rotation is acceptable only within this anchor,
the qualified provider's timestamp service and exact approved TSTInfo policy,
with every validation rule still satisfied. Archive each new chain; do not pin
only a short-lived leaf. The provider qualification must document how an
authorized timestamp service is recognized under this root; a subject-name
string or the root alone does not authorize another subscriber's TSA. An
unrecognized service, new root or policy requires a prospective profile
decision/new run, never an in-run relaxation.

## 3. Certificate and cryptographic checks

RFC 3161 requires a dedicated timestamp signing key and a critical timestamp
EKU. This profile requires exactly one EKU extension, marked critical, with
exactly the single OID `1.3.6.1.5.5.7.3.8`; reject missing, noncritical,
duplicate, mixed-purpose or any-purpose EKUs. Require the timestamp signing
purpose, consistent key usage, and strict PKIX chain validation, including
constraints and unknown-critical-extension rejection.
[RFC 3161 section 2.3](https://www.rfc-editor.org/rfc/rfc3161.html#section-2.3),
[RFC 5280](https://www.rfc-editor.org/rfc/rfc5280.html).

Profile choices: accept RSA PKCS#1 v1.5 or RSA-PSS signatures with RSA keys
at least 2048 bits and SHA-256/384/512. Reject other signature families until
a prospective profile revision; this restriction fits the recommended RSA
service and is not a universal RFC requirement. PSS parameters must verify
under the pinned validator's strict policy. Non-anchor certificate signatures
must meet the same algorithm policy. An anchor's self-signature is not an
alternative source of trust to its pinned DER identity.

Verify CMS signed attributes, content type, content digest and the binding of
the actual signer certificate using ESSCertID/ESSCertIDv2. ESSCertID's legacy
SHA-1 certificate identifier is allowed only as an authenticated certificate
identifier, never as the imprint or signature digest; ESSCertIDv2 permits
SHA-256/384/512. No manual CMS signature verification.
[RFC 5816](https://www.rfc-editor.org/rfc/rfc5816.html).

Check every chain certificate, including the anchor under this profile, over
the entire conservative token interval: `notBefore <= L` and `U <= notAfter`.
Also run PKIX validation at genTime. Never substitute the editable current
local clock for issuance-time validation. Reject any failed/ambiguous check.
OpenSSL documents purpose, explicit trust, strict verification and time
controls; the future invocation must demonstrate that default stores are
disabled, rather than assuming that providing a CA file alone suffices.
[OpenSSL verification options](https://docs.openssl.org/3.5/man1/openssl-verification-options/).

## 4. Exact private commitment format

Use SHA-256, messageImprint OID **`2.16.840.1.101.3.4.2.1`**, with a 32-byte
imprint. The hashed input is exactly:

```text
ASCII("IdentityInvariant-FAS") || 0x00 || ASCII("MSU-RFC3161-V2") || 0x00 || J
```

`J` is one JSON object with exactly the keys below, all values JSON strings.
Keys are sorted by ASCII byte value; use UTF-8 without BOM, separators `,`
and `:`, no whitespace, no final newline, no escaped slashes. All keys/values
are restricted to the ASCII characters required by their grammars below;
reject duplicate/unknown keys, escapes, nulls, numbers and noncanonical bytes.
This is an application-specific canonical format, not a claim of generic JSON
canonicalization. Hash the stored bytes, not a later reconstructed object.

| Key | Exact value grammar/meaning |
| --- | --- |
| `schema` | Literal `msu-rfc3161-commitment-v2.1` |
| `protocol_version` | Literal `msu-correspondence-pilot-v2` |
| `protocol_sha256` | SHA-256 of exact finalized V2 document bytes |
| `trust_profile_version` | Literal `msu-rfc3161-v2.1` |
| `trust_profile_sha256` | SHA-256 of exact finalized bytes of this document |
| `trust_bundle_manifest_sha256` | SHA-256 of the frozen bundle manifest |
| `tool_commit` | Exact future reviewed V2 Git commit, 40 lowercase hexadecimal digits; not the historical V1 commit |
| `tool_manifest_sha256` | Frozen source/executable/dependency manifest hash |
| `run_id` | Fresh 32 CSPRNG bytes, encoded as 64 lowercase hexadecimal digits |
| `session_id` | Fresh 32 CSPRNG bytes for the designated session, same encoding |
| `session_number` | Literal `1` for T1, `2` for T2 |
| `phase` | Exactly `SESSION1_POST_LOCK_T1` or `SESSION2_PRE_START_T2`, consistent with session number |
| `plan_sha256` | Exact sealed plan-file bytes hash |
| `plan_seal_sha256` | Exact write-once seal-file bytes hash |
| `session1_lock_sha256` | Exact valid new Session-1 lock-file bytes hash, required for both phases |
| `session1_events_sha256` | Exact completed Session-1 event archive bytes hash, as bound by that lock |
| `t1_token_sha256` | Empty string for T1; selected T1 token DER hash for T2 |
| `attempt_number` | Globally increasing positive decimal integer string, no leading zero |
| `nonce_hex` | The attempt's 32 random nonce bytes, 64 lowercase hexadecimal digits |
| `salt_hex` | Separate fresh 32 CSPRNG privacy-salt bytes, same encoding |

All SHA-256 values above are exactly 64 lowercase hexadecimal digits. No
private name, measurement, image path, mapping, coordinates or ordering is
serialized here. Digests transitively bind private records without exposing
their values. Session IDs, plan/seal and lock associations must be replayed
and verified locally before constructing the envelope. Finalize documents
and tooling, then plan/seal, then commitments: no self-referential digest.
V2's prose labels `S1_POST_LOCK`/`S2_PRE_START` denote these two exact wire
phase strings; neither is accepted as an alternate encoding.

## 5. Nonce, request and durable attempt archive

Generate nonce and salt independently using the operating-system CSPRNG
(Python `secrets.token_bytes(32)` or an audited equivalent). Nonce entropy is
256 bits; redraw an all-zero value. Encode its unsigned big-endian integer
as a positive DER INTEGER; leading zero bytes in the original random value
do not change its numeric comparison. Require exact response nonce equality.
Missing/mismatched nonce is invalid, never a warning.

Before network transmission, exclusively create and durably flush the attempt
number, canonical bytes, imprint, nonce, salt and exact DER TimeStampReq.
Request version 1, SHA-256 imprint, the frozen exact `reqPolicy` OID, nonce,
`certReq=true`, no extensions. Do not issue concurrent requests. A global
append-only ledger binds each attempt to its predecessor and plan/seal.

For every attempt preserve: request and commitment bytes/hashes; phase and
number; HTTP endpoint, status, headers and raw response body (including partial
bytes); exact DER TimeStampResp; extracted token DER/hash when parseable;
signer/chain/revocation material; validator version/configuration, checks and
diagnostics; parsed time fields/bounds; final classification and reason.
Absence of a response/token is explicit, not a fabricated empty success.
Use exclusive files, hash-bound append records and durable finalization
markers. No rewrite of an attempt or earlier status. On Windows, crash-safety
and flush semantics must be tested at every persistence boundary.

Interrupted/partial attempts remain `interrupted` or `pending_evidence` and
cannot be retrospectively promoted to successful attempts. Full, already
finalized responses with temporarily unavailable verification evidence may
receive append-only verification records; they must resolve before another
request is issued, preventing selection around an earlier potentially valid
token. Lost/ambiguous ledger integrity blocks the run. Store immutable
selection/eligibility receipts with exact attempt/token/ledger hashes.

## 6. Time parsing, uncertainty and elapsed-time rule

Accept only DER GeneralizedTime UTC form `YYYYMMDDhhmmss[.fraction]Z` with a
valid Gregorian date and explicit seconds. When a fractional part is present,
its numeric value must be nonzero and its final digit must be nonzero; leading
and internal zeros are allowed. For example, `.001` is valid and `.010` is not
canonical because it ends in zero. Reject offsets, local times, omitted seconds,
commas, malformed encodings, leap-second `60`, and fractions longer than nine
digits. The nine-digit cap is a parser/resource choice, not a scientific
threshold. Use exact rational seconds; no floats or rounding for eligibility.

For a present accuracy field, interpret missing components as zero and form
`a = seconds + millis/1000 + micros/1000000`; seconds must be nonnegative and
millis/micros, if present, must be integers 1..999. These are RFC accuracy
semantics. **This profile rejects absent accuracy and zero/empty accuracy**:
no provider-policy substitute is approved here. A provider unable to supply
positive bounded accuracy is unsuitable under this profile.
[RFC 3161 section 2.4.2](https://www.rfc-editor.org/rfc/rfc3161.html#section-2.4.2).

Let `d` be fractional digit count and `q = 10^-d` seconds (whole-second time:
`q=1`). This profile adds a conservative representation allowance of one
quantum: `e = a + q`, `L = t-e`, `U = t+e`. This extra allowance is our
operational choice; fractional precision is not evidence of clock accuracy.
If qualified provider documentation contradicts the signed accuracy claim
or leaves UTC/smear interpretation unresolved, reject; do not select a smaller
bound. Archive the original fields and exact rational calculation.

Freeze a **no-leap execution window** before preparation: archive authoritative
IERS Bulletin C evidence covering the intended interval and the conversion
table/version. Both full token intervals and their intervening span must lie
within that window and contain no UTC leap adjustment. Otherwise refuse;
never extrapolate beyond the evidence horizon. Ordinary Gregorian-to-seconds
subtraction is allowed only after that check. Provider qualification must
establish that its UTC accuracy bound covers any clock-smear behavior; an
unknown smear is not zero error. A run exceeding the frozen window requires
a separate decision, not a silent table/profile change.
[IERS Bulletin C archive](https://datacenter.iers.org/availableVersions.php?id=16)
is the authoritative source for the leap-announcement material to freeze.

```text
gap_lower_bound_seconds = L2 - U1
eligible iff all verification succeeds AND gap_lower_bound_seconds >= 172800
```

For example, with `a=1`, integer-second genTime and therefore `e=2` for each
token, a center separation of 172804 seconds yields exactly 172800; 172803
does not. These are synthetic examples, not measured provider properties.
The lock precedes acquisition of T1 and Session-2 start follows validated T2;
those enforced causal relationships make the bound conservative. Tokens do
not independently observe a session or establish an exact physical gap.

Only the authoritative verifier may emit Session-2 `gap_hours`, calculated
as this validated lower bound divided by 3600 with
`gap_hours_kind=external_conservative_lower_bound`. Session 1 has null gap.
Neither rounded hours nor local UTC participates in eligibility.

## 7. Complete token validation and revocation policy

A successful HTTP response is not a valid timing token. Require all of:

1. Strict complete DER parsing, one expected signed token, supported version,
   no unknown extensions; response status exactly `granted` (0). Reject
   `grantedWithMods` as a deliberately stricter profile choice and all errors.
2. Correct CMS signature and signer binding, section 3 algorithms/EKU/purpose,
   explicit qualified chain/root and all PKIX checks.
   If TSTInfo includes a TSA name, it must identify the verified signer;
   never treat that name as a substitute for certificate validation.
3. SHA-256 imprint equals the stored commitment digest; nonce matches; the
   TSTInfo policy equals the frozen request policy exactly. No default OID.
4. Every envelope identity, phase, session, tool/profile/plan/seal, event/lock
   hash and prior-token association matches replayed local evidence. A valid
   signature on another run/phase is still invalid for this request.
5. Valid genTime, accuracy, conversion window, full-interval certificate
   validity and authenticated revocation evidence below. Contradictory time
   ordering between serialized attempts blocks progression pending resolution.
6. Durable archive and finalization, then the first-token selection rules in
   section 8. A malformed, incomplete or invalid token supplies no eligibility.

**Prospective revocation choice: mandatory complete, direct issuer-signed
base CRLs for every non-anchor certificate in the selected chain.** OCSP may
be archived as corroboration but cannot replace a missing CRL in this profile.
Reject unknown/revoked status, invalid CRL signature/issuer/scope, unsupported
indirect/delta-only CRLs and missing `nextUpdate`. Verify issuer cRLSign and
chain, applicable distribution-point coverage and critical extensions.
Require `thisUpdate <= L`, `U <= nextUpdate` and `U-thisUpdate <= 604800`
seconds; the seven-day freshness ceiling is an explicit operational choice.
Any listed certificate is rejected, regardless of an asserted later
revocation time. Anchor trust is pinned policy, not a root CRL inference.
[RFC 5280 CRL processing](https://www.rfc-editor.org/rfc/rfc5280.html#section-6.3).

Retrieve applicable CRLs during token verification and archive the signed
bytes, hashes, source, retrieval attempt and all checks. Previously archived
CRLs may be reused only if they cover the new token interval and meet the
same rules. A CRL is a periodic issuer assertion, not proof that no later
revocation can be discovered. Missing/unavailable/stale evidence leaves the
attempt pending and prevents progression. Qualification must demonstrate
that this service's chain supports this CRL policy; no OCSP fallback is
invented after Session 1.

Execution-time eligibility requires the complete checks above. Later offline
audit replays them at the archived issuance intervals with archived trust,
CRLs, policy and validator configuration. Certificate expiry today alone
does not invalidate historical verification. Missing archives do prevent a
successful audit. New authenticated compromise/revocation evidence must be
retained and investigated; an unresolved trust contradiction blocks further
analysis, with no deletion of the original decision. Report historical
verification separately from current trust status. RFC 3161 alone supplies
neither perpetual cryptographic security nor a complete long-term evidence
renewal scheme; provider disappearance need not prevent archived verification
but can prevent fresh revocation checks or new tokens.

## 8. First-token rules, reboot and failure handling

T1 is the **first valid RFC 3161 token successfully obtained after the new
Session-1 lock**. Verify the completed session and lock before any T1 request.
Select T1 once and durably bind it to its attempt; never request a preferable
replacement. Delayed T1 acquisition extends the conservative waiting bound.

T2 is the **first qualifying valid pre-Session-2 token** satisfying section 6.
Retain all valid-but-too-early attempts. Later interval-seeking requests are
allowed only because previous valid attempts were too early; failed transport
attempts may be retried as new records. Once qualifying T2 exists, it is
authoritative. No searching for a favorable token, provider, policy or error
bound. Replayed T1 cannot satisfy T2's nonce/phase/commitment bindings.

After durable T1, any number of shutdowns, sleeps, reboots, power losses or
process terminations between sessions are allowed. No monotonic process or
clock continuity is required across the gap. Replay seals, ledger, session
lock, tokens and archive on reopening. Before Session-2 start, persist the
validated eligibility receipt; consume it exactly once in the start event.
A reboot before that start does not require a new token if the selected
evidence remains valid. Interruptions during active measurement retain V2's
existing non-resumable-session rules; this design does not change them.

No network/provider at T1: keep Session 1 locked and wait. No network/provider
near 48 hours or before T2: do not start Session 2. Timeout, malformed response
or failed verification: retain the evidence and fail closed. Power loss during
an attempt preserves partial records; a new attempt cannot overwrite them.
No local-clock, screenshot, HTTP Date, GitHub anchor, manual attestation or
unapproved provider fallback. Unrecoverable local integrity loss requires a
separate disposition, not reconstruction of evidence.

## 9. Privacy and archive custody

Only the salted envelope digest and RFC 3161 fields leave the machine in TSA
requests. Certificate/CRL retrieval concerns public PKI material only. No
envelope, licensed image, measurement, candidate label, opaque mapping,
annotation coordinate or Run 1 content is transmitted. Provider/network logs
can still observe IP address, timing, digest and nonce; this is not anonymity.
Do not add credentials or identifying telemetry to this public endpoint.

Keep records under the fresh ignored root with owner-restricted access and
an encrypted local/offline backup. Backup must preserve immutable bytes and
hash manifests; restoration verifies them and never merges divergent ledgers.
Review outputs expose no mapping or scientific result through timing errors.
Do not commit private artifacts or upload them to GitHub. Optional hash
corroboration is not selected for this profile and cannot authorize eligibility.

## 10. Preferred implementation strategy, without implementation

Use Python for deterministic records, persistence, request orchestration and
semantic checks; mature ASN.1 support for typed request/token fields; and
OpenSSL TS/CMS/X.509 verification for cryptography. A small audited transport
and semantic wrapper is preferable to hand-written ASN.1/CMS verification.
Use a pinned supported OpenSSL build with explicit configuration, trust
store and process environment; archive executable/library hashes.

OpenSSL `ts -verify` supports request-bound verification. Its CLI query
generator documents a **64-bit default nonce**, insufficient for this profile's
256-bit requirement: use an audited ASN.1 encoder or libcrypto request API to
set the full nonce, and verify the produced DER before sending. Do not mistake
CLI defaults or an exit code alone for the complete application verifier.
[OpenSSL 3.5 timestamp documentation](https://docs.openssl.org/3.5/man1/openssl-ts/).

Exact dependency versions, request API and tested invocation are implementation
specification blockers. Preserve one authoritative verifier for session start,
analysis CLI and public API; no raw-record bypass. The specification must add
synthetic golden canonical-byte/request vectors and demonstrate equivalent
checks through every entry point before any real execution is authorized.

## 11. Required synthetic/offline acceptance tests

Use generated test keys, certificates, CRLs and synthetic locks only; test
roots must be impossible to enable in production configuration. No candidate
image or Run 1 measurement is a fixture.

| Area | Required cases and expected behavior |
| --- | --- |
| Valid path | Known synthetic request/token, full chain/CRLs and conservative interval accepted; independent archived replay agrees. |
| Commitment | Golden byte/hash vectors; changed whitespace, duplicate key, wrong imprint/algorithm, run/session/phase/lock/tool/plan binding rejected; T1 replay as T2 rejected. |
| Nonce | Missing, wrong, reused nonce rejected; leading-zero unsigned encoding matches correctly; 256-bit generation verified. |
| Encoding/status | Malformed/truncated/trailing DER, multiple/absent token, unsupported extensions, nonzero status and HTTP mismatch rejected. |
| Trust/signature | Untrusted root, implicit OS-store trust, partial-chain shortcut, invalid signature or signer-certificate binding rejected. |
| Certificate purpose/time | Missing/noncritical/mixed EKU; invalid key usage; expired/not-yet-valid signer or intermediate, including uncertainty crossing validity boundary, rejected. |
| Revocation | Revoked/unknown, wrong issuer/scope/signature, missing/stale CRL, absent nextUpdate, unsupported delta/indirect CRL rejected; unavailable evidence blocks progression; OCSP cannot override. |
| Time | Missing/zero/negative/invalid accuracy; fractional genTime; offsets; malformed dates; excess precision; leap/smear ambiguity; expired no-leap window. All invalid inputs fail closed. |
| Boundary | Exactly 172800 conservative seconds accepted; any rational amount below rejected; 48-hour center separation alone insufficient with positive uncertainty. |
| Rotation | New authorized signer/intermediate under frozen root/policy accepted; new root, service or policy rejected. |
| Selection | Earlier valid T1 cannot be skipped; valid early T2 retained; first qualifying T2 cannot be replaced; unresolved earlier response blocks cherry-picking. |
| Crash/network | Power loss at each write boundary; reboot/offline interval between T1/T2; local clock jumps; T1/T2 network failure, timeout and provider disappearance; no fallback or overwrite. |
| State/API | Stopped/incomplete/unlocked session, altered seal/ledger, duplicate start, premature analysis and direct raw-session API rejected. All six V1 integrity fixes remain tested. |
| Privacy/science | Captured synthetic outbound traffic contains only permitted fields; no labels in reviewer error paths; P4/readiness unchanged; no TEST or PAD path invoked. |

Also test resource limits, immutable archive restoration, chain-policy binding,
late evidence contradictions and offline reproducibility with the live TSA
unavailable. All results in this design remain `not_executed`.

## 12. Remaining blockers and freeze sequence

The serialization, nonce, root identity, strict validation, CRL-only policy,
missing-accuracy rejection and conservative interval rule above are resolved
design choices. They must not be weakened to fit an observed response.

Before finalizing implementation, obtain authoritative endpoint-specific
policy OID/service authorization, research-use/access/cost terms, and UTC
accuracy/smear evidence; qualify availability of suitable complete CRLs.
The reviewed provider pages do not settle those facts. Download/verify/archive
the explicit root bundle and provider policy snapshots. If DigiCert cannot
meet the profile, evaluate another official provider prospectively with a
reviewed document revision; it is not an automatic fallback.

Freeze the request/verification API, supported dependency builds, exact
transport bounds and durable archive schema in the implementation specification.
Obtain synthetic-only endpoint qualification authorization separately; this
document makes no live requests and does not invent observed token properties.
Require byte-level test vectors and the full test matrix before release.

Before new preparation, archive the IERS evidence/conversion window, finalize
and hash protocol/profile/tool/bundle, confirm private backup and reviewer
exposure disclosure, then seal the new plan. Obtain separate execution
authorization. No real T1/T2, session or analysis is authorized by this design.
V1 and historical evidence remain unchanged; Run 1 remains excluded forever.

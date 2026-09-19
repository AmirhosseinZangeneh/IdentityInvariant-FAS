# Pre-Protocol Lock State

## IdentityInvariant-FAS Research Snapshot

Date:
2026-09-19

Repository:
IdentityInvariant-FAS

Branch:
main

Commit:
bd18725f8774c614aecf5d75613e1176ca611de4


---

# 1. Purpose of This Document

This document records the exact scientific and implementation state of the IdentityInvariant-FAS repository before any further reconstruction, modification, or experimental redesign.

The goal is to preserve a reproducible snapshot of:

- current implementation status
- available datasets and infrastructure
- implemented models
- experimental capabilities
- known uncertainties

No scientific conclusion about the final contribution is made at this stage.


---

# 2. Current Research Direction

The current research direction investigates the relationship between:

- identity-related information in Face Anti-Spoofing representations
- PAD decision behavior
- generalization under subject/domain shift


The current working hypothesis is:

Identity-related information may exist inside FAS representations and may influence model behavior. Reducing unnecessary identity dependence may improve robustness while preserving spoof-discriminative information.


This hypothesis is not considered validated at this stage.


---

# 3. Current Repository State

The repository currently contains:

## Models

### PaperECNN

Purpose:

Baseline implementation inspired by the ECNN baseline used in the original FAS study.


Status:

Implemented.

Scientific fidelity:

Requires verification against the original paper.


---

### IdentityInvariantECNN

Purpose:

Identity-aware extension of ECNN using adversarial subject suppression.


Main components:

- ECNN encoder
- spoof classification head
- subject classification head
- Gradient Reversal Layer


Status:

Implemented.


---

### AblationECNN

Purpose:

Controlled comparison for evaluating the effect of adversarial identity suppression.


Design goal:

Keep architecture comparable while removing the GRL mechanism.


Status:

Implemented.


---

# 4. Dataset Availability

## NUAA

Status:

Available inside repository.


Current usage:

Subject-disjoint evaluation pipeline exists.


Verification required:

- exact correspondence with official protocol
- split reproducibility
- preprocessing consistency


---

## MSU-MFSD

Status:

Dataset infrastructure exists.


Available:

- adapters
- preprocessing pipeline
- evaluation support


Verification required:

- complete experiment execution
- protocol alignment


---

## Replay-Attack

Status:

Infrastructure exists.


Dataset availability:

Requires verification due to dataset access restrictions.


---

# 5. Current Experimental Capabilities

The repository currently supports evaluation of:

## Representation Analysis

Question:

Can identity information be recovered from learned representations?


Available approach:

Identity probing.


---

## Identity Suppression Analysis

Question:

Does adversarial suppression reduce identity-related information?


Available comparison:

- ECNN baseline
- IdentityInvariantECNN
- AblationECNN


---

## Generalization Analysis

Question:

Does identity-related suppression improve robustness?


Potential evaluation:

- subject-disjoint evaluation
- cross-dataset evaluation


Current status:

Requires complete experimental execution.


---

# 6. Known Scientific Risks

The following issues require investigation before claiming contribution:

## 6.1 Baseline Fidelity

The current ECNN implementation has not yet been fully verified against the original paper regarding:

- architecture details
- preprocessing
- training procedure
- evaluation protocol


---

## 6.2 Identity Leakage Interpretation

Identity predictability does not automatically prove:

- identity causes PAD errors
- identity is harmful
- identity suppression improves generalization


Additional controlled experiments are required.


---

## 6.3 Generalization Claims

No claim regarding improved cross-domain generalization should be made before completing controlled experiments.


---

# 7. Reconstruction Principle

Any future modification should prioritize:

1. scientific validity
2. reproducibility
3. controlled comparisons
4. clear separation between:
   - representation information
   - decision dependence
   - generalization effects


Implementation changes should not be introduced before completing the necessary audit.


---

# 8. Next Planned Investigation

The next investigation is:

Paper Reproduction Fidelity Audit


Objectives:

- verify whether the current ECNN implementation faithfully reproduces the baseline paper
- identify missing components
- identify implementation deviations
- determine required reconstruction steps


---

# 9. Frozen Reference

Current repository snapshot:

Commit:

bd18725f8774c614aecf5d75613e1176ca611de4


This commit represents the baseline state before further research reconstruction.
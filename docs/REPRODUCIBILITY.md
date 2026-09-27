# Reproducibility Checklist

Before generating final manuscript tables:

The [prospective publication matrix](PUBLICATION_EXPERIMENT_MATRIX.md) fixes
the required experiments, training-only selection, seeds and reporting rules.
Its frozen design does not waive dataset or execution gates.

1. Install the project in editable mode.
2. Record Python, PyTorch, CUDA, and GPU versions.
3. Use the same image size and transforms for all compared models.
4. Keep the outer test groups completely untouched during model selection.
5. Use subject-disjoint validation for the strict identity-generalization
   protocol only with verified human IDs. For NUAA, current splits are
   folder-token-disjoint and support proxy claims only; see
   [NUAA scientific role](NUAA_SCIENTIFIC_ROLE.md).
6. Keep optimizer, learning rate, warm-up schedule, batch size, and epoch
   budget identical across GRL lambda values.
7. Use the matched three-arm controlled ablation for prospective comparisons;
   NUAA auxiliary labels remain folder-token proxies.
8. Run the prespecified verified-client probes at the training image size;
   t-SNE and other descriptive representation plots are optional, not evidence
   of identity suppression by themselves.
9. Report fold-level values as well as mean and standard deviation.
10. Report matched-seed effects and the matrix's paired client/seed bootstrap
    for MSU; do not use image/frame-iid uncertainty for correlated observations.
    NUAA reports seed variability without image-iid confidence intervals.
11. For video datasets, report video-level metrics after a pre-declared score
    aggregation rule.
12. Keep dataset licenses and raw media outside Git.

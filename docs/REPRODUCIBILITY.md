# Reproducibility Checklist

Before generating final manuscript tables:

1. Install the project in editable mode.
2. Record Python, PyTorch, CUDA, and GPU versions.
3. Use the same image size and transforms for all compared models.
4. Keep the outer test subjects completely untouched during model selection.
5. Use subject-disjoint validation for the strict identity-generalization
   protocol.
6. Keep optimizer, learning rate, warm-up schedule, batch size, and epoch
   budget identical across GRL lambda values.
7. Re-train the corrected architecture-matched ablation model.
8. Re-run identity leakage and t-SNE/representation metrics at the same image
   size used during training.
9. Report fold-level values as well as mean and standard deviation.
10. Treat five-fold significance tests as exploratory; prefer paired
    sample-level bootstrap on a fixed test set when available.
11. For video datasets, report video-level metrics after a pre-declared score
    aggregation rule.
12. Keep dataset licenses and raw media outside Git.

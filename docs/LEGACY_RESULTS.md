# Legacy NUAA Development Results

These numbers are preserved only as a development checkpoint from the
pre-refactor implementation. They are **not final paper results** because the
scientific audit identified protocol/preprocessing inconsistencies that require
selected experiments to be re-run.

## Five-fold FAS results recorded in the legacy workspace

| Model | Mean ACER | Std ACER |
|---|---:|---:|
| Paper ECNN | 0.06037 | 0.11230 |
| II-ECNN | 0.05161 | 0.09443 |
| Historical Ablation | 0.03630 | 0.02750 |

The large fold variance, especially for the baseline and II-ECNN, means the
small average difference between Paper ECNN and II-ECNN should not be described
as statistically established.

Using the recorded five fold values, a two-sided Wilcoxon test for Paper ECNN
vs II-ECNN gives p=0.625. This supports treating the current performance
difference as descriptive rather than significant.

## Legacy identity leakage probe

| Model | Accuracy |
|---|---:|
| Paper ECNN | 0.6963 |
| II-ECNN, lambda=0.01 | 0.6451 |
| II-ECNN, lambda=0.05 | 0.5930 |
| II-ECNN, lambda=0.10 | 0.7411 |
| Historical Ablation | 0.8105 |

The lambda=0.05 result was the strongest identity suppression in the recorded
development run, but this analysis used a 64x64 representation input while the
models were trained using 160x160 inputs. Re-run with matched preprocessing.

## Interpretation

The development evidence is promising for the identity-suppression hypothesis,
but it does not yet establish superiority over the baseline. The strongest
scientific claim supported at this stage is that the project has a testable
identity-invariance mechanism and preliminary evidence of reduced linearly
decodable identity information. Cross-dataset results and corrected controlled
ablations are still required.

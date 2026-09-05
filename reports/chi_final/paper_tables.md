# HyperTrace CHI Aggregate Results

## Strict Unseen-Operation Evaluation

| Target operation | AUROC | AUPRC |
|---|---:|---:|
| Honduras | 0.8850 +/- 0.0842 | 0.4677 +/- 0.3578 |
| Uae | 0.9005 +/- 0.0312 | 0.5559 +/- 0.2650 |

## Label-Blind Rank Consensus at 1% Review Budget

| Target operation | Precision | Recall | Lift | Pairwise Jaccard |
|---|---:|---:|---:|---:|
| Honduras | 0.1050 | 0.1265 | 12.65x | 0.3554 |
| Uae | 0.1679 | 0.3365 | 33.64x | 0.4539 |

## Mechanism Ablation (15 paired scenario/seed folds per variant)

| Variant | AUPRC | AUROC | Macro-F1 | Balanced accuracy | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Lorentz-HGT + real warm-start | 0.9791 +/- 0.0082 | 0.9944 +/- 0.0030 | 0.9296 +/- 0.0171 | 0.9631 +/- 0.0177 | 0.0116 +/- 0.0027 | 0.0182 +/- 0.0057 |
| Lorentz-HGT (scratch) | 0.9792 +/- 0.0080 | 0.9938 +/- 0.0034 | 0.9205 +/- 0.0348 | 0.9538 +/- 0.0350 | 0.0129 +/- 0.0051 | 0.0201 +/- 0.0081 |
| Euclidean-HGT (scratch) | 0.9836 +/- 0.0054 | 0.9970 +/- 0.0013 | 0.9238 +/- 0.0130 | 0.9741 +/- 0.0062 | 0.0126 +/- 0.0022 | 0.0129 +/- 0.0049 |
| Lorentz-HGT, no temporal input | 0.9776 +/- 0.0083 | 0.9927 +/- 0.0044 | 0.9301 +/- 0.0159 | 0.9643 +/- 0.0123 | 0.0116 +/- 0.0026 | 0.0194 +/- 0.0045 |

## Interpretation Boundaries

- The scratch Euclidean model is higher on AUPRC, AUROC, balanced accuracy, and ECE; hyperbolic geometry is not claimed to universally improve predictive accuracy.
- Temporal input has a small positive AUROC contribution; its paired intervals for the other reported metrics cross zero.
- The real-graph Lorentz warm-start does not change AUPRC materially; improvements in F1, balanced accuracy, and calibration are trends whose paired intervals cross zero.
- Explanation packets passed provenance and label-blind preflight, but the formal explanation sample and participant study remain separate required work.
- Psychological features and role classification are excluded from every result in this package.

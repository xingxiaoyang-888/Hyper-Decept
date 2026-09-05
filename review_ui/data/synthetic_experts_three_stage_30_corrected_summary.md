# HyperTrace Three-Stage Reviewer Check: Corrected Summary

> Corrected condition-level summary based on the user's re-measured results. This file does not regenerate participants or alter the existing trial-level records.

## Descriptive results

All conditions contain 10 participants and 80 reviews. Each condition contains 50 correct-recommendation trials and 30 incorrect-recommendation trials.

| Condition | Initial accuracy | Final accuracy | Initial-to-final change | Correct-AI acceptance | Incorrect-AI rejection | Latency median (IQR) |
|---|---:|---:|---:|---:|---:|---:|
| Risk-only | 62.5% (50/80) | 67.5% (54/80) | +5.0 pp | 92.0% (46/50) | 26.7% (8/30) | 47.4 s (42.3-53.7) |
| Standard-signals | 61.3% (49/80) | 75.0% (60/80) | +13.8 pp | 88.0% (44/50) | 53.3% (16/30) | 67.5 s (60.1-74.9) |
| HyperTrace evidence | 61.3% (49/80) | 85.0% (68/80) | +23.8 pp | 96.0% (48/50) | 66.7% (20/30) | 75.9 s (62.3-87.2) |

## Direct contrasts

- HyperTrace vs Risk-only final accuracy: **+17.5 percentage points** (85.0% - 67.5%).
- HyperTrace vs Standard-signals final accuracy: **+10.0 percentage points** (85.0% - 75.0%).
- HyperTrace vs Risk-only incorrect-recommendation rejection: **+40.0 percentage points** (66.7% - 26.7%).
- HyperTrace vs Standard-signals incorrect-recommendation rejection: **+13.3 percentage points** (20/30 - 16/30).
- Latency ratio, Standard-signals / Risk-only: **1.42x** (67.5 / 47.4).
- Latency ratio, HyperTrace / Risk-only: **1.60x** (75.9 / 47.4).
- Latency ratio, HyperTrace / Standard-signals: **1.12x** (75.9 / 67.5).

## Scope and unavailable recomputations

The corrected values above are condition-level summaries supplied after re-measurement. They are not a replacement 240-row trial-level dataset. The existing `synthetic_experts_three_stage_30.csv`, participant-point file, error-bar file, and inferential outputs were generated from an earlier synthetic record and do not match the corrected initial-accuracy and latency summaries.

The following therefore remain **not recomputed** from the corrected data: RAIR, RSR, participant-level points and error bars, final-accuracy and rejection bootstrap intervals, crossed GLMM and sensitivity-model results, adjusted probabilities, latency-model confidence intervals and p-values, Holm-adjusted contrasts, and convergence/singular-fit diagnostics. These require the corrected trial- or participant-level records.

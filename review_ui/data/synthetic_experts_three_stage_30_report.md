# HyperTrace Three-Stage Synthetic Reviewer Check (30 participants)

> Synthetic template-validation data only; not human-subject findings.

All values below were recomputed from the combined 240 trial-level records; no old percentages were averaged.

## Descriptive results

| Condition | Participants | Reviews | Initial | Final | Improvement | Correct-AI acceptance (50) | Incorrect-AI rejection (30) | RAIR (n) | RSR (n) | Latency median (IQR) | Workload mean (SD) | Confidence change (SD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| risk_only | 10 | 80 | 67.5% | 67.5% | +0.0% | 92.0% | 26.7% | 77.8% (18) | 36.4% (22) | 47.4s (42.3-53.7) | 2.70 (0.82) | +2.30 (14.20) |
| standard_signals | 10 | 80 | 70.0% | 75.0% | +5.0% | 88.0% | 53.3% | 88.2% (17) | 60.9% (23) | 77.5s (65.5-89.7) | 3.60 (0.84) | +4.51 (12.36) |
| hypertrace_evidence | 10 | 80 | 61.3% | 85.0% | +23.7% | 96.0% | 66.7% | 94.7% (19) | 83.3% (18) | 101.9s (92.3-127.2) | 4.70 (0.82) | +11.18 (12.58) |

## Inferential results

```json
{
  "condition_bootstrap_ci": {
    "risk_only": {
      "initial_accuracy": {
        "ci": [
          0.5875,
          0.7625
        ]
      },
      "final_accuracy": {
        "ci": [
          0.5875,
          0.7625
        ]
      },
      "incorrect_rejection": {
        "ci": [
          0.13333333333333333,
          0.43333333333333335
        ]
      }
    },
    "standard_signals": {
      "initial_accuracy": {
        "ci": [
          0.65,
          0.75
        ]
      },
      "final_accuracy": {
        "ci": [
          0.65,
          0.85
        ]
      },
      "incorrect_rejection": {
        "ci": [
          0.3333333333333333,
          0.7333333333333333
        ]
      }
    },
    "hypertrace_evidence": {
      "initial_accuracy": {
        "ci": [
          0.5,
          0.725
        ]
      },
      "final_accuracy": {
        "ci": [
          0.7625,
          0.925
        ]
      },
      "incorrect_rejection": {
        "ci": [
          0.4666666666666667,
          0.8333333333333334
        ]
      }
    }
  },
  "final_accuracy_differences": {
    "risk_only": {
      "difference": 0.17499999999999993,
      "ci": [
        0.04999999999999993,
        0.29999999999999993
      ]
    },
    "standard_signals": {
      "difference": 0.09999999999999998,
      "ci": [
        -0.025312499999999738,
        0.22531249999999553
      ]
    }
  },
  "incorrect_rejection_differences": {
    "risk_only": {
      "difference": 0.39999999999999997,
      "ci": [
        0.13333333333333336,
        0.6333333333333334
      ]
    },
    "standard_signals": {
      "difference": 0.1333333333333333,
      "ci": [
        -0.1333333333333333,
        0.4
      ]
    }
  },
  "latency_ratios": {
    "risk_only": {
      "ratio": 2.1501446032382683,
      "ci": [
        2.067192820127374,
        2.489321122841344
      ]
    },
    "standard_signals": {
      "ratio": 1.3149387744563288,
      "ci": [
        1.2462496070418108,
        1.565748469472888
      ]
    }
  },
  "binary_model": {
    "chi2": 5.932919908619029,
    "p": 0.05148524823881626,
    "contrasts": {
      "standard_signals": {
        "aor": 0.1809091123447535,
        "ci": [
          0.04424285477712674,
          0.7397376840675036
        ]
      },
      "hypertrace_evidence": {
        "aor": 0.27508067160277366,
        "ci": [
          0.056261728796884986,
          1.3449529103987032
        ]
      },
      "hypertrace_vs_standard": {
        "aor": 1.5205462457776038,
        "ci": [
          0.3265556125972255,
          7.080144380798214
        ]
      }
    }
  },
  "time_model": {
    "standard_signals": {
      "ratio": 1.590876448990056,
      "ci": [
        1.5134431834472972,
        1.6722714824261808
      ],
      "p": 2.6123239842817882e-74
    },
    "hypertrace_evidence": {
      "ratio": 2.2369203897535757,
      "ci": [
        2.2042304033272027,
        2.2700951872101127
      ],
      "p": 0.0
    },
    "hypertrace_vs_standard": {
      "ratio": 1.4060930948934791,
      "ci": [
        1.320537696709462,
        1.4971914822527883
      ],
      "p": 1.9212432926873684e-26
    }
  },
  "secondary_associations": {
    "geometry": {
      "aor": 1.1433245964843946,
      "ci": [
        0.7465280909399631,
        1.7510273877039273
      ],
      "p": 0.5379790116038856
    },
    "sufficiency": {
      "aor": 0.9372673718184726,
      "ci": [
        0.4887326409319502,
        1.797445172887118
      ],
      "p": 0.8453844411731195
    }
  }
}
```

## Participant-level data

`synthetic_experts_three_stage_30_participant_points.csv` contains all participant-level points; `synthetic_experts_three_stage_30_error_bars.csv` contains 10,000-draw participant-cluster bootstrap intervals for plotted accuracy and rejection metrics.

## Provenance

The first 160 trial values are retained from the prior 20-participant CSV; rows 161-240 are the ten appended synthetic participants.

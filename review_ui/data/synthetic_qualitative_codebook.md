# HyperTrace Synthetic Qualitative Codebook

> This codebook validates the Section 7.5 analysis structure. It is applied to generated text and does not constitute human-subject evidence.

## Units and counting

- Primary unit: one trial-level final rationale.
- Secondary unit: one participant-level post-task comment.
- Trial frequency counts each rationale once per code, even if the behavior is mentioned repeatedly.
- Participant frequency counts a participant once when the code occurs in any of their rationales or comment.
- Codes are non-exclusive.

## Codes

| Code | Include when the account states | Exclude when |
|---|---|---|
| `sequence_reconstruction` | The reviewer compares event order, timing, activity windows, or a temporal pattern. | Time is mentioned only as case metadata. |
| `provenance_verification` | The reviewer reports checking a source identifier, underlying record, or timestamped source. | The reviewer only notes that provenance is available. |
| `initial_final_comparison` | The reviewer explicitly compares the recommendation or evidence with their initial judgment. | The final decision is stated without comparison. |
| `geometry_audit` | Geometry fidelity is interpreted as preservation of detector structure or the reduced packet. | Geometry is treated as proof of correctness; use `metric_confusion`. |
| `counterfactual_use` | The account compares keep-only, removed-edge, or full-neighborhood outcomes. | The account only mentions a prediction score. |
| `risk_priority_only` | The account relies on priority or percentile without reconstructing evidence. | Priority is used together with a mechanism-level check. |
| `unresolved_evidence` | Missing, conflicting, or incomplete evidence leads to qualification, rejection, or a request for more evidence. | The missing field is ignored. |
| `evidence_overload` | The account reports excessive evidence, comparison difficulty, or increased mental effort. | Longer review is described without cognitive burden. |
| `metric_confusion` | Geometry fidelity is interpreted as model accuracy, confidence, or proof that coordination occurred. | Geometry is correctly interpreted as a preservation diagnostic. |
| `scope_warning_missed` | The account explicitly states that the snapshot warning was not used, or omits it when the generated rationale marks it as decision-relevant. | The scope boundary is acknowledged or irrelevant. |

## Reliability boundary

The generated codes are assigned by deterministic simulation rules, so inter-coder reliability would be tautological and is not reported. The genuine qualitative analysis must document its human coding process and report agreement or reconciliation according to the study protocol.

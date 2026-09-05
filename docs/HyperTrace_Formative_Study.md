# 3 Formative Study

## 3.1 Study Aim and Materials

We conducted a formative study to understand how participants reason about suspected coordinated information operations (CIOs), what makes a coordination claim inspectable, and how review support can preserve independent judgment. The study informed HyperTrace design rather than evaluating a finished interface.

The analysis package contains anonymized, structured written responses from ten participants (P01–P10), with approximately two years of research experience on average. The study was reviewed and approved by the institutional review board (IRB) of the authors’ college. We analyzed the complete responses to Q3–Q6 as 40 response-level meaning units; Q2 was used only as background. Names, contact information, IP addresses, and submission metadata were removed. Quotations are translated from the original Chinese responses.

## 3.2 Coding and Analysis

Two researchers independently performed AI-assisted open coding on the same anonymized material and checked their coding outputs. They then compared codes item by item, merged synonymous labels, and resolved differences in coding granularity or theme boundaries through consensus. The analysis combined inductive coding with a four-theme analytic structure (F1–F4). The complete subtheme mapping and audit trail are provided in the supplementary materials. We did not use code frequency as a prevalence estimate and do not claim fully manual coding, statistical inter-rater reliability, or thematic saturation.

## 3.3 Findings

Q3–Q6 respectively informed F1–F4, and all ten participants answered each question. Each finding therefore draws on P01–P10; the identifiers below show which participants contributed the specific points highlighted in the text, not prevalence estimates.

### F1. Coordination judgments require mechanism-level evidence, not risk aggregation

Participants distinguished jointly high-risk accounts from an operation coordinated through a shared mechanism. Content similarity or one synchronized burst was a clue, not a conclusion. They combined repeated timing and shared targets (P01, P02, P06, P10) with graph structure, account or device links, and common generation templates (P03–P08), while testing benign alternatives such as news events or interest communities. P01 stated: “You cannot look only at whether these accounts’ content is similar. You also need to consider posting times, retweet targets, interaction relationships, and behavioral patterns.” HyperTrace therefore presents converging relational and temporal evidence rather than a single salient signal.

### F2. Static prediction explanations do not cover adaptive coordination

Participants described three connected failures: local explanations miss harms emerging across otherwise-normal posts and social context (P01, P02, P06, P10); adversarial noise or graph camouflage can redirect attribution (P03, P04); and older patterns can become stale under distribution shift (P08). P07 and P09 further warned that slow, technical, or dense displays become unusable at review scale. As P08 noted, “Even if XAI provides an explanation, it may be using old data patterns to explain a new attack method.” HyperTrace consequently exposes scope, uncertainty, and explanation limits without requiring inspection of the entire graph at once.

### F3. Explanation claims must be traceable item by item to source records

Participants expected each important claim to resolve to an account, post, relation, timestamp, or behavioral log (P01–P06), while remaining accessible without overwhelming the reviewer (P07, P09). They also requested counterfactual checks (P08) and preserved evidence chains for appeals or external review (P05, P10). P03 stated: “Every node and edge in the explanation should be traceable back to the original data.” HyperTrace therefore provides item-level provenance, cutoff metadata, source links, and a clear distinction between observed records and model-derived evidence.

### F4. Dynamic review requires explicit and reversible explanation updates

Participants wanted new evidence to appear as an explicit update rather than silently replacing the previous assessment. They requested version histories and graph differences (P01, P03, P05, P06), changes in risk or uncertainty (P04, P07, P08, P10), and non-disruptive reminders with rollback (P02, P09). P05 explained: “In addition to showing the current conclusion, the system should retain the previous judgment, the rules and evidence used at that time, and why it later changed.” HyperTrace records evidence differences, highlights substantive changes, and leaves escalation and final judgment to the reviewer.

## 3.4 From Findings to Design Requirements

| Finding | Design requirement | HyperTrace response | Later evaluation |
|---|---|---|---|
| F1: Mechanism-level coordination evidence | Expose converging behavioral, relational, and temporal signals while preserving alternatives. | Typed evidence subgraph, event sequence, shared targets, and mechanism-level evidence labels. | Evidence verification and final decision accuracy. |
| F2: Adaptive and cognitively difficult explanations | Make scope, uncertainty, and explanation limits visible without overwhelming reviewers. | Cutoff and scope state, compact evidence budget, warnings, and progressive disclosure. | Temporal validity, sparsity, stability, and workload. |
| F3: Source-verifiable explanation claims | Provide item-level provenance and support independent challenge. | Evidence IDs, source records, timestamps, relation direction, and source-verification actions. | Provenance coverage and verification accuracy. |
| F4: Explicit and reversible updates | Preserve changes, reasons, and uncertainty across review states. | Version history, graph differences, changed margins, reminders, and rollback actions. | Update traceability and erroneous-recommendation rejection. |

## 3.5 Methodological Scope

This study provides design evidence, not a prevalence estimate or a claim that all platform reviewers reason in the same way. Its conclusions are bounded by the sample, structured written-response format, and research context. HyperTrace is evaluated against these identified needs, but this formative study does not establish that the system improves reviewer accuracy; that question is addressed separately through the technical explanation evaluation and controlled human review study.

# Introduction Citation Map

This map records the recommended citation keys for the empty `\\cite{}` slots in the supplied Introduction. It also marks claims that should be treated as the paper's own definition or formative-study finding rather than attributed to prior work.

| Introduction claim | Recommended keys | Evidence boundary |
|---|---|---|
| Coordination operates across accounts, content, relations, and time | `cima2024coordinated`, `graham2024toolkit`, `minici2024crossplatform` | These works analyze operation-level coordination networks and cross-platform traces. |
| Account-level signals are insufficient for networked coordination | `cima2024coordinated`, `zouzou2024fakefollowers`, `burghardt2024sociolinguistic` | Use “insufficient on their own,” not “irrelevant.” |
| Coordination is not by itself a policy violation | `graham2024toolkit`, `rogers2025manufactured` | Graham explicitly discusses online activism; Rogers documents ambiguity in applying CIB categories. |
| Legitimate organizations, advocacy groups, and emergency communities may coordinate | `graham2024toolkit` | The paper discusses online activism. Add an emergency-response source only if the sentence specifically discusses emergency response. |
| Suspicious operations can mix human and automated accounts | `zouzou2024fakefollowers`, `burghardt2024sociolinguistic`, `piao2025social`, `jin2025synthetic` | The first two support mixed/coordination patterns; the latter two support LLM-agent behavior. |
| Automation or coordination does not establish harmful intent or deceptive identity | `rogers2025manufactured` | This is partly a scope definition for this paper; do not present it as a universal legal definition. |
| CIB is used only when deceptive identity, concealed orchestration, or platform enforcement is established | `rogers2025manufactured`, `luceri2026tiktok` | Keep “in this paper” in the sentence; platform definitions vary. |
| Evidence is distributed across accounts and interactions | `cima2024coordinated`, `graham2024toolkit`, `minici2024crossplatform` | These support network-level and cross-platform evidence, not the paper's interview findings. |
| Synchronization, repeated interaction, and cross-account diffusion jointly support operation-level assessment | `cima2024coordinated`, `graham2024toolkit`, `zouzou2024fakefollowers` | All three discuss multi-behavior or anomalous coordination signals. |
| Mechanism, cutoff, and source verification are reviewer requirements | **Do not force a prior-work citation.** | Attribute this to the formative interviews and the HyperTrace design requirements. |
| Risk score or salient subgraph cannot establish all evidentiary conditions | `agarwal2023evaluating`, `schemmer2023appropriate` | Rewrite as a limitation/argument, not as a direct empirical finding of these papers. |
| Coordinated networks have hierarchical and multiscale structure | `choudhary2023hyperbolic`, `yang2024hypformer`, `park2024hyperbolic`, `park2024multihyperbolic` | These support hierarchical/multiscale hyperbolic graph representation. |
| Hyperbolic representations are used for graph learning and social-bot detection | `choudhary2023hyperbolic`, `yang2024hypformer`, `lu2026sahg` | SAHG must remain labeled as an arXiv preprint. |
| Existing representation studies primarily report predictive performance | `choudhary2023hyperbolic`, `yang2024hypformer`, `lu2026sahg` | Phrase as “primarily” or “largely,” not an exhaustive universal claim. |
| Geometry fidelity means correspondence between full-graph and evidence-subgraph geometry | **No external citation required.** | This is HyperTrace's operational definition and must be defined in Methods. |
| Visual coherence concerns reviewer-facing appearance, not geometric preservation | `wang2024humanexplain` | Use as human-centered explanation-evaluation context, not proof of a geometry distinction. |
| Lorentz-HGT models heterogeneous account/content relations in Lorentz space | `hu2020hgt`, `yang2024hypformer`, `park2024hyperbolic`, `lu2026sahg` | `hu2020hgt` is the foundational heterogeneous-transformer citation; the actual `DomainAwareLorentzHGT` implementation is this paper's system component. |

## Suggested Replacements

For the empty citation after the review-requirement sentence, replace the citation with:

> These requirements emerged from our formative interviews and are formalized as HyperTrace design requirements.

For the geometric-fidelity sentence, replace the citation with:

> In this paper, we define geometric fidelity as the correspondence between the detector's full-graph decision geometry and that preserved by the evidence subgraph.

For the visual-coherence sentence, use:

> Visual coherence concerns whether the explanation is legible and interpretable to a reviewer, whereas geometric fidelity concerns whether it preserves the detector's decision structure (Wang, 2024).


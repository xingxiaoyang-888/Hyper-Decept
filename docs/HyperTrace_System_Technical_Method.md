# 5 System and Technical Method

HyperTrace is an evidence-centered review workflow built around a frozen coordination detector. The detector supplies a preliminary assessment; the workflow reconstructs the records needed to inspect and challenge that assessment. Thus, the system is not an additional prediction head or a visualization of all internal activations. It is a contract between a detector, a constrained evidence reconstruction procedure, and a reviewer-facing audit interface.

## 5.1 Workflow and Scope

For a case under review, the workflow receives an observed heterogeneous graph (G_{≤ t}), a frozen detector (f_θ), and a target account or local account neighborhood (v). (G_{≤ t}) is either bounded by an explicit observation range or constructed by retaining only records available at a declared decision cutoff (t). Its nodes represent accounts and observable content or activity objects; typed edges represent the relations supplied by the dataset manifest, such as posting, sharing, replying, mentioning, or other observed interactions. The workflow never treats a derived relation as an original record: each derived edge retains the source events from which it was constructed.

The output is an `ExplanationPacket` with seven components:

```text
ExplanationPacket = {
  review priority and model recommendation,
  compact typed evidence subgraph,
  temporal scope,
  geometry summary,
  counterfactual effect,
  provenance links and hashes,
  warnings and review state
}
```

This compact schema is sufficient for the main text; the complete machine-readable field schema and validation rules belong in the supplementary material.

The detector output is used to rank cases for review. We report it as a dataset-relative percentile: for example, the 95th percentile means that a case ranks above 95\% of cases in the evaluation bundle, not that it has a 95\% probability of being coordinated. A calibrated probability is reported only if a separate calibration procedure has been validated. The packet records the model version and the evidence used to derive the recommendation, but it does not expose private label keys or real labels to the reviewer interface.

The review-time scope is always explicit in the packet and interface. When a historical cutoff is available, only records observable by that cutoff enter \(G_{\leq t}\); when a dataset provides only a bounded release, the packet records that bounded scope rather than implying online availability. Dataset-specific scope and external-test limitations are described in the Evaluation Method section.

**[Figure 2 about here: End-to-end HyperTrace workflow.]** The figure should connect event records, the cutoff-safe heterogeneous graph, the coordination detector, constrained reconstruction, the `ExplanationPacket`, and the reviewer's final action. A visual boundary must separate model training from the review-time path.

## 5.2 Coordination Detector

We introduce and implement `DomainAwareLorentzHGT`, a domain-aware intrinsic Lorentz heterogeneous graph detector for coordination assessment. It combines typed-neighborhood aggregation from heterogeneous graph transformers (Hu et al., 2020) with Lorentz-manifold graph representation learning (Choudhary et al., 2023; Yang et al., 2024; Park et al., 2024a, 2024b). Its account and content/activity nodes are connected by typed relations, and its observable input follows the project-defined `observable18` contract. The exact detector name and domain adapters are specific to HyperTrace, rather than an established external model.

The following table defines the `observable18` contract used by the detector. The eight semantic slots are representation slots, not psychological or affective measurements. Availability is recorded separately for every field; zero-filled values in a model matrix never imply that the corresponding evidence was observed.

| Group | Feature | Observable interpretation |
|---|---|---|
| Semantic | `Semantic_0`--`Semantic_7` | Eight project-defined semantic representation slots; no direct psychological, sentiment, or personality interpretation |
| Behavior | `Follower_Following_Ratio` | Followers relative to following count |
| Behavior | `Action_Frequency` | Observed actions or events associated with the account |
| Behavior | `Like_Ratio` | Fraction of observed activities involving likes |
| Behavior | `Retweet_Ratio` | Fraction of observed activities involving reposts/retweets |
| Behavior | `Reply_Ratio` | Fraction of observed activities involving replies |
| Behavior | `Temporal_Entropy` | Entropy of the account's observed activity-time distribution |
| Behavior | `URL_Ratio` | Fraction of observed activities containing URLs |
| Behavior | `Mention_Ratio` | Fraction of observed activities containing mentions |
| Behavior | `Hashtag_Ratio` | Fraction of observed activities containing hashtags |
| Behavior | `Media_Ratio` | Fraction of observed activities containing media references |

When a dataset does not provide a feature, we mark it as unavailable in a separate availability file. The placeholder value used for computation is never treated as observed evidence.

The detector is trained only for coordination assessment with a class-balanced coordination loss; no auxiliary prediction heads are part of the formal objective or review-time inference. It estimates whether the observed account neighborhood exhibits anomalous coordination patterns; it does not infer intent, identity, or tactical role. The choice of Lorentz geometry is not accompanied by a claim of universal classification superiority over Euclidean graph models. Prior hyperbolic graph studies motivate its use for hierarchical and multiscale structure, but predictive gains remain task- and data-dependent (Choudhary et al., 2023; Yang et al., 2024; Park et al., 2024a, 2024b; Lu et al., 2026). We therefore evaluate Euclidean and Lorentz detectors empirically and use Lorentz geometry here because it exposes a decision geometry that can be audited in the evidence reconstruction step.

We use the standard Lorentz (hyperboloid) model of hyperbolic space, with the time coordinate first and Minkowski bilinear form \(\langle x,y\rangle_L=-x_0y_0+\sum_{i=1}^{d}x_i y_i\) (Nickel and Kiela, 2018). In the implementation, \(k>0\) denotes the magnitude of negative sectional curvature \(-k\); valid points lie on the upper sheet \(\langle x,x\rangle_L=-1/k\), \(x_0>0\). Accordingly, the curvature-scaled geodesic distance is

```text
d_L(x,y) = arcosh(-k <x,y>_L) / sqrt(k)
```

which reduces to the unit-curvature Lorentz distance when \(k=1\). This is the standard distance, with the explicit curvature scaling and sign convention stated here to make the implementation auditable.

For a user representation (z_v) and learned Lorentz prototypes (p_{\mathrm{normal}}) and (p_{\mathrm{coord}}), we define the following geodesic decision margin:

```text
Δ(v) = d_L(z_v, p_normal) - d_L(z_v, p_coordination)
```

where \(d_L\) is the Lorentz geodesic distance above. A positive \(\Delta(v)\) means that the representation is closer to the coordination prototype than to the normal prototype. The distance is adopted from the standard Lorentz model; the signed difference between the two learned prototype distances is a **HyperTrace-defined operational audit statistic**. It is a task-specific reformulation of a nearest-prototype comparison, not a new universal classification metric, and is not a causal claim about intent or identity. In Section 5.3, the full-graph margin is compared with the margin of a candidate evidence subgraph to test whether the displayed evidence preserves the detector's decision geometry.

Thus, Eq. 2 should be cited as a paper-defined audit statistic derived from a standard Lorentz distance, rather than as an established metric adopted from prior work.

The workflow evaluates the detector across 15 audited checkpoints using rank consensus. We aggregate within-checkpoint distances, radial values, margins, and evidence rankings, rather than averaging raw manifold coordinates from separately trained models. Because different Lorentz spaces may have different orientations, their raw coordinates are not treated as directly commensurable. Rank consensus therefore provides a stable ordering for review while preserving the meaning of each checkpoint's prototype-relative quantities. The resulting role of the Lorentz encoder is narrow and explicit: it supplies a hierarchy-aware representation and an auditable prototype-distance geometry. Whether this geometry improves predictive accuracy over Euclidean-HGT is reported as an empirical comparison, not assumed by the method.

## 5.3 Constrained Evidence Reconstruction

HyperTrace defines an explanation as a compact evidence set rather than an unconstrained importance mask. Let (E') denote the typed edges and events displayed to a reviewer, and let (G[E']) denote the corresponding induced evidence subgraph. For a budget fraction \(\rho\) and an observed relation \(r\) with \(n_r\) candidate evidence units, the retained count is defined as:

```text
N_r(ρ) = max(1, ceil(n_r ρ)),  for each observed relation r
```

The implementation evaluates \(\rho\in\{.01,.02,.05,.10,.20,.50,1.00\}\). The lower bound of one is applied only to relation groups present in the candidate set (\(n_r>0\)); empty relation types are not created or displayed. This relation-stratified prefix rule is a HyperTrace-specific budget design, not a formula adopted from a prior explainer. It is used to prevent a global budget from silently dropping a relation type, while the candidate units within each relation are ordered by the 15-checkpoint mean attention--reliability score.

The reconstruction problem is:

```text
minimize        |E'|
subject to      D(fθ(G≤t), fθ(G[E'])) ≤ εpred
                Fgeo(G≤t, G[E'], v) ≥ 0.95
                timestamp(e) ≤ t          for every e ∈ E'
                resolvable_source(e) = 1  for every e ∈ E'
```

Here, \(D\) is the absolute difference between the full-graph and evidence-subgraph dataset-relative review percentiles, and the implementation fixes \(\epsilon_{\mathrm{pred}}=0.02\) percentile units. \(\Delta_G(v)\) and \(\Delta_{E'}(v)\) are the full-graph and evidence-subgraph Lorentz margins. **Eq. 3 (Geometry fidelity)** is implemented with

```text
Fgeo = max(0, 1 - |ΔG(v) - ΔE'(v)| / (|dG,normal| + |dG,coord|))
```

and the fixed acceptance threshold is \(F_{\mathrm{geo}}\ge 0.95\). The denominator uses the **unmasked full-graph** prototype distances \(d_{G,\mathrm{normal}}\) and \(d_{G,\mathrm{coord}}\), not the evidence-masked distances. The implementation computes this quantity independently at each of the 15 checkpoints and then takes the arithmetic mean. The only numerical floor is `np.finfo(float).eps = 2.220446049250313e-16`, used in the denominator to avoid division by zero; it is not a substantive fidelity tolerance. Thus the implementation does **not** use a single global \(\epsilon_{\mathrm{geo}}\); its equivalent case-specific margin tolerance is \(0.05\,[|d_{G,\mathrm{normal}}|+|d_{G,\mathrm{coord}}|]\) whenever the unclipped expression is positive. The prediction constraint preserves the detector's assessment, and the geometry constraint preserves its decision-relevant Lorentz comparison. Temporal validity and source resolution are hard constraints, not terms that can be traded away by changing loss weights. Minimizing \(|E'|\) directly operationalizes evidence sparsity. Eq. 3 is a HyperTrace-defined normalized geometry-fidelity statistic; no prior work is claimed for this exact normalization.

The constrained reconstruction expression above is a HyperTrace-specific method formulation, not a standalone evaluation metric. The sparsity objective and the joint prediction, geometry, temporal, and provenance constraints define the method's evidence-selection contract; the resulting prediction and geometry quantities are reported as evaluation measures.

The reconstruction protocol is deterministic and budgeted. It starts from the target account's typed candidate neighborhood. For each candidate edge or event, it evaluates the change caused by removing that item from the current subgraph, including prediction divergence and geodesic-margin change. Candidates are removed in increasing order of their measured influence. The procedure stops as soon as another removal would violate either the prediction or geometry constraint. The packet retains both the kept evidence and the removed candidates, so sparsity does not hide what was considered. The implementation and unit tests must apply the same cutoff, provenance, and threshold checks as the evaluation bundle; formal explanation results are reported only after those checks pass.

If no evidence set satisfies the hard constraints, HyperTrace returns an explicit failure state such as `insufficient evidence`, `temporal scope unavailable`, or `unresolved provenance`. It does not relax the temporal or provenance requirements to produce a visually complete explanation. This failure behavior is essential for auditability: absence of a valid packet is itself information for the reviewer.

## 5.4 Counterfactuals and Explanation Metrics

For the selected evidence (E'), we report the following properties, following the broader requirement that graph explanations be evaluated rather than judged by visual plausibility alone (Agarwal et al., 2023; Chen and Ying, 2023).

- **Comprehensiveness:** the decrease in the detector's score or rank when (E') is removed from the full graph.
- **Sufficiency:** the divergence between the full-graph assessment and the assessment obtained from (G[E']) alone.
- **Sparsity:** the retained evidence ratio (|E'|/|E_{candidate}|).
- **Geometry fidelity:** agreement between full-graph and evidence-subgraph margins, prototype distances, or rank/radial ordering in Lorentz space.
- **Temporal validity:** the proportion of displayed evidence outside the declared cutoff or observation range; the target is zero.
- **Provenance coverage:** the proportion of displayed items with resolvable evidence IDs and matching source hashes.
- **Stability:** agreement of evidence ranks, margins, and selected sets across frozen checkpoints or audited perturbations.
- **Latency:** time required to parse the case and generate the packet.

We compare HyperTrace with static top-(k) evidence, random-edge evidence, and degree-matched evidence. The controls test whether apparent fidelity is explained merely by retaining highly connected edges. Explanation parameters, budgets, and thresholds are fixed before the final human evaluation; they are not tuned using reviewer outcomes.

## 5.5 ExplanationPacket and Audit Contract

Each packet records a case identifier and irreversible anonymous account identifiers; the dataset-relative review percentile and model recommendation; the model version and checkpoint-manifest hash; each evidence ID, timestamp, relation type, and source-file hash; the observation range or cutoff; prediction and geometry constraint states; keep-only and deletion counterfactual results; unresolved evidence and scope warnings; and append-only review actions.

The packet distinguishes observed records from derived graph relations and model counterfactuals. A source link identifies the underlying record; it does not convert a model inference into an observed fact. Private labels, label keys, and unreleased account identities never enter the front-end packet. When a provenance hash or cutoff check fails, the corresponding item is marked unresolved rather than silently displayed as verified.

## 5.6 Human Review Interface

The reviewer interface is organized around the adjudication task rather than the detector's internal modules. **[Figure 3 about here: Implemented HyperTrace reviewer interface.]** The figure is a screenshot of the running implementation, showing a case from a provisional model recommendation through evidence inspection to the final human-decision panel. No private ground-truth label is exposed. The three-stage review protocol is specified in the text below and is not inferred from the screenshot.

The implemented workspace has five functional regions:

1. **Recommendation and scope.** The header reports the model recommendation as a provisional escalation decision, a dataset-relative review-priority percentile, the operation label, and the full-release snapshot warning. The percentile is used for prioritization; it is not presented as a calibrated probability.
2. **Observed activity summary.** Typed relation counts and the observed activity window provide a compact account-level context before the reviewer inspects the selected evidence.
3. **Auditable evidence summary.** The constrained-reconstruction panel reports evidence retained, sufficiency error, Lorentz geometry fidelity, and checkpoint agreement. These values summarize detector behavior and are not interpreted as intent, personality, or psychological attributes.
4. **Source records and audit status.** Each selected event is displayed with its timestamp, action type, source excerpt, and evidence identifier. Provenance coverage, timestamp coverage, and prototype-vote agreement are shown as explicit audit checks. The interface also retains the full-release scope warning; it does not imply historical online early detection.
5. **Human decision panel.** The reviewer selects `Coordinated` or `Not coordinated`, records confidence on a continuous scale, and may add a short review note before submitting the final decision. The recommendation remains provisional, and the interface does not automatically impose a platform sanction.

The controlled review study uses this interface in three stages. Participants first submit an independent judgment without explanatory evidence, then inspect the assigned model-assistance condition, and finally submit a revised judgment with confidence and review-note fields. This separation lets us measure whether the evidence supports appropriate reliance: accepting correct recommendations while rejecting incorrect ones. The interface intentionally omits psychological profiles, sentiment or personality judgments, tactical role names, training loss, epochs, and other internal features that cannot be independently checked by a reviewer.

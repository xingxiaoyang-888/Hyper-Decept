# Introduction Reference Audit

The entries below were checked against publisher metadata and/or the available abstract record. The quoted evidence is deliberately conservative: it states what the source directly supports, not a stronger conclusion inferred from its title.

| Key | Directly supported claim in the Introduction | Evidence located in the source |
|---|---|---|
| `cima2024coordinated` | IOs use coordinated behavior across large account networks; coordination can exhibit central and peripheral groups. | The abstract describes organized IOs, coordination as a tactic for spreading messages, a 624K-user/4M-tweet dataset, and central/peripheral coordination patterns. |
| `graham2024toolkit` | Coordination analysis can model multiple behaviors and legitimate online activism with weighted directed multigraphs. | The abstract explicitly names online influence, digital astroturfing, and online activism, and describes a multi-behavior weighted directed multigraph framework. |
| `zouzou2024fakefollowers` | Coordinated manipulation may be detected from anomalous following patterns and consistent behavior across multiple accounts. | The abstract describes coordinated fake-follower campaigns, anomalous following patterns, and groups whose behavior is consistent across accounts. |
| `burghardt2024sociolinguistic` | Coordinated inauthentic accounts can be studied through socio-linguistic/account-level patterns. | The paper title and ICWSM publication concern the socio-linguistic characteristics of coordinated inauthentic accounts. Use only for that narrower claim. |
| `minici2024crossplatform` | Information operations can leave traces across platforms and link-sharing ecosystems. | The abstract describes a coordinated cross-platform IO whose links point to other social platforms and websites. |
| `rogers2025manufactured` | CIB categories can include non-deceptive or ambiguous coordination and depend on platform policy definitions. | The abstract reports media groups, activists, advertising networks, and hijacked groups in a CIB analysis, and explicitly discusses when coordination crosses into inauthenticity and how Meta's definition has changed. |
| `luceri2026tiktok` | Video-first affordances change observable coordination signals and expose limits of text-centric detection. | The abstract contrasts text-centric methods with TikTok, and distinguishes synchronized posting/content reuse from platform-native Duet/Stitch interactions that may be organic. |
| `piao2025social` | LLM-driven social agents can exhibit human-like social behavior during interaction. | The abstract describes LLM-driven social agents, self-reflection, communication, and social interaction. |
| `jin2025synthetic` | LLM agents can generate posts, alter social ties, and adapt interaction behavior in a simulated environment. | The abstract describes post generation, follow/unfollow decisions, RL adaptation, and evolving influence dynamics. |
| `agarwal2023evaluating` | Graph explanation quality requires explicit evaluation rather than visual inspection alone. | The abstract states that explanation quality and reliability are difficult to assess and introduces benchmark datasets and evaluation metrics. |
| `yang2024hypformer` | Hyperbolic/Lorentz representations are motivated by hierarchical structure and can support fully hyperbolic Transformer computation. | The abstract motivates hyperbolic geometry for tree-like/hierarchical data and identifies the Lorentz model used by Hypformer. |
| `park2024hyperbolic`, `park2024multihyperbolic` | Heterogeneous graphs may contain hierarchical or power-law structure, motivating one or multiple hyperbolic spaces. | Both abstracts explicitly motivate hyperbolic spaces by complex, hierarchical, or diverse power-law structures in heterogeneous graphs. |
| `wang2024humanexplain` | Explanation evaluation should consider how people use and process explanations, including changes over time. | The abstract calls for a human-centered perspective and studies explanation use, model updates, and human cognitive interaction with explanations. |

## Claims That Should Not Receive a Forced External Citation

- The four formative-interview requirements are evidence from this study, not prior-work claims.
- `Geometric fidelity` as correspondence between full-graph and evidence-subgraph Lorentz geometry is HyperTrace's operational definition; define it in Methods and cite `agarwal2023evaluating` only for the broader need to evaluate explanations.
- Cutoff safety and evidence provenance are HyperTrace system contracts. Prior work can motivate auditability, but it should not be cited as if it established your exact contract.
- The exact Lorentz-HGT implementation is a component of this paper. Cite `hu2020hgt` for heterogeneous message-passing background and recent hyperbolic work for representation context, but do not imply that any cited paper implements `DomainAwareLorentzHGT`.


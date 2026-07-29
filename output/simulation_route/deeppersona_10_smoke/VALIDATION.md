# DeepPersona → MultiAgent4Collusion → White-box smoke validation

Date: 2026-07-28

## Verdict

The isolated synthetic route is runnable end to end at smoke-test scale:

```text
official DeepPersona profiles
  -> per-agent persona chunks and vector store
  -> OASIS agent CSV
  -> MultiAgent4Collusion with local Qwen
  -> two-timestep behavior database and audit artifacts
  -> binary detection + SHAP
  -> gang discovery
  -> HGT + Poincare + DPMM role output
```

TwiBot-22 was not used anywhere in this route.

## Inputs and controls

- 10 DeepPersona profiles selected deterministically from the official world-profile source.
- 65 persona chunks embedded with `all-mpnet-base-v2`.
- Agent roles: 7 `good`, 1 `bad_leader`, 2 `bad_member`.
- Random follow graph: 23 stored follow edges.
- Local model: `qwen3.5:9b` through Ollama's OpenAI-compatible endpoint.
- Two timesteps, random recommender, all agents activated, no reflection, no defense.

## DeepPersona evidence

- All 10 CSV rows contain a non-empty full persona summary.
- The runtime loaded the isolated Chroma collection.
- RAG injection succeeded 20/20 times across all 10 agents and both timesteps.
- The preserved `rag_debug.txt` identifies the agent, query, retrieved sections,
  distances, and injection result for every action context.

## Simulation evidence

Clean regression run: `simulation_10_t2_fixed.db`

- SQLite integrity check: `ok`.
- Users: 10; posts: 53; follow edges: 23; traces: 58.
- Trace timestamps: 29 events at `created_at=3`, 29 events at `created_at=6`.
- Dynamic actions include 14 likes, 13 comments, 6 comment likes, 2 new
  posts, 1 repost, 1 dislike, and 20 feed refreshes.
- Model calls: 19 successful, 0 failed. One bad member executed a
  deterministic task path without an LLM call.
- The clean run contains three bounded leader task-creation audit records and
  no JSON parse error.

The preceding diagnostic run is also preserved because it demonstrates the
full blackboard handoff: the leader created three tasks at timestep 1 and
`bad_member` agent 6 selected and executed task 1 at timestep 2. That run
exposed an overlong leader JSON response later in timestep 2.

## Reliability fix

The leader prompt now requires at most three concise function calls. Runtime
parsing also caps a valid leader batch at three actions and adds a corrective
message before retrying an invalid response. The clean regression reproduced
the previously problematic second leader turn as one valid three-task JSON
object.

## Current white-box output

The clean database was accepted by all currently implemented downstream
stages:

- 10 accounts and 3 bad labels loaded.
- 32-dimensional final node features produced.
- 19 original graph edges plus 3 cosine edges.
- LOOCV predictions and SHAP summary produced.
- Predicted-bot subgraph processed by Louvain.
- HGT/Poincare/DPMM role assignments produced for all 10 agents.
- The configured `bad_leader` was assigned `Opinion Leader` in this smoke run.

The measured detection accuracy is 0.40 and bot recall is 0.00. These values
are not research results: ten samples cannot train or evaluate the detector.
They only establish interface compatibility from simulation DB to explanation
artifacts.

## Not yet complete

- Psychology mode was disabled because the spaCy dependency/models are not
  installed; psychology columns are therefore zero.
- HGT ran for two epochs only.
- Detection is post-run batch processing, not yet invoked after every
  simulation timestep.
- The planned `ExplanationPacket`, Geo-PGExplainer, four fidelity metrics,
  counterfactual evidence deletion, and evidence-trace dashboard are not yet
  implemented as one unified white-box system.
- Formal experiments need more agents, more timesteps, multiple seeds, and a
  detector trained independently of the 10-agent evaluation run.

## Canonical artifacts

- `agents_10.csv`: simulation population and ground-truth roles.
- `simulation_10_t2_fixed.yaml`: exact clean-run configuration.
- `simulation_10_t2_fixed.db`: dynamic behavior database.
- `simulation_10_t2_fixed_manifest.json`: checksums and audit counts.
- `simulation_10_t2_fixed_artifacts/`: logs, raw actions, RAG audit, task audit,
  detector pickle, and plots.
- `whitebox/simulation_10_t2_fixed/`: predictions, SHAP, gang, role, and
  Poincare outputs.

# HyperDecept P2 Smoke data preparation report

- Code commit: `09427a1f6f7ac642b3e0221c06d8b68db9bbc2ce`
- Scope: Smoke data preparation only; no formal training executed
- Raw TwiBot uploaded: `false`
- DatasetPlan: `artifact_contract_valid = true`, `errors = []`
- Relocation: all files exist and all paths remain inside relocated root
- Tests: `194 passed, 25 warnings in 7.75s`
- Compileall: passed
- Git diff check: passed

## Bundles

| Bundle | Scale | Size |
|---|---:|---:|
| TwiBot-22 | 1000 core users | 49.90 MiB |
| MGTAB | 10199 users | 76.99 MiB |
| Synthetic | 2 x 500-Agent episodes | 12.62 MiB |

## Synthetic episodes

- `leader_amplifier`: 500 agents, seed 11, 50 steps
- `independent_attack`: 500 agents, seed 11, 50 steps

MGTAB keeps the required files at bundle root and also includes the current code's `derived/` compatibility copies. TwiBot 26d psychology fields remain zero placeholders and are Smoke-only.

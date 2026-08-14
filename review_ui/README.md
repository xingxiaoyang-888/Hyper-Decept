---
title: HyperTrace Review Study
emoji: "🔎"
colorFrom: gray
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
license: other
---

# HyperTrace Review Study

Private CHI study deployment for comparing risk-only, standard-signal, and
auditable HyperTrace evidence interfaces.

Required Space secrets:

- `ADMIN_TOKEN`: protects aggregate and CSV export endpoints.
- `STUDY_SALT`: hashes participant codes and derives completion codes.

For a formal study, upload the audited private case package as
`data/study_cases.private.json` when preparing the private Space. Without that
file the application starts with synthetic demonstration cases and reports
`demo_data: true` from `/api/health`.

Persistent Storage should be mounted at `/data`. If it is unavailable, set
`HYPERTRACE_DB_PATH` to another durable path before collecting participants.
Set `HYPERTRACE_DURABLE_STORAGE=1` only after that storage has been verified.

The protected administration console is available at `/admin`. See
`DEPLOYMENT.md` for the formal pre-pilot gates.

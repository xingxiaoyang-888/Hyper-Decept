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

# HyperTrace Review Study and Demonstration

This directory contains the reviewer-facing HyperTrace web application. The
`Demonstration` branch uses only the de-identified synthetic case package and
is intended for UI walkthroughs and screen recording; it is not a deployment
of the private human-subject study.

## Demonstration quick start

Requirements: Python 3.10+ and the packages in `requirements.txt`.

On Windows PowerShell:

```powershell
cd review_ui
python -m pip install -r requirements.txt
.\run_demo.ps1
```

Open `http://127.0.0.1:8765/`. For a direct preview without entering a
participant code, open:

```text
http://127.0.0.1:8765/?preview=hypertrace_evidence
```

Use the interface selector in the amber preview bar to switch among
`Risk only`, `Standard signals`, and `HyperTrace evidence`; the arrow buttons
switch among the bundled cases. Stop the server with `Ctrl+C`.

## Current frontend functionality

1. **Consent and private session entry** — validates consent and age
   confirmation, hashes the participant code before storage, and resumes an
   incomplete session.
2. **Briefing and progress** — explains the task, shows the three-stage flow,
   reports case progress, and displays the full-release-snapshot warning.
3. **Common case evidence** — exposes only the label-blind, model-blind case
   context before the initial judgment: scope, record count, observation range,
   event counts, and shared activity records.
4. **Three assistance conditions** — risk priority only; risk priority plus
   relation/time summaries; or the full HyperTrace evidence packet.
5. **Auditable evidence view** — shows retained evidence, sufficiency error,
   geometry fidelity, checkpoint agreement, full-vs-evidence-only risk,
   timestamp coverage, provenance coverage, prototype vote, and selected
   source records.
6. **Three-stage decision flow** — initial unaided decision and confidence,
   locked reveal of the assigned assistance, then final decision, confidence,
   optional rationale, and submission.
7. **Evidence version comparison** — in HyperTrace preview, the version panel
   compares the current packet with an earlier snapshot and counts added,
   removed, and retained evidence records.
8. **Evidence rollback prototype** — `Restore as current` creates a new local
   version from the selected snapshot without overwriting the prior version.
   This is an offline browser prototype; it is not yet persisted to the study
   database or an append-only audit ledger.
9. **Post-task questionnaire** — collects trust/calibration, clarity,
   workload, evidence usefulness, and optional comments.
10. **Admin export (when configured)** — the private deployment provides the
    protected `/admin` console and aggregate CSV export using `ADMIN_TOKEN`.

## Recording workflow

For a short product demonstration, use the preview URL and follow this order:

1. Start with `Risk only` to show the model recommendation and percentile
   review priority.
2. Switch to `Standard signals` to show relation counts and the observed
   activity window.
3. Switch to `HyperTrace evidence` to show the constrained packet, source
   records, provenance/timestamp checks, and geometry metrics.
4. Open the version panel, compare `v1` with the current snapshot, then click
   `Restore as current` to demonstrate non-destructive rollback.
5. Use the next-case arrow to show that the same workflow works across the
   bundled cases.

The demonstration cases are synthetic/de-identified examples. Do not use the
demonstration database or screenshots as human-subject study results.

## Scope limitations

The current prototype does **not** implement live event ingestion, automatic
new-evidence notifications, server-persistent version history, graph-level
visual diffing of arbitrary packets, or review-state rollback of a prior
judgment/confidence/note. Those features remain future extensions. The current
version comparison is evidence-record based and local to the open browser
case.

The formal protocol assigns eight cases per participant. Each case records an
initial unaided judgment, an explicit model-assistance reveal, and a final
judgment so that accuracy change, RAIR, RSR, and wrong-AI rejection can be
computed from observed decision transitions.

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

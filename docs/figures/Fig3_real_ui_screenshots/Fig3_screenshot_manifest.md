# Figure 3 Screenshot Manifest

## Capture metadata

- Case used: `HT-H-01` (displayed in the UI as `Case 01`)
- Operation shown: `Operation H` / `honduras`
- Source page: running local HyperTrace preview at `http://127.0.0.1:8768/?preview=hypertrace_evidence`
- Browser: Google Chrome headless, default device scale factor
- Browser zoom: 100%
- Viewport: 1600 x 3000 CSS pixels
- Image format: PNG
- Capture mode: read-only preview using an isolated copy of the SQLite database
- Visible UI state: the implementation's `Researcher preview` banner is retained; no synthetic data, placeholder, or demo records were added by this capture.
- Formal research data changed: No
- Reviewer judgments submitted or revised: No
- Synthetic update or rollback triggered: No

## Files

| File | UI state | Captured content | Requirements not available in this state |
|---|---|---|---|
| `Fig3a_case_scope.png` | HyperTrace evidence, model assistance revealed, Case 01 | Operation H, Case 01, provisional model recommendation, within-operation review-priority percentile, full-release snapshot scope, available record count, and observed date range | No ground-truth label or correctness field is exposed. The page also contains common case records and the read-only decision panel because these are part of the same real UI state. |
| `Fig3b_relational_temporal_context.png` | HyperTrace evidence, same Case 01 | Activity summary with relation-type counts (`Posts`, `Retweets`), observed activity window, event summary, and surrounding case context | No model internals or fabricated propagation graph is shown. |
| `Fig3c_evidence_audit_versions.png` | HyperTrace evidence, same Case 01 | Evidence retention, sufficiency error, Lorentz geometry fidelity, checkpoint agreement, full-graph/evidence-only percentiles, alternative-explanation reviewer aid, selected evidence IDs and timestamps, provenance/timestamp coverage, version comparison, restore control, and update check control | The shown update control is polling-based. No live platform stream is represented. |
| `Fig3d_reviewer_decision_revision.png` | HyperTrace evidence preview, final-decision panel visible | `Coordinated`, `Not coordinated`, confidence slider, optional review note, and the real `Preview only` state | The current read-only preview has no submitted judgment, revision history, or save/submit action. Therefore `submitted-judgment revision` is not shown and was not simulated. `Request more evidence`, external source opening, sanctioning, and automatic rollback are not shown. |

## Privacy and integrity checks

- No real account names, contact information, IP addresses, participant identifiers, or ground-truth labels are visible.
- The screenshots retain only the UI's anonymized `@account` text and evidence IDs.
- No image was AI-generated or manually redrawn; all four files are crops of one browser screenshot of the running implementation.
- No arrows, panel labels, `(a)`--`(d)` markers, or explanatory overlays were added.
- The screenshots preserve the UI's full-release-snapshot warning and do not imply historical online early detection.

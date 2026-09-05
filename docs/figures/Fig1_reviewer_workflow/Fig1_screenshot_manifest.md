# Fig1 Reviewer Workflow Screenshot Manifest

All screenshots use the same de-identified Case 01 from Operation A (`DEMO-A01` / `Case 01`) and PNG format. Preview screenshots use the local researcher-preview interface; the independent-judgment screenshot uses the real initial stage of the same local demonstration study flow. No judgment was submitted and no formal study record was modified.

## Capture settings

- Browser: local headless Chrome
- Viewport: 1440 x 1100 CSS pixels
- Browser/page zoom: 100%
- Device scale factor: 2
- Browser chrome, address bar, taskbar, and pointer: excluded
- Sensitive fields: no real account names, contact information, IP addresses, participant codes, ground-truth labels, or correctness labels are shown

## Captured regions

| File | Interface state | Case consistency | Notes |
| --- | --- | --- | --- |
| `Fig1_case_scope_common_context.png` | Formal study, Stage 1 initial judgment; Common case context | Case 01 / Operation A | Shows release scope and the currently available pre-assistance context. The local demo reports records and observed range as unavailable in this state; this is an authentic UI state, not a fabricated value. |
| `Fig1_independent_judgment.png` | Formal study, Stage 1 initial judgment | Case 01 / Operation A | Shows Coordinated / Not coordinated, confidence, and Lock initial decision. No response was submitted. |
| `Fig1_risk_only_priority.png` | Researcher preview, Risk only | Case 01 / Operation A | Shows model recommendation and within-universe review-priority percentile. |
| `Fig1_standard_signals_relations_time.png` | Researcher preview, Standard signals | Case 01 / Operation A | Shows relation counts and observed activity window. |
| `Fig1_hypertrace_compact_evidence.png` | Researcher preview, HyperTrace evidence | Case 01 / Operation A | Shows compact evidence, sufficiency error, geometry fidelity, checkpoint/prototype agreement, counterfactual panel, source records, provenance, timestamp coverage, version comparison, rollback control, and update controls. |
| `Fig1_final_judgment.png` | Researcher preview, final-decision view | Case 01 / Operation A | Shows final classification controls, confidence, review note, and the preview-only action state. |

## Requested functionality check

- Common case context: present.
- Independent judgment: present in the formal initial-stage flow; not present in the already-assisted preview mode.
- Risk-only review-priority card: present.
- Standard-signals relations and observed-window cards: present.
- HyperTrace compact evidence and audit fields: present.
- Final judgment controls: present.
- Version comparison and restore control: visible in the HyperTrace preview.
- Live update polling/check control: visible in the HyperTrace preview.
- Submitted-judgment revision: not shown in these captures because the preview has no submitted judgment and no new judgment was created for the screenshots.
- Automatic rollback: not triggered; only the available restore control is shown.

## Integrity note

The screenshots are direct crops of the rendered frontend. No UI elements were redrawn, composited, or generated, and no backend case data was changed for capture.

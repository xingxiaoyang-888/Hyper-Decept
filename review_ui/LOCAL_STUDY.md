# Local human-subject study

## Start

Run from PowerShell:

```powershell
cd "F:\aidetect\Hyper Decept\.runtime-downloads\review_ui_space"
.\run_local_study.ps1
```

Participant interface: `http://127.0.0.1:8765/`

Administration console: `http://127.0.0.1:8765/admin`

The admin token is stored in `data/.admin_token`. Participant responses are
stored in `data/hypertrace_study.sqlite3` using SQLite WAL mode.

## Formal data boundary

- `data/study_cases.private.json` contains server-side ground truth.
- Trial API responses never include `ground_truth` or platform identifiers.
- Raw participant codes are HMAC-hashed before storage.
- IP addresses and browser fingerprints are not stored.
- Each participant receives five coordinated and five non-coordinated cases.
- Interface assignment is balanced across risk-only, standard-signal, and
  HyperTrace evidence conditions.

## Daily operating procedure

1. Start the server before opening recruitment.
2. Confirm `/api/health` reports `demo_data: false`.
3. Open `/admin` and verify the session totals.
4. Export the response CSV after each collection block.
5. Back up `data/hypertrace_study.sqlite3`, its `-wal`/`-shm` files if present,
   and the exported CSV while the study server is stopped.

Do not delete or replace the SQLite database during active data collection.
Pilot responses should use a separate database and must not be mixed with the
formal study database.

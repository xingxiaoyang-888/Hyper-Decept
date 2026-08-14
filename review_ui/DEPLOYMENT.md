# Hugging Face Space deployment

The formal study must use a **private Docker Space**. The private case package
contains server-side ground truth and must not be published before data
collection and paper review are complete.

## Required configuration

1. Create a private Docker Space with port `7860`.
2. Enable Persistent Storage mounted at `/data`.
3. Add Space Secrets:
   - `ADMIN_TOKEN`: a new random value of at least 32 bytes.
   - `STUDY_SALT`: a different random value of at least 32 bytes.
4. Add the Space Variable `HYPERTRACE_DURABLE_STORAGE=1` only after Persistent
   Storage is mounted and verified at `/data`.
5. Add `data/study_cases.private.json` to the private Space repository. The
   public source repository intentionally ignores this file.
6. Push the contents of `review_ui/` as the Space repository root.

## Pre-pilot gates

- `/api/health` returns `demo_data: false` and `durable_storage: true`.
- `/admin` accepts the configured token and exports a CSV.
- Three test participants are assigned one each to risk-only, standard-signal,
  and HyperTrace evidence conditions.
- Trial payloads do not contain `ground_truth` or platform account IDs.
- Desktop and mobile layouts pass the review checklist.
- The consent text, recruitment wording, and compensation match the approved
  institutional protocol.

Do not recruit participants while `demo_data` or `durable_storage` is false.

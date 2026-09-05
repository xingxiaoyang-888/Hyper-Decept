param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Data = Join-Path $Root "data"
$Cases = Join-Path $Data "study_cases.demo.json"
$Database = Join-Path $Data "hypertrace_demo.sqlite3"

if (-not (Test-Path -LiteralPath $Cases)) {
    throw "Demonstration case package not found: $Cases"
}

$env:HYPERTRACE_CASES_PATH = $Cases
$env:HYPERTRACE_DB_PATH = $Database
$env:STUDY_SALT = "demonstration-only-local-salt"
$env:ADMIN_TOKEN = "demonstration-admin-token"
$env:HYPERTRACE_DURABLE_STORAGE = "0"
$env:HYPERTRACE_PREVIEW_MODE = "1"
$env:HYPERTRACE_TRIAL_COUNT = "8"

Write-Host "HyperTrace demonstration UI: http://127.0.0.1:$Port/"
Write-Host "Preview: http://127.0.0.1:$Port/?preview=hypertrace_evidence"
Write-Host "Conditions: risk_only, standard_signals, hypertrace_evidence"
Write-Host "Press Ctrl+C to stop."

Set-Location $Root
python -m uvicorn app:app --host 127.0.0.1 --port $Port

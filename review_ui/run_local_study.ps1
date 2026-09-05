param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Data = Join-Path $Root "data"
$Cases = Join-Path $Data "study_cases.private.json"
$Database = Join-Path $Data "hypertrace_study.sqlite3"
$SaltFile = Join-Path $Data ".study_salt"
$AdminFile = Join-Path $Data ".admin_token"

if (-not (Test-Path -LiteralPath $Cases)) {
    throw "Formal case package not found: $Cases"
}

function New-Secret([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        $Bytes = New-Object byte[] 32
        $Generator = [Security.Cryptography.RandomNumberGenerator]::Create()
        try {
            $Generator.GetBytes($Bytes)
        }
        finally {
            $Generator.Dispose()
        }
        $Secret = -join ($Bytes | ForEach-Object { $_.ToString("x2") })
        [IO.File]::WriteAllText($Path, $Secret)
    }
    return [IO.File]::ReadAllText($Path).Trim()
}

$env:HYPERTRACE_CASES_PATH = $Cases
$env:HYPERTRACE_DB_PATH = $Database
$env:STUDY_SALT = New-Secret $SaltFile
$env:ADMIN_TOKEN = New-Secret $AdminFile
$env:HYPERTRACE_DURABLE_STORAGE = "1"
$env:HYPERTRACE_PREVIEW_MODE = "1"
$env:HYPERTRACE_TRIAL_COUNT = "8"

Write-Host "Participant UI: http://127.0.0.1:$Port/"
Write-Host "Admin console: http://127.0.0.1:$Port/admin"
Write-Host "HyperTrace preview: http://127.0.0.1:$Port/?preview=hypertrace_evidence"
Write-Host "Admin token file: $AdminFile"
Write-Host "Response database: $Database"
Write-Host "Stop the study server with Ctrl+C."

Set-Location $Root
python -m uvicorn app:app --host 127.0.0.1 --port $Port

# Whisper Typer — Start Faster-Whisper server locally (no Docker)
# Run setup-transcribe-local.ps1 first if you haven't already.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Whisper Typer — Starting transcribe server (local)" -ForegroundColor Cyan

# Ensure venv exists
if (-not (Test-Path ".venv") -and -not (Test-Path "venv")) {
    Write-Host "No virtual environment found. Run setup-transcribe-local.ps1 first." -ForegroundColor Red
    exit 1
}

# Load .env if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

# Run uvicorn from root/app (same module layout as Docker)
Push-Location "root\app"
try {
    uv run uvicorn transcribe_api:app --host 127.0.0.1 --port 8000
} finally {
    Pop-Location
}

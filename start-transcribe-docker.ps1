# Whisper Typer — Start Faster-Whisper server (Docker Desktop, Windows)
# Uses Docker Desktop directly. GPU support requires Docker Desktop with WSL2 backend
# and NVIDIA drivers installed — no separate NVIDIA Container Toolkit needed on Windows.

$ErrorActionPreference = "Stop"

# Run from script directory (project root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Whisper Typer — Starting Faster-Whisper server (Docker)" -ForegroundColor Cyan

# Create .env from .env.example if it doesn't exist
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host ".env created from .env.example — fill in your HF_TOKEN if desired." -ForegroundColor Yellow
    }
}

docker compose up -d --build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start container. Check Docker Desktop is running." -ForegroundColor Red
    exit 1
}

Write-Host "`nContainer started. Logs: docker compose logs -f faster-whisper" -ForegroundColor Green

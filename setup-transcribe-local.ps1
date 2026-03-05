# Whisper Typer — Setup local Whisper server (no Docker)
# Creates a venv, installs server deps, and pre-downloads the model.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Whisper Typer — Setting up local server" -ForegroundColor Cyan

# Load .env if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

$model = if ($env:WHISPER_MODEL) { $env:WHISPER_MODEL } else { "tiny" }

# Create .env from .env.example if missing
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host ".env created from .env.example — fill in HF_TOKEN if desired." -ForegroundColor Yellow
    }
}

# Create venv if not present
if (-not (Test-Path ".venv") -and -not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    uv venv
}

$venvPython = if (Test-Path ".venv") { ".venv\Scripts\python.exe" } else { "venv\Scripts\python.exe" }

# Install server dependencies (ctranslate2 is large — increase timeout)
$env:UV_HTTP_TIMEOUT = 300
Write-Host "Installing server dependencies..." -ForegroundColor Gray
uv pip install fastapi uvicorn python-multipart faster-whisper numpy
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependency install failed. Try again or increase UV_HTTP_TIMEOUT." -ForegroundColor Red
    exit 1
}

# Pre-download model
Write-Host "Downloading Whisper model '$model'..." -ForegroundColor Gray
& $venvPython -c @"
import os
from faster_whisper import WhisperModel
model = '$model'
models_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(models_dir, exist_ok=True)
print(f'Saving to {models_dir}')
WhisperModel(model, device='cpu', compute_type='int8', download_root=models_dir)
print(f'Model [{model}] ready.')
"@

Write-Host "`nDone. Start the transcribe server with: .\start-transcribe-local.ps1" -ForegroundColor Green

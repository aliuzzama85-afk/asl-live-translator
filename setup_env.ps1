# Creates/updates the Python virtual environment and installs dependencies.
# Usage: .\setup_env.ps1

$ErrorActionPreference = "Stop"

$venvPath = ".venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath..."
    python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists at $venvPath."
}

$python = Join-Path $venvPath "Scripts\python.exe"

Write-Host "Upgrading pip..."
& $python -m pip install --upgrade pip

Write-Host "Installing requirements..."
& $python -m pip install -r requirements.txt

Write-Host "Running pip-audit..."
& (Join-Path $venvPath "Scripts\pip-audit.exe")

Write-Host ""
Write-Host "Done. Activate with:  .\$venvPath\Scripts\Activate.ps1"

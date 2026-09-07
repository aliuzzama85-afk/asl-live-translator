#!/usr/bin/env bash
# Creates/updates the Python virtual environment and installs dependencies.
# Usage: ./setup_env.sh
set -euo pipefail

VENV_PATH=".venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
else
    echo "Virtual environment already exists at $VENV_PATH."
fi

# Windows venvs place binaries in Scripts/, POSIX venvs in bin/
if [ -f "$VENV_PATH/Scripts/python.exe" ]; then
    PYTHON="$VENV_PATH/Scripts/python.exe"
    PIP_AUDIT="$VENV_PATH/Scripts/pip-audit.exe"
    ACTIVATE="$VENV_PATH/Scripts/activate"
else
    PYTHON="$VENV_PATH/bin/python"
    PIP_AUDIT="$VENV_PATH/bin/pip-audit"
    ACTIVATE="$VENV_PATH/bin/activate"
fi

echo "Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

echo "Installing requirements..."
"$PYTHON" -m pip install -r requirements.txt

echo "Running pip-audit..."
"$PIP_AUDIT"

echo ""
echo "Done. Activate with:  source $ACTIVATE"

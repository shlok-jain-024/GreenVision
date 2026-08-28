#!/usr/bin/env bash
# ===================================================================
#  Green Vision - one command to run the whole thing.
#
#  Creates the virtual environment if missing, installs the core
#  dependencies once, then starts the engine server. The server holds
#  the trained 42-month panel AND serves index.html, so the studio and
#  the engine share one origin and one process.
#
#  Then open http://127.0.0.1:8000
# ===================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY=".venv/Scripts/python.exe"     # Git Bash on Windows

if [ ! -x "$PY" ]; then
  echo "[1/3] Creating virtual environment..."
  python3 -m venv .venv 2>/dev/null || python -m venv .venv
  PY=".venv/bin/python"; [ -x "$PY" ] || PY=".venv/Scripts/python.exe"
else
  echo "[1/3] Virtual environment present."
fi

echo "[2/3] Installing dependencies..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt

echo "[3/3] Starting Green Vision on http://127.0.0.1:8000"
echo "    (first start trains on the panel - give it a few seconds)"
echo "    Press Ctrl+C to stop."
echo
exec "$PY" -m greenplan.server --config config/city.yaml --port 8000

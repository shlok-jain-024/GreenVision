@echo off
REM ===================================================================
REM  Green Vision - one command to run the whole thing.
REM
REM  Creates the virtual environment if it is missing, installs the core
REM  dependencies once, then starts the engine server. The server holds
REM  the trained 42-month panel AND serves index.html, so the studio and
REM  the engine share one origin and one process.
REM
REM  Then open http://127.0.0.1:8000
REM ===================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  python -m venv .venv || goto :fail
) else (
  echo [1/3] Virtual environment present.
)

echo [2/3] Installing dependencies...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :fail

echo [3/3] Starting Green Vision on http://127.0.0.1:8000
echo     (first start trains on the panel - give it a few seconds)
echo     Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m greenplan.server --config config/city.yaml --port 8000
goto :eof

:fail
echo.
echo Setup failed. Check that Python 3.10+ is installed and on PATH.
pause

@echo off
REM ===================================================================
REM  GreenVision - start the engine server from the repo root.
REM
REM  Delegates to Green-Vision\start.bat, which creates the virtual
REM  environment if missing, installs dependencies, then runs the
REM  greenplan server (studio + engine on one origin).
REM
REM  Then open http://127.0.0.1:8000
REM ===================================================================
setlocal
cd /d "%~dp0Green-Vision"
call start.bat
endlocal

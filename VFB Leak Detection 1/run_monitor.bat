@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

REM Monitoring runs continuously until you close this window (Ctrl+C) or set mode OFF.
%PYTHON% scripts\monitor.py --config config\config.yaml

endlocal

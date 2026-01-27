@echo off
setlocal
cd /d "%~dp0"

REM Use venv python if available, else fall back to system python.
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

%PYTHON% scripts\capture_images.py --config config\config.yaml

echo.
pause
endlocal

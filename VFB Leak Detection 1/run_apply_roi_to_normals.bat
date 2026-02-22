@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

REM Default behavior writes cropped images to data\normal_roi_adjusted.
REM Add --overwrite to replace originals (with backups in data\normal_backup_before_roi).
%PYTHON% scripts\apply_roi_to_normals.py --config config\config.yaml

echo.
pause
endlocal

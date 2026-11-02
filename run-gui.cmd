@echo off
setlocal
cd /d "%~dp0"
if not exist "target\release\autotab-gui.exe" (
    echo GUI missing. Run setup-rust.ps1 first.
    pause
    exit /b 1
)
start "" "%~dp0target\release\autotab-gui.exe"

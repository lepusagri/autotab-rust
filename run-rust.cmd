@echo off
setlocal
cd /d "%~dp0"
if not exist "target\release\autotab.exe" (
    echo Rust executable missing. Run setup-rust.ps1 first.
    exit /b 1
)
"target\release\autotab.exe" %*
exit /b %errorlevel%

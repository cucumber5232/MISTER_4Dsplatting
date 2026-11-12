@echo off
rem Double-click installer for Windows: runs scripts\install.ps1 and keeps the window open.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause

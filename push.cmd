@echo off
chcp 65001 >nul
echo Running push.ps1 ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push.ps1"
echo.
pause

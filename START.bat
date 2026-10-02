@echo off
rem Lip-Sync Studio: double-click to start, then use the browser tab it opens.
rem Close this window to stop it.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run SETUP.bat first.
  pause
  exit /b 1
)
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%PATH%;%LOCALAPPDATA%\Microsoft\WinGet\Links"
".venv\Scripts\python.exe" lipsync-3d\webapp\server.py --local %*
pause

@echo off
rem Lip-Sync Studio on this PC: no login, only reachable from this machine.
rem Double-click it, or run it from the repository's terminal.
cd /d "%~dp0\..\.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python lipsync-3d\webapp\server.py --local %*
pause

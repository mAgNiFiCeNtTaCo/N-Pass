@echo off
rem Starts N-Pass without a console window. Double-click this file.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo N-Pass is not set up yet. Open a terminal in this folder and run:  uv sync
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m npass.app

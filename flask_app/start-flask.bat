@echo off
REM Flask Server Starter
REM Double-click this file to start the Flask development server

echo ========================================
echo Starting Flask Development Server...
echo ========================================
echo.

REM First, kill any existing Python processes to ensure clean start
echo Stopping any existing Python processes...
taskkill /F /IM python.exe /T 2>nul
timeout /t 1 /nobreak >nul
echo.

cd /d "%~dp0"

echo Starting server on http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.

REM Use the virtual environment's Python directly
"..\.venv\Scripts\python.exe" run.py

pause

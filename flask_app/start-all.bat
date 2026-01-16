@echo off
echo ============================================================
echo Starting Candidate Evaluator - Flask + Background Worker
echo ============================================================
echo.

REM First, kill any existing Python processes to ensure clean start
echo Stopping any existing Python processes...
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul
echo.

REM Start Flask in a new window
echo [1/2] Starting Flask web server...
start "Flask Web Server" cmd /k "cd /d "%~dp0" && ..\\.venv\\Scripts\\python.exe run.py"

REM Wait a moment for Flask to initialize
timeout /t 2 /nobreak >nul

REM Start Worker in a new window
echo [2/2] Starting background worker...
start "Background Worker" cmd /k "cd /d "%~dp0" && ..\\.venv\\Scripts\\python.exe worker.py"

echo.
echo ============================================================
echo ✅ Both services started in separate windows
echo ============================================================
echo    - Flask Web Server (port 5000)
echo    - Background Worker (polls every 5 seconds)
echo.
echo Close this window or press any key to continue...
pause >nul

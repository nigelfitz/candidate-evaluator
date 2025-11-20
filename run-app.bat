@echo off
setlocal EnableExtensions
pushd "%~dp0"

rem === Config ===
set "VENV_DIR=.venv"
set "PYV=%VENV_DIR%\Scripts\python.exe"
set "APP=app.py"
set "PORT=8502"  rem avoid 8501 leftovers

rem === Ensure venv exists ===
if not exist "%PYV%" (
  echo [setup] Creating virtual environment in "%VENV_DIR%"...
  python -m venv "%VENV_DIR%" || (echo [error] Could not create venv & goto :end)
)

rem === Upgrade pip tooling (quiet)
"%PYV%" -m pip install --upgrade pip setuptools wheel >nul 2>&1

rem === Install from requirements.txt if present (idempotent)
if exist "requirements.txt" (
  echo [setup] Ensuring dependencies from requirements.txt...
  "%PYV%" -m pip install --disable-pip-version-check --no-input -r requirements.txt
) else (
  echo [setup] Installing core packages...
  "%PYV%" -m pip install --disable-pip-version-check --no-input streamlit pandas numpy pdfplumber python-docx scikit-learn altair matplotlib reportlab tabulate
)

rem === Launch Streamlit in a NEW window (non-blocking) ===
start "Candidate Pack Summariser" "%PYV%" -m streamlit run "%APP%" --server.headless true --server.port %PORT% --browser.gatherUsageStats false

rem === Wait until the server is listening (up to 60s), then open browser ===
for /l %%i in (1,1,60) do (
  >nul timeout /t 1
  powershell -NoProfile -Command "$p=Test-NetConnection -ComputerName 'localhost' -Port %PORT% -InformationLevel Quiet; if($p){exit 0}else{exit 1}" 2>nul
  if not errorlevel 1 goto :open
  (netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul) && goto :open
)

echo [warn] Server not ready after 60s. You can still open: http://localhost:%PORT%/
goto :end

:open
start "" "http://localhost:%PORT%/"
goto :end

:end
popd
exit /b

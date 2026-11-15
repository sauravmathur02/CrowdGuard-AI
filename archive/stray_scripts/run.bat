@echo off
setlocal
echo ============================================
echo   Deepfake Detector - Starting Server...
echo ============================================
cd /d "%~dp0UniversalFakeDetect"
call "..\venv_gend\Scripts\activate.bat"

echo Checking for stale processes on ports 8000 and 8001...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    if not "%%a"=="" taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8001 "') do (
    if not "%%a"=="" taskkill /F /PID %%a >nul 2>&1
)

echo.
echo Starting server... Please wait for "Ensemble models warmup completed."
echo Open your browser at: http://localhost:8000
echo Press Ctrl+C to stop.
echo.
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause


@echo off
setlocal
title CrowdGuard AI Launcher
echo ==========================================================
echo   CrowdGuard AI - Physical Anomaly & Deepfake Defense System
echo ==========================================================
echo.

set "ROOT=%~dp0"
set "PATH=C:\Program Files\nodejs;%PATH%"

echo Checking for stale processes on ports 8000, 8001, and 3000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    if not "%%a"=="" taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8001 "') do (
    if not "%%a"=="" taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000 "') do (
    if not "%%a"=="" taskkill /F /PID %%a >nul 2>&1
)

echo Starting FastAPI Backend (CUDA Vision Engine on :8000)...
start "CrowdGuard AI Backend" /d "%ROOT%CrowdGuardAI" cmd /k ""%ROOT%venv_gend\Scripts\activate.bat" && python -m uvicorn app:app --host 0.0.0.0 --port 8000"

echo Starting Next.js Production Frontend on :3000...
start "CrowdGuard AI Frontend" /d "%ROOT%frontend" cmd /k "set PATH=C:\Program Files\nodejs;%%PATH%% && npm run start"

echo.
echo ==========================================================
echo   CrowdGuard AI System is launching!
echo   - Next.js Web Console:   http://localhost:3000
echo   - FastAPI Backend API:   http://localhost:8000
echo   - Interactive API Docs:  http://localhost:8000/docs
echo ==========================================================
echo.
timeout /t 4 >nul
start http://localhost:3000

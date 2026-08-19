@echo off
cd /d "%~dp0"

echo.
echo ============================================
echo  IntelliBase — full launch (infra + app)
echo ============================================
echo.

echo [1/2] Starting Docker infrastructure...
docker compose -f compose.infra.yaml up -d
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Is Docker Desktop running?
    pause
    exit /b 1
)

echo [2/2] Starting app (uvicorn --reload)...
echo.
echo  Press Ctrl+C to STOP
echo  Docker services will stay up — use dev-infra-down.bat to stop them
echo.

".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

@echo off
cd /d "%~dp0"

echo ============================================
echo  IntelliBase — start infrastructure (Docker)
echo ============================================
echo.

docker compose -f compose.infra.yaml up -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start Docker containers.
    echo Make sure Docker Desktop is running.
    pause
    exit /b 1
)

echo.
echo Infrastructure is running:
echo   Redis    — localhost:6379
echo   Postgres — localhost:5432
echo   Phoenix  — localhost:6006  (UI)
echo.
echo Now start the app:
echo   ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
echo   or use "IntelliBase Debug" run configuration in PyCharm
echo.
pause

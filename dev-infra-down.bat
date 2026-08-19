@echo off
cd /d "%~dp0"

echo Stopping IntelliBase infrastructure...
docker compose -f compose.infra.yaml down
echo Done.
pause

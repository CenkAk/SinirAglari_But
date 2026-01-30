@echo off
cd /d "%~dp0"
start "Flask App" python app.py
timeout /t 3 /nobreak >nul
start http://localhost:5000

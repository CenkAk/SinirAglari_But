@echo off
cd /d "%~dp0"
echo Gereklilikler kuruluyor...
pip install -r requirements.txt
echo.
echo Kurulum tamamlandi.
echo Simdi calistirabilirsiniz.
pause

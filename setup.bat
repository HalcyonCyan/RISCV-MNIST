@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 scripts\setup.py
goto finished
:use_python
python scripts\setup.py
:finished
if errorlevel 1 (
    echo Setup failed. Install Python 3.10+ and Java 8+, then try again.
    pause
    exit /b 1
)
echo Ready. Double-click start.bat to launch.
pause

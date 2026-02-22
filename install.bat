@echo off
echo Installing SubEditor...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment and install dependencies
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Installation complete!
echo Run 'run.bat' to start the application.
pause

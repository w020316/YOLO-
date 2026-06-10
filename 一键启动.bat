@echo off
echo ============================================================
echo Food Processing Safety Detection System - Quick Start
echo ============================================================

cd /d "c:\Users\admin\Desktop\zhuashengb1\all_data\yolov8_project"

echo.
echo [1/4] Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [2/4] Installing dependencies...
pip install -r requirements.txt -q

echo.
echo [3/4] Converting dataset (if needed)...
if not exist "dataset\images\train\*.jpg" (
    python _run_convert.py
)

echo.
echo [4/4] Starting web server...
echo.
echo ============================================================
echo Server starting at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo ============================================================
python app.py

pause

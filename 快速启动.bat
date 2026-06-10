@echo off
chcp 65001 >nul 2>&1
title Food Processing Safety Detection System - 快速版

echo ╔══════════════════════════════════════════════════╗
echo ║   食品加工人员异常行为检测系统 - 快速部署        ║
echo ╚══════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [✓] 项目目录: %cd%
echo.
echo [ℹ] 使用国内镜像源 (阿里云) 加速下载...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [✗] 错误: 未找到Python
    pause
    exit /b 1
)

echo [1/4] 快速安装依赖包 (国内镜像)...
pip install ultralytics flask opencv-python numpy pillow PyYAML tqdm -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -q
if errorlevel 1 (
    echo     备用: 尝试清华源...
    pip install ultralytics flask opencv-python numpy pillow PyYAML tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn -q
)
echo     ✓ 依赖安装完成

echo.
echo [2/4] 检查数据集...
if not exist "dataset\images\train\*.jpg" (
    echo     正在转换数据集...
    python _run_convert.py
) else (
    echo     数据集已就绪 ✓
)

echo.
echo [3/4] 下载YOLOv8模型 (首次~6MB)...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('    模型就绪 ✓')" 2>nul

echo.
echo [4/4] 启动Web服务器...
echo.
echo ══════════════════════════════════════════════════
echo   ★ 系统启动成功！
echo   访问地址: http://127.0.0.1:5000
echo   按 Ctrl+C 停止服务
echo ══════════════════════════════════════════════════
echo.

python app.py
pause

@echo off
chcp 65001 >nul 2>&1
title 食品加工人员异常行为检测系统

echo ╔════════════════════════════════════════════════╗
echo ║   食品加工人员异常行为检测系统                  ║
echo ╚════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/3] 检查Python环境...
py -3.8 --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python 3.8 未找到，尝试默认Python...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [X] 错误: 未找到Python，请安装Python 3.8+
        pause
        exit /b 1
    )
    set PYTHON=python
) else (
    set PYTHON=py -3.8
)

echo     Python环境就绪

echo.
echo [2/3] 安装依赖包 (国内镜像)...
%PYTHON% -m pip install ultralytics flask opencv-python numpy pillow PyYAML tqdm torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple/ -q

echo.
echo [3/3] 启动Web服务器...
echo.
echo ══════════════════════════════════════════════════
echo   系统启动成功！
echo   访问地址: http://127.0.0.1:5000
echo   按 Ctrl+C 停止服务
echo ══════════════════════════════════════════════════
echo.

%PYTHON% app.py
pause

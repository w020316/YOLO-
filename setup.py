import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"[步骤] {description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ {description} 完成")
        return True
    else:
        print(f"✗ {description} 失败")
        return False

def main():
    base_dir = Path(__file__).parent
    
    print("=" * 60)
    print("食品加工人员异常行为检测系统 - 一键部署")
    print("=" * 60)
    
    steps = [
        ("安装Python依赖", f'cd "{base_dir}" && pip install -r requirements.txt'),
        ("转换数据集格式", f'python "{base_dir}/convert_dataset.py"'),
        ("训练YOLOv8模型", f'python "{base_dir}/train.py"'),
        ("启动Web服务器", f'python "{base_dir}/app.py"'),
    ]
    
    completed = []
    for i, (desc, cmd) in enumerate(steps):
        if run_command(cmd, desc):
            completed.append(desc)
            if i < len(steps) - 1:
                input("\n按回车键继续下一步...")
        else:
            if i == 3:
                break
    
    print("\n" + "=" * 60)
    print("部署完成!")
    print("=" * 60)
    print("已完成的步骤:")
    for step in completed:
        print(f"  ✓ {step}")
    print("\n访问地址: http://127.0.0.1:5000")

if __name__ == "__main__":
    main()

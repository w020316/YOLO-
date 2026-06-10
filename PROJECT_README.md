# 食品加工人员异常行为检测系统 - 完整项目交付

## 项目概述
基于YOLOv8的食品加工人员异常行为检测系统，用于检测工作人员是否佩戴帽子（正常/异常行为）。

## 检测类别
- **hat (0)**: 佩戴帽子 - 正常行为
- **nohat (1)**: 未佩戴帽子 - 异常行为

## 项目结构
```
yolov8_project/
├── app.py              # Flask Web服务器 (主入口)
├── train.py            # YOLOv8训练脚本
├── convert_dataset.py  # 数据格式转换脚本
├── _run_convert.py     # 快速数据转换脚本
├── inference.py        # 命令行推理脚本
├── validate_data.py    # 数据验证脚本
├── setup.py            # 一键部署脚本
├── 一键启动.bat         # Windows一键启动
├── start.bat           # 简单启动脚本
├── requirements.txt    # 依赖包列表
├── dataset/
│   ├── data.yaml       # 数据集配置
│   ├── images/
│   │   ├── train/      # 训练图片 (已转换)
│   │   └── val/        # 验证图片 (已转换)
│   └── labels/
│       ├── train/      # 训练标签 (已转换)
│       └── val/        # 验证标签 (已转换)
├── templates/
│   └── index.html      # Web前端页面
├── static/
│   ├── css/style.css   # 页面样式
│   ├── js/main.js      # 前端交互逻辑
│   ├── uploads/        # 上传文件目录
│   └── results/        # 检测结果目录
└── runs/detect/train/  # 模型训练输出
```

## 快速启动

### 方法1：双击一键启动 (推荐)
双击 `一键启动.bat` 文件即可自动完成所有步骤并启动Web服务。

### 方法2：命令行启动
```bash
cd c:\Users\admin\Desktop\zhuashengb1\all_data\yolov8_project

# 安装依赖
pip install -r requirements.txt

# 转换数据集 (首次运行需要)
python _run_convert.py

# 启动Web服务
python app.py
```

### 方法3：训练模型后启动
```bash
# 训练YOLOv8模型 (可选，已有预训练后备)
python train.py

# 启动Web服务
python app.py
```

## Web界面功能

访问地址：**http://127.0.0.1:5000**

### 功能特性：
1. **图片上传检测**
   - 支持拖拽上传或点击选择
   - 支持 JPG, PNG 格式
   
2. **摄像头实时检测**
   - 打开摄像头实时画面
   - 拍照后进行检测

3. **检测结果展示**
   - 可视化标注框显示
   - 绿色框 = 正常 (佩戴帽子)
   - 红色框 = 异常 (未佩戴帽子)

4. **统计信息**
   - 正常/异常人数统计
   - 总检测数量
   - 异常行为警告提示

## 技术栈
- **深度学习框架**: PyTorch + Ultralytics YOLOv8
- **Web框架**: Flask
- **图像处理**: OpenCV + PIL
- **前端**: HTML5 + CSS3 + JavaScript

## 注意事项
1. 首次运行会自动下载YOLOv8n预训练模型 (~6MB)
2. 如需自定义训练，修改train.py中的epochs参数
3. 建议使用GPU加速训练 (设置device=0)
4. 默认使用CPU模式，兼容性更好

## 故障排除
- 如果端口5000被占用，修改app.py中的端口号
- 如果模型加载失败，程序会自动使用预训练模型作为后备
- 如果数据集转换失败，检查data目录下的JSON文件是否完整

## 作者信息
- 项目名称: 食品加工人员异常行为检测系统
- 技术支持: YOLOv8 + Flask

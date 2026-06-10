# Food Processing Personnel Abnormal Behavior Detection System

基于YOLO的食品加工人员异常行为检测系统

## 项目概述

本项目使用YOLOv8深度学习模型检测食品加工人员的异常行为，包括：
- 佩戴帽子检测 (hat)
- 未佩戴帽子检测 (nohat) - 异常行为

## 项目结构

```
yolov8_project/
├── dataset/
│   ├── images/
│   │   ├── train/          # 训练集图片
│   │   └── val/            # 验证集图片
│   ├── labels/
│   │   ├── train/          # 训练集标签
│   │   └── val/            # 验证集标签
│   ├── data.yaml           # 数据集配置文件
│   └── class_names.txt     # 类别名称
├── runs/
│   └── detect/             # 训练输出目录
│       └── train/          # 训练结果
├── models/                 # 模型存储
├── configs/                # 配置文件
├── convert_dataset.py       # 数据格式转换脚本
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
└── requirements.txt         # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

将标注数据放在 `data/` 文件夹中，每个图片对应一个同名的 `.json` 标注文件。

### 标签说明
- `hat`: 佩戴帽子（正常）
- `nohat`: 未佩戴帽子（异常行为）

## 使用步骤

### 1. 数据格式转换

将Labelme格式转换为YOLO格式：

```bash
python convert_dataset.py
```

### 2. 训练模型

```bash
python train.py
```

训练参数说明：
- epochs: 训练轮数
- imgsz: 输入图片大小
- batch: 批次大小
- device: 训练设备 (0=NVIDIA GPU, cpu=CPU)

### 3. 模型验证

训练完成后可以使用 `validate_model()` 函数验证模型效果。

### 4. 推理检测

```bash
python inference.py
```

检测模式：
1. 图片检测
2. 视频检测
3. 实时摄像头检测
4. 批量检测

## 检测结果说明

- **绿色框**: hat (佩戴帽子) - 正常行为
- **红色框**: nohat (未佩戴帽子) - 异常行为
- **警告信息**: 检测到异常行为时会显示警告文字

## 模型性能

训练完成后模型保存在：
```
runs/detect/train/weights/best.pt
```

关键指标：
- mAP50: 50%IoU下的平均精度
- mAP50-95: 50%-95%IoU下的平均精度

## 硬件要求

- GPU: NVIDIA GPU with CUDA support (推荐)
- RAM: 16GB or higher
- Storage: 10GB or more

## 注意事项

1. 确保数据集中的标签名称与代码中定义的标签一致
2. 训练前检查data.yaml配置是否正确
3. 根据硬件配置调整batch大小
4. 建议训练至少100个epochs以获得较好效果

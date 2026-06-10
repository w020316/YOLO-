# 🔍 食品加工人员异常行为检测系统

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-brightgreen?logo=github&logoColor=white)

基于 **YOLOv8** 深度学习的食品加工人员安全检测系统，可实时识别工作人员是否佩戴帽子，对未佩戴帽子的异常行为进行自动告警，保障食品加工环节的卫生安全规范。

## 🌐 在线演示

👉 [点击访问在线演示](https://w020316.github.io/YOLO-/)

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🖼️ 图片检测 | 上传图片进行单张检测，标注结果可视化 |
| 🎬 视频检测 | 上传视频逐帧检测，输出标注后的结果视频 |
| 📹 摄像头实时检测 | 调用摄像头进行实时检测与告警 |
| 🚨 异常告警 | 检测到未佩戴帽子人员时自动触发告警提示 |
| 📋 检测历史 | 记录所有检测历史，支持筛选、删除、导出报告 |
| 📊 仪表盘统计 | 可视化展示检测统计、训练指标、趋势图表 |

## 📈 模型性能

| 指标 | 值 |
|------|-----|
| mAP50 | 0.866 |
| mAP50-95 | 0.43 |
| Precision | 0.923 |
| Recall | 0.817 |

## 🏷️ 检测类别

| 类别 | 标签 | 说明 | 状态 |
|------|------|------|------|
| 佩戴帽子 | `hat` | 工作人员正确佩戴帽子 | ✅ 正常 |
| 未佩戴帽子 | `nohat` | 工作人员未佩戴帽子 | ❌ 异常 |

## 🛠️ 技术栈

- **深度学习框架**：YOLOv8 (Ultralytics)
- **后端服务**：Flask
- **图像处理**：OpenCV、Pillow
- **前端界面**：HTML / CSS / JavaScript
- **模型推理**：PyTorch
- **部署**：GitHub Pages（在线演示）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

```
ultralytics>=8.0.0
opencv-python>=4.8.0
flask>=3.0.0
torch>=2.0.0
pillow>=10.0.0
PyYAML>=6.0
```

### 2. 数据准备

将 Labelme 标注数据放在 `data/` 目录下，每张图片对应一个同名的 `.json` 标注文件，然后执行格式转换：

```bash
python convert_dataset.py
```

转换后数据集结构：

```
dataset/
├── images/
│   ├── train/       # 训练集图片
│   └── val/         # 验证集图片
├── labels/
│   ├── train/       # 训练集标签
│   └── val/         # 验证集标签
├── data.yaml        # 数据集配置
└── class_names.txt  # 类别名称
```

### 3. 训练模型

```bash
python train.py
```

训练参数（可在 `train.py` 中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 30 | 训练轮数 |
| imgsz | 640 | 输入图片尺寸 |
| batch | 8 | 批次大小 |
| device | auto | 训练设备（cuda / cpu） |
| optimizer | SGD | 优化器 |
| lr0 | 0.01 | 初始学习率 |

训练完成后模型保存在 `runs/detect/train/weights/best.pt`。

### 4. 启动系统

```bash
python app.py
```

或使用快捷启动：

```bash
启动系统.bat
```

启动后访问 [http://127.0.0.1:5000](http://127.0.0.1:5000) 即可使用。

## 📡 API 接口文档

### 检测接口

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/detect` | 图片检测（上传文件） |
| POST | `/api/detect_base64` | 图片检测（Base64 编码，用于摄像头） |
| POST | `/api/batch_detect` | 批量图片检测 |
| POST | `/api/video_detect` | 视频检测 |

### 数据与历史

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/history` | 获取检测历史（支持 `limit`、`type`、`alert` 参数） |
| DELETE | `/api/history/<record_id>` | 删除指定历史记录 |
| POST | `/api/history/clear` | 清空所有历史记录 |
| GET | `/api/export_report` | 导出检测报告（CSV 格式） |

### 系统与模型

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/status` | 获取系统状态（模型、设备、数据集信息） |
| GET | `/api/dashboard` | 获取仪表盘统计数据 |
| POST | `/api/train` | 触发模型训练 |
| POST | `/api/export_model` | 导出 ONNX 模型 |
| GET | `/api/training_plots/<plot_name>` | 获取训练过程图表 |
| GET | `/api/download_result/<filename>` | 下载检测结果图片 |

### 请求示例

**图片检测：**

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -F "image=@test.jpg" \
  -F "conf=0.25"
```

**响应示例：**

```json
{
  "success": true,
  "filename": "20260610_120000_test.jpg",
  "detections": [
    {
      "class": "hat",
      "class_cn": "佩戴帽子",
      "confidence": 0.92,
      "bbox": [100, 50, 200, 180],
      "alert": null
    },
    {
      "class": "nohat",
      "class_cn": "未佩戴帽子",
      "confidence": 0.87,
      "bbox": [300, 60, 400, 190],
      "alert": "检测到未佩戴帽子的工作人员！"
    }
  ],
  "result_image": "/static/results/20260610_120000_test.jpg",
  "inference_time": 45.2
}
```

## 📁 项目结构

```
yolov8_project/
├── app.py                    # Flask 主应用（后端服务）
├── train.py                  # 模型训练脚本
├── inference.py              # 推理检测脚本
├── convert_dataset.py        # Labelme → YOLO 数据格式转换
├── validate_data.py          # 数据验证
├── setup.py                  # 安装配置
├── requirements.txt          # Python 依赖
├── yolov8n.pt                # YOLOv8 预训练权重
├── dataset/
│   ├── data.yaml             # 数据集配置
│   ├── class_names.txt       # 类别名称
│   ├── images/
│   │   ├── train/            # 训练集图片
│   │   └── val/              # 验证集图片
│   └── labels/
│       ├── train/            # 训练集标签
│       └── val/              # 验证集标签
├── runs/
│   └── detect/
│       └── train/            # 训练输出（权重、图表）
├── models/                   # 模型存储目录
├── static/
│   ├── css/style.css         # 样式表
│   ├── js/main.js            # 前端逻辑
│   ├── uploads/              # 上传文件目录
│   ├── results/              # 检测结果目录
│   └── detection_history.json # 检测历史数据
├── templates/
│   └── index.html            # 主页面模板
├── docs/
│   └── index.html            # GitHub Pages 在线演示页面
├── .github/
│   └── workflows/
│       └── pages.yml         # GitHub Pages 部署工作流
├── 启动系统.bat               # Windows 快捷启动脚本
└── README.md
```

## 📸 截图

> 系统界面截图待补充

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。

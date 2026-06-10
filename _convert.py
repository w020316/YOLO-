import os
import json
import shutil
import random
from pathlib import Path

source_dir = Path(r"c:\Users\admin\Desktop\zhuashengb1\all_data\data")
output_dir = Path(r"c:\Users\admin\Desktop\zhuashengb1\all_data\yolov8_project\dataset")

label_map = {'hat': 0, 'nohat': 1}

def convert_box(points, img_width, img_height):
    x1, y1 = points[0]
    x2, y2 = points[1]
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = abs(x2 - x1) / img_width
    height = abs(y2 - y1) / img_height
    return x_center, y_center, width, height

json_files = list(source_dir.glob('*.json'))
print(f"找到 {len(json_files)} 个JSON文件")

random.seed(42)
random.shuffle(json_files)

split_idx = int(len(json_files) * 0.8)
train_files = json_files[:split_idx]
val_files = json_files[split_idx:]

for split, files in [('train', train_files), ('val', val_files)]:
    img_dir = output_dir / 'images' / split
    label_dir = output_dir / 'labels' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for json_file in files:
        img_path = source_dir / (json_file.stem + '.jpg')
        if not img_path.exists():
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_width = data['imageWidth']
        img_height = data['imageHeight']
        shapes = data['shapes']

        yolo_labels = []
        for shape in shapes:
            label = shape['label']
            if label not in label_map:
                continue
            class_id = label_map[label]
            points = shape['points']
            x_center, y_center, width, height = convert_box(points, img_width, img_height)
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if not yolo_labels:
            continue

        shutil.copy(img_path, img_dir / img_path.name)
        
        label_path = label_dir / (json_file.stem + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        count += 1
    
    print(f"  {split}: {count} 张图片")

yaml_content = f"""path: {str(output_dir.absolute())}
train: images/train
val: images/val

nc: 2
names: ['hat', 'nohat']
"""
with open(output_dir / 'data.yaml', 'w') as f:
    f.write(yaml_content)

print(f"\n数据集转换完成!")
print(f"训练集: {len(train_files)} 个文件")
print(f"验证集: {len(val_files)} 个文件")

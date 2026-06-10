import os
import json
from pathlib import Path
from collections import defaultdict

def validate_labelme_data(data_dir):
    print("=" * 60)
    print("Labelme Data Validation Report")
    print("=" * 60)

    data_path = Path(data_dir)
    json_files = list(data_path.glob('*.json'))

    print(f"\nTotal JSON files found: {len(json_files)}")

    label_stats = defaultdict(int)
    shape_type_stats = defaultdict(int)
    issues = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_width = data.get('imageWidth', 0)
            image_height = data.get('imageHeight', 0)

            if image_width <= 0 or image_height <= 0:
                issues.append(f"{json_file.name}: Invalid image dimensions ({image_width}x{image_height})")

            for i, shape in enumerate(data.get('shapes', [])):
                label = shape.get('label', 'NO_LABEL')
                label_stats[label] += 1

                shape_type = shape.get('shape_type', 'unknown')
                shape_type_stats[shape_type] += 1

                if ' ' in label:
                    issues.append(f"{json_file.name}: Label contains space: '{label}'")

                points = shape.get('points', [])
                if len(points) != 2:
                    issues.append(f"{json_file.name}: shape[{i}] has {len(points)} points, expected 2")

        except json.JSONDecodeError as e:
            issues.append(f"{json_file.name}: JSON parse error - {e}")
        except Exception as e:
            issues.append(f"{json_file.name}: Error - {e}")

    print("\n--- Label Statistics ---")
    for label, count in sorted(label_stats.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    print("\n--- Shape Type Statistics ---")
    for shape_type, count in sorted(shape_type_stats.items(), key=lambda x: -x[1]):
        print(f"  {shape_type}: {count}")

    print("\n--- Issues Found ---")
    if issues:
        for issue in issues[:50]:
            print(f"  - {issue}")
        if len(issues) > 50:
            print(f"  ... and {len(issues) - 50} more issues")
    else:
        print("  No issues found!")

    print("\n" + "=" * 60)
    return len(issues) == 0

def validate_yolo_dataset(dataset_dir):
    print("\n" + "=" * 60)
    print("YOLO Dataset Validation Report")
    print("=" * 60)

    dataset_path = Path(dataset_dir)

    train_images = list((dataset_path / 'images' / 'train').glob('*.jpg'))
    val_images = list((dataset_path / 'images' / 'val').glob('*.jpg'))
    train_labels = list((dataset_path / 'labels' / 'train').glob('*.txt'))
    val_labels = list((dataset_path / 'labels' / 'val').glob('*.txt'))

    print(f"\nTraining set:")
    print(f"  Images: {len(train_images)}")
    print(f"  Labels: {len(train_labels)}")

    print(f"\nValidation set:")
    print(f"  Images: {len(val_images)}")
    print(f"  Labels: {len(val_labels)}")

    missing_labels = []
    empty_labels = []

    for img_file in train_images[:10]:
        label_file = dataset_path / 'labels' / 'train' / (img_file.stem + '.txt')
        if not label_file.exists():
            missing_labels.append(img_file.name)
        else:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    empty_labels.append(img_file.name)

    print(f"\nSample validation (first 10 train images):")
    if missing_labels:
        print(f"  Missing labels: {missing_labels}")
    if empty_labels:
        print(f"  Empty labels: {empty_labels}")
    if not missing_labels and not empty_labels:
        print("  All OK")

    print("\n" + "=" * 60)

def main():
    import sys

    labelme_dir = r"c:\Users\admin\Desktop\zhuashengb1\all_data\data"
    yolo_dir = r"c:\Users\admin\Desktop\zhuashengb1\all_data\yolov8_project\dataset"

    print("\nStep 1: Validate Labelme Source Data")
    validate_labelme_data(labelme_dir)

    if Path(yolo_dir).exists():
        print("\nStep 2: Validate YOLO Dataset")
        validate_yolo_dataset(yolo_dir)

if __name__ == "__main__":
    main()

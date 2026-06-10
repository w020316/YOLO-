import os
import json
import shutil
import random
from pathlib import Path

class Labelme2YOLOConverter:
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.label_map = {
            'hat': 0,
            'nohat': 1,
        }
        self.train_ratio = 0.8

    def convert_box(self, points, img_width, img_height):
        x1, y1 = points[0]
        x2, y2 = points[1]
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        return x_center, y_center, width, height

    def process_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_width = data['imageWidth']
        img_height = data['imageHeight']
        shapes = data['shapes']

        yolo_labels = []
        for shape in shapes:
            label = shape['label']
            if label not in self.label_map:
                continue
            class_id = self.label_map[label]
            points = shape['points']
            x_center, y_center, width, height = self.convert_box(points, img_width, img_height)
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return yolo_labels

    def split_dataset(self):
        json_files = list(self.source_dir.glob('*.json'))
        random.shuffle(json_files)

        split_idx = int(len(json_files) * self.train_ratio)
        train_files = json_files[:split_idx]
        val_files = json_files[split_idx:]

        return train_files, val_files

    def convert(self):
        train_files, val_files = self.split_dataset()

        for split, files in [('train', train_files), ('val', val_files)]:
            img_dir = self.output_dir / 'images' / split
            label_dir = self.output_dir / 'labels' / split
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            for json_file in files:
                img_path = self.source_dir / json_file.with_suffix('.jpg').name
                if not img_path.exists():
                    img_path = self.source_dir / json_file.with_suffix('.JPG').name
                if not img_path.exists():
                    continue

                yolo_labels = self.process_json(json_file)
                if not yolo_labels:
                    continue

                dest_img = img_dir / img_path.name
                shutil.copy(img_path, dest_img)

                label_path = label_dir / (json_file.stem + '.txt')
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))

        self.create_data_yaml()

    def create_data_yaml(self):
        yaml_content = f"""path: {str(self.output_dir.absolute())}
train: images/train
val: images/val

nc: {len(self.label_map)}
names: {list(self.label_map.keys())}
"""
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created data.yaml at: {yaml_path}")

    def create_class_names(self):
        names_path = self.output_dir / 'class_names.txt'
        with open(names_path, 'w') as f:
            for label, idx in sorted(self.label_map.items(), key=lambda x: x[1]):
                f.write(f"{idx}: {label}\n")
        print(f"Created class_names.txt at: {names_path}")

def main():
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    converter = Labelme2YOLOConverter(source_dir, output_dir)
    converter.convert()
    converter.create_class_names()

    print("\nConversion completed!")
    print(f"Dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()

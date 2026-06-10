import os
os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics_config')

from ultralytics import YOLO
import torch

def train_yolov8():
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics_config'), exist_ok=True)
    local_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
    model = YOLO(local_model)

    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "data.yaml")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    results = model.train(
        data=data_path,
        epochs=30,
        imgsz=640,
        batch=8,
        project=project_path,
        name='train',
        patience=15,
        save=True,
        save_period=10,
        device=device,
        workers=0,
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True
    )

    print("\nTraining completed!")
    print(f"Best model saved at: {results.save_dir}")
    return results

def export_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "train", "weights", "best.pt")
    model = YOLO(model_path)
    export_path = model.export(format='onnx')
    print(f"Model exported to: {export_path}")
    return export_path

def validate_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "train", "weights", "best.pt")
    model = YOLO(model_path)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "data.yaml")

    results = model.val(data=data_path, split='val')
    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    return results

if __name__ == "__main__":
    train_yolov8()

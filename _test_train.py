from ultralytics import YOLO
import os

print("Testing YOLOv8...")
try:
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")
    
    data_path = r"c:\Users\admin\Desktop\zhuashengb1\all_data\yolov8_project\dataset\data.yaml"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
    else:
        print(f"Data file found: {data_path}")
        
        results = model.train(
            data=data_path,
            epochs=5,
            imgsz=640,
            batch=4,
            project=r"c:\Users\admin\Desktop\zhuashengb1\all_data\yolov8_project\runs\detect",
            name='train',
            device='cpu',
            exist_ok=True,
            verbose=True
        )
        
        print("\nTraining completed!")
        print(f"Results: {results}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

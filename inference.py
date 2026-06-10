import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

class FoodProcessingDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = {0: 'hat', 1: 'nohat'}
        self.alert_messages = {
            'nohat': 'WARNING: Personnel not wearing hat! Abnormal behavior detected!'
        }

    def detect_image(self, image_path, save_path=None, conf=0.25):
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=False,
            show=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                detection = {
                    'class': self.class_names[cls_id],
                    'confidence': conf_score,
                    'bbox': bbox.tolist(),
                    'alert': self.alert_messages.get(self.class_names[cls_id], None)
                }
                detections.append(detection)

                if save_path:
                    img = cv2.imread(image_path)
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 0, 255) if self.class_names[cls_id] == 'nohat' else (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.class_names[cls_id]} {conf_score:.2f}"
                    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if self.alert_messages.get(self.class_names[cls_id]):
                        cv2.putText(img, self.alert_messages[self.class_names[cls_id]],
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imwrite(save_path, img)

        return detections

    def detect_video(self, video_path, output_path=None, conf=0.25):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=conf, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()

                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 0, 255) if self.class_names[cls_id] == 'nohat' else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.class_names[cls_id]} {conf_score:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if self.alert_messages.get(self.class_names[cls_id]):
                        cv2.putText(frame, self.alert_messages[self.class_names[cls_id]],
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if output_path:
                out.write(frame)

            frame_count += 1

        cap.release()
        if output_path:
            out.release()

        return frame_count

    def detect_realtime(self, camera_index=0, conf=0.25):
        cap = cv2.VideoCapture(camera_index)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=conf, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()

                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 0, 255) if self.class_names[cls_id] == 'nohat' else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.class_names[cls_id]} {conf_score:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if self.alert_messages.get(self.class_names[cls_id]):
                        cv2.putText(frame, self.alert_messages[self.class_names[cls_id]],
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Food Processing Safety Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "train", "weights", "best.pt")
    detector = FoodProcessingDetector(model_path)

    print("\n" + "="*60)
    print("Food Processing Personnel Abnormal Behavior Detection System")
    print("="*60)
    print("\n1. Detect from image")
    print("2. Detect from video")
    print("3. Real-time detection")
    print("4. Batch detection")

    choice = input("\nSelect mode (1-4): ")

    if choice == '1':
        image_path = input("Enter image path: ")
        save_path = input("Enter save path (or press Enter to skip): ")
        detections = detector.detect_image(image_path, save_path if save_path else None)
        print(f"\nDetected {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.2f}")
            if det['alert']:
                print(f"    ALERT: {det['alert']}")

    elif choice == '2':
        video_path = input("Enter video path: ")
        output_path = input("Enter output path (or press Enter to skip): ")
        frame_count = detector.detect_video(video_path, output_path if output_path else None)
        print(f"\nProcessed {frame_count} frames")

    elif choice == '3':
        camera_index = int(input("Enter camera index (default 0): ") or "0")
        detector.detect_realtime(camera_index)

    elif choice == '4':
        folder_path = input("Enter folder path: ")
        output_folder = input("Enter output folder (or press Enter to skip): ")
        output_folder = output_folder if output_folder else None

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for img_path in Path(folder_path).glob('*'):
            if img_path.suffix.lower() in image_extensions:
                save_path = None
                if output_folder:
                    save_path = os.path.join(output_folder, img_path.name)
                detections = detector.detect_image(str(img_path), save_path)
                print(f"Processed {img_path.name}: {len(detections)} detections")

if __name__ == "__main__":
    main()

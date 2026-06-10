from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from datetime import datetime
import base64
import io
import json
import glob as glob_module
import csv
import time
from PIL import Image
from ultralytics import YOLO
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['RESULT_FOLDER'] = 'static/results'
app.config['HISTORY_FILE'] = 'static/detection_history.json'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['YOLO_CONFIG_DIR'] = os.path.join(PROJECT_DIR, 'ultralytics_config')
os.makedirs(os.path.join(PROJECT_DIR, 'ultralytics_config'), exist_ok=True)
MODEL_PATH = os.path.join(PROJECT_DIR, "runs", "detect", "train", "weights", "best.pt")
FALLBACK_MODEL = os.path.join(PROJECT_DIR, "yolov8n.pt")
model = None

CLASS_NAMES = {0: 'hat', 1: 'nohat'}
CLASS_NAMES_CN = {0: '佩戴帽子', 1: '未佩戴帽子'}
ALERT_MESSAGES = {
    'nohat': '检测到未佩戴帽子的工作人员！'
}

# 中文字体支持
FONT_PATH = os.path.join(PROJECT_DIR, "static", "fonts", "simhei.ttf")
CHINESE_FONT = None

def get_chinese_font():
    global CHINESE_FONT
    if CHINESE_FONT is None:
        try:
            from PIL import ImageFont
            if os.path.exists(FONT_PATH):
                CHINESE_FONT = ImageFont.truetype(FONT_PATH, 20)
            else:
                # 尝试系统字体
                sys_fonts = [
                    "C:/Windows/Fonts/simhei.ttf",
                    "C:/Windows/Fonts/msyh.ttc",
                    "C:/Windows/Fonts/simsun.ttc",
                ]
                for fp in sys_fonts:
                    if os.path.exists(fp):
                        CHINESE_FONT = ImageFont.truetype(fp, 20)
                        break
        except Exception as e:
            print(f"Font loading failed: {e}")
    return CHINESE_FONT

def put_chinese_text(img, text, position, color=(0, 255, 0), font_size=20):
    """在OpenCV图像上绘制中文文字"""
    font = get_chinese_font()
    if font is None:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color[::-1])  # BGR to RGB
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

from PIL import ImageDraw

def get_model():
    global model
    if model is None:
        try:
            if os.path.exists(MODEL_PATH):
                model = YOLO(MODEL_PATH)
                print(f"Loaded trained model: {MODEL_PATH}")
            elif os.path.exists(FALLBACK_MODEL):
                model = YOLO(FALLBACK_MODEL)
                print(f"Using fallback model: {FALLBACK_MODEL}")
            else:
                model = YOLO('yolov8n.pt')
                print(f"Using default model: yolov8n.pt")
        except Exception as e:
            print(f"Model loading failed: {e}")
            model = None
    return model

def load_history():
    if os.path.exists(app.config['HISTORY_FILE']):
        try:
            with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(app.config['HISTORY_FILE'], 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def add_to_history(filename, detections, image_path, result_path, detection_type='image'):
    history = load_history()
    hat_count = sum(1 for d in detections if d.get('class') == 'hat')
    nohat_count = sum(1 for d in detections if d.get('class') == 'nohat')
    record = {
        'id': datetime.now().strftime('%Y%m%d%H%M%S%f'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filename': filename,
        'image_path': image_path,
        'result_path': result_path,
        'hat_count': hat_count,
        'nohat_count': nohat_count,
        'total_count': len(detections),
        'has_alert': nohat_count > 0,
        'detection_type': detection_type,
        'detections': detections
    }
    history.insert(0, record)
    if len(history) > 200:
        history = history[:200]
    save_history(history)
    return record

def draw_detection_result(image_path, results, conf):
    """绘制美观的检测结果图"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    hat_count = 0
    nohat_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            class_name = CLASS_NAMES.get(cls_id, 'unknown')
            class_name_cn = CLASS_NAMES_CN.get(cls_id, 'unknown')

            if class_name == 'hat':
                hat_count += 1
                color = (0, 200, 83)  # 绿色
                label_cn = f"{class_name_cn} {conf_score:.0%}"
            else:
                nohat_count += 1
                color = (0, 0, 255)  # 红色
                label_cn = f"{class_name_cn} {conf_score:.0%}"

            # 绘制圆角矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景
            font = get_chinese_font()
            if font:
                from PIL import Image as PILImage, ImageDraw as PILImageDraw
                pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = PILImageDraw.Draw(pil_img)

                # 标签背景
                label_bbox = font.getbbox(label_cn)
                label_w = label_bbox[2] - label_bbox[0] + 10
                label_h = label_bbox[3] - label_bbox[1] + 6

                draw.rectangle([x1, y1 - label_h - 4, x1 + label_w, y1 - 2], fill=color[::-1])
                draw.text((x1 + 5, y1 - label_h - 2), label_cn, font=font, fill=(255, 255, 255))

                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                label = f"{class_name} {conf_score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 绘制信息面板
    panel_h = 70
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # 面板文字
    font = get_chinese_font()
    if font:
        from PIL import Image as PILImage, ImageDraw as PILImageDraw
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = PILImageDraw.Draw(pil_img)

        small_font = None
        try:
            sys_fonts = [
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/msyh.ttc",
            ]
            for fp in sys_fonts:
                if os.path.exists(fp):
                    small_font = ImageFont.truetype(fp, 16)
                    break
        except:
            pass
        if small_font is None:
            small_font = font

        draw.text((10, h - panel_h + 8), f"检测统计:  正常(佩戴帽子): {hat_count}   异常(未佩戴帽子): {nohat_count}   总计: {hat_count + nohat_count}", font=small_font, fill=(255, 255, 255))
        if nohat_count > 0:
            draw.text((10, h - panel_h + 35), f"WARNING: 检测到 {nohat_count} 名未佩戴帽子的工作人员！", font=small_font, fill=(0, 0, 255))
        else:
            draw.text((10, h - panel_h + 35), "所有人员均佩戴帽子，状态正常", font=small_font, fill=(0, 200, 83))

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, f"Normal: {hat_count}  Abnormal: {nohat_count}  Total: {hat_count + nohat_count}",
                   (10, h - panel_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img, hat_count, nohat_count

def process_detection(image_path, conf=0.25):
    detections = []
    net = get_model()

    if net is None:
        return [{'error': 'Model not loaded. Please train the model first.'}]

    try:
        results = net.predict(source=image_path, conf=conf, verbose=False)

        img = cv2.imread(image_path)
        if img is None:
            return [{'error': 'Cannot read image file'}]

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(int, bbox)
                class_name = CLASS_NAMES.get(cls_id, 'unknown')
                class_name_cn = CLASS_NAMES_CN.get(cls_id, 'unknown')

                detection = {
                    'class': class_name,
                    'class_cn': class_name_cn,
                    'confidence': round(conf_score, 4),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'alert': ALERT_MESSAGES.get(class_name, None)
                }
                detections.append(detection)

        # 使用新的绘制函数
        draw_result = draw_detection_result(image_path, results, conf)
        if draw_result is not None:
            result_img, _, _ = draw_result
            result_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
            cv2.imwrite(result_path, result_img)
        else:
            result_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
            cv2.imwrite(result_path, img)

    except Exception as e:
        return [{'error': str(e)}]

    return detections

# ============ Routes ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    conf = float(request.form.get('conf', 0.25))
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    start_time = time.time()
    detections = process_detection(filepath, conf)
    inference_time = round((time.time() - start_time) * 1000, 1)

    result_image = f"/static/results/{unique_filename}"
    record = add_to_history(unique_filename, detections, f"/static/uploads/{unique_filename}", result_image, 'image')

    return jsonify({
        'success': True,
        'filename': unique_filename,
        'detections': detections,
        'result_image': result_image,
        'record_id': record.get('id'),
        'inference_time': inference_time
    })

@app.route('/api/detect_base64', methods=['POST'])
def detect_base64():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    conf = float(data.get('conf', 0.25))
    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_capture.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    start_time = time.time()
    detections = process_detection(filepath, conf)
    inference_time = round((time.time() - start_time) * 1000, 1)

    result_image = f"/static/results/{filename}"
    record = add_to_history(filename, detections, f"/static/uploads/{filename}", result_image, 'camera')

    return jsonify({
        'success': True,
        'filename': filename,
        'detections': detections,
        'result_image': result_image,
        'record_id': record.get('id'),
        'inference_time': inference_time
    })

@app.route('/api/batch_detect', methods=['POST'])
def batch_detect():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images provided'}), 400

    conf = float(request.form.get('conf', 0.25))
    results = []
    total_time = 0

    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        start_time = time.time()
        detections = process_detection(filepath, conf)
        inference_time = round((time.time() - start_time) * 1000, 1)
        total_time += inference_time

        result_image = f"/static/results/{unique_filename}"
        record = add_to_history(unique_filename, detections, f"/static/uploads/{unique_filename}", result_image, 'batch')

        results.append({
            'filename': unique_filename,
            'detections': detections,
            'result_image': result_image,
            'hat_count': sum(1 for d in detections if d.get('class') == 'hat'),
            'nohat_count': sum(1 for d in detections if d.get('class') == 'nohat'),
            'inference_time': inference_time
        })

    return jsonify({
        'success': True,
        'total': len(results),
        'total_time': round(total_time, 1),
        'results': results
    })

@app.route('/api/video_detect', methods=['POST'])
def video_detect():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400

    conf = float(request.form.get('conf', 0.25))
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    net = get_model()
    if net is None:
        return jsonify({'error': 'Model not loaded'}), 500

    cap = cv2.VideoCapture(filepath)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = f"{timestamp}_result_{filename}"
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0
    hat_total = 0
    nohat_total = 0
    sample_frames = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = net.predict(frame, conf=conf, verbose=False)
        frame_detections = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                class_name = CLASS_NAMES.get(cls_id, 'unknown')

                color = (0, 0, 255) if class_name == 'nohat' else (0, 200, 83)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf_score:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_detections += 1
                if class_name == 'hat':
                    hat_total += 1
                elif class_name == 'nohat':
                    nohat_total += 1

        total_detections += frame_detections

        # 帧信息叠加
        info_text = f"Frame: {frame_count} | Hat: {hat_total} | NoHat: {nohat_total}"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

        if frame_count % max(1, fps * 2) == 0 and len(sample_frames) < 8:
            _, buffer = cv2.imencode('.jpg', frame)
            sample_frames.append(base64.b64encode(buffer).decode('utf-8'))

        frame_count += 1

    cap.release()
    out.release()
    total_time = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        'success': True,
        'filename': unique_filename,
        'result_video': f"/static/results/{output_filename}",
        'total_frames': frame_count,
        'total_detections': total_detections,
        'hat_total': hat_total,
        'nohat_total': nohat_total,
        'has_alert': nohat_total > 0,
        'sample_frames': sample_frames,
        'processing_time': total_time
    })

@app.route('/api/status', methods=['GET'])
def status():
    model_loaded = get_model() is not None
    model_exists = os.path.exists(MODEL_PATH)
    using_custom = model_exists and model_loaded

    train_images = len(glob_module.glob(os.path.join(PROJECT_DIR, 'dataset', 'images', 'train', '*.jpg')))
    val_images = len(glob_module.glob(os.path.join(PROJECT_DIR, 'dataset', 'images', 'val', '*.jpg')))

    # 模型文件大小
    model_size = 0
    if model_exists:
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    return jsonify({
        'model_loaded': model_loaded,
        'model_path': MODEL_PATH if model_loaded else None,
        'using_custom_model': using_custom,
        'model_size_mb': round(model_size, 2),
        'classes': CLASS_NAMES,
        'classes_cn': CLASS_NAMES_CN,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset': {
            'train_images': train_images,
            'val_images': val_images
        }
    })

@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    """仪表盘数据API"""
    history = load_history()

    # 基本统计
    total_detections = len(history)
    total_hat = sum(h.get('hat_count', 0) for h in history)
    total_nohat = sum(h.get('nohat_count', 0) for h in history)
    alert_count = sum(1 for h in history if h.get('has_alert', False))

    # 按日期统计
    daily_stats = {}
    for h in history:
        date = h.get('timestamp', '')[:10]
        if date not in daily_stats:
            daily_stats[date] = {'hat': 0, 'nohat': 0, 'total': 0}
        daily_stats[date]['hat'] += h.get('hat_count', 0)
        daily_stats[date]['nohat'] += h.get('nohat_count', 0)
        daily_stats[date]['total'] += 1

    # 按检测类型统计
    type_stats = {'image': 0, 'camera': 0, 'video': 0, 'batch': 0}
    for h in history:
        dt = h.get('detection_type', 'image')
        if dt in type_stats:
            type_stats[dt] += 1

    # 训练指标
    training_metrics = {}
    results_csv = os.path.join(PROJECT_DIR, 'runs', 'detect', 'train', 'results.csv')
    if os.path.exists(results_csv):
        try:
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                epochs_data = []
                for row in reader:
                    epoch_data = {}
                    for k, v in row.items():
                        k_clean = k.strip()
                        try:
                            epoch_data[k_clean] = float(v.strip())
                        except:
                            epoch_data[k_clean] = v.strip()
                    epochs_data.append(epoch_data)
                training_metrics['epochs'] = epochs_data
                training_metrics['total_epochs'] = len(epochs_data)
                if epochs_data:
                    last = epochs_data[-1]
                    training_metrics['final_metrics'] = {
                        'mAP50': last.get('metrics/mAP50(B)', 0),
                        'mAP50-95': last.get('metrics/mAP50-95(B)', 0),
                        'precision': last.get('metrics/precision(B)', 0),
                        'recall': last.get('metrics/recall(B)', 0),
                        'box_loss': last.get('train/box_loss', 0),
                        'cls_loss': last.get('train/cls_loss', 0),
                    }
        except Exception as e:
            training_metrics['error'] = str(e)

    # 模型信息
    model_info = {}
    args_yaml = os.path.join(PROJECT_DIR, 'runs', 'detect', 'train', 'args.yaml')
    if os.path.exists(args_yaml):
        try:
            import yaml
            with open(args_yaml, 'r') as f:
                model_info = yaml.safe_load(f) or {}
        except:
            pass

    return jsonify({
        'success': True,
        'overview': {
            'total_detections': total_detections,
            'total_hat': total_hat,
            'total_nohat': total_nohat,
            'alert_count': alert_count,
            'alert_rate': round(alert_count / max(total_detections, 1) * 100, 1)
        },
        'daily_stats': daily_stats,
        'type_stats': type_stats,
        'training_metrics': training_metrics,
        'model_info': model_info
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    limit = int(request.args.get('limit', 50))
    filter_type = request.args.get('type', '')
    filter_alert = request.args.get('alert', '')

    history = load_history()

    if filter_type:
        history = [h for h in history if h.get('detection_type') == filter_type]
    if filter_alert == 'true':
        history = [h for h in history if h.get('has_alert', False)]
    elif filter_alert == 'false':
        history = [h for h in history if not h.get('has_alert', False)]

    return jsonify({
        'success': True,
        'total': len(history),
        'records': history[:limit]
    })

@app.route('/api/history/<record_id>', methods=['DELETE'])
def delete_history(record_id):
    history = load_history()
    history = [h for h in history if h.get('id') != record_id]
    save_history(history)
    return jsonify({'success': True})

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    save_history([])
    return jsonify({'success': True})

@app.route('/api/export_report', methods=['GET'])
def export_report():
    """导出检测报告CSV"""
    history = load_history()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['时间', '文件名', '检测类型', '正常数', '异常数', '总数', '是否有异常'])

    for h in history:
        writer.writerow([
            h.get('timestamp', ''),
            h.get('filename', ''),
            h.get('detection_type', 'image'),
            h.get('hat_count', 0),
            h.get('nohat_count', 0),
            h.get('total_count', 0),
            '是' if h.get('has_alert', False) else '否'
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'detection_report_{datetime.now().strftime("%Y%m%d")}.csv'
    )

@app.route('/api/train', methods=['POST'])
def start_training():
    from train import train_yolov8
    try:
        results = train_yolov8()
        return jsonify({
            'success': True,
            'message': 'Training completed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export_model', methods=['POST'])
def export_model():
    """导出ONNX模型"""
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Trained model not found'}), 404

        model = YOLO(MODEL_PATH)
        export_path = model.export(format='onnx')
        return jsonify({
            'success': True,
            'export_path': str(export_path)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_result/<filename>')
def download_result(filename):
    """下载检测结果图片"""
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/training_plots/<plot_name>')
def training_plot(plot_name):
    """获取训练过程图表"""
    plot_path = os.path.join(PROJECT_DIR, 'runs', 'detect', 'train', plot_name)
    if os.path.exists(plot_path):
        return send_file(plot_path)
    return jsonify({'error': 'Plot not found'}), 404

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    print("=" * 60)
    print("  食品加工人员异常行为检测系统")
    print("  Food Processing Safety Detection System")
    print("=" * 60)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"\n  Starting server...")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

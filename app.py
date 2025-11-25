import os
# Force all HF caches to a writable place
_cache = "/data/hf-cache" if os.getenv("HF_SPACE") else os.getenv("HF_CACHE_DIR", "/tmp/hf-cache")
for var in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "HF_CACHE_DIR", "XDG_CACHE_HOME"]:
    os.environ.setdefault(var, _cache)
os.makedirs(_cache, exist_ok=True)

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import torch
import cv2
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download
import time
from collections import deque
import threading

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_DIR", "/data/uploads")
app.config["VIDEO_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "videos")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["VIDEO_FOLDER"], exist_ok=True)

# Exercise classes
CLASSES = [
    "benchpress",
    "deadlift", 
    "squat",
    "leg_ext",
    "pushup",
    "shoulder_press"
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Performance settings
SKIP_FRAMES = 2  # Process every Nth frame (1 = every frame, 2 = every other frame, 3 = every third)
TARGET_FPS = 15  # Target FPS for video processing
INFERENCE_SIZE = 480  # Smaller = faster (try 320, 480, or 640)
JPEG_QUALITY = 85  # JPEG compression quality (70-95 recommended)

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_times = deque(maxlen=30)  # For FPS calculation

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video(filename: str) -> bool:
    VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in VIDEO_EXTENSIONS

def load_model():
    """Load the trained object detection model"""
    global model
    
    print("\n" + "=" * 60)
    print("STARTING MODEL LOAD")
    print("=" * 60)
    
    try:
        # Determine checkpoint path
        if os.getenv("HF_SPACE"):
            token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            print(f"HF_SPACE mode - Token present: {bool(token)}")
            checkpoint_path = hf_hub_download(
                repo_id="gym-vision/gymvision-model",
                filename="yolo_best.pt",
                repo_type="model",
                cache_dir=os.environ["HF_CACHE_DIR"],
                token=token   
            )
        else:
            checkpoint_path = "yolo_best.pt"
            print(f"Local mode - Looking for model at: {os.path.abspath(checkpoint_path)}")
            print(f"Model file exists: {os.path.exists(checkpoint_path)}")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model file not found at {checkpoint_path}")
                
            if os.path.exists(checkpoint_path):
                print(f"Model file size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
        
        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {device}")
        
        # Try loading with ultralytics YOLO first (recommended)
        loaded = False
        
        try:
            from ultralytics import YOLO
            print("\nAttempting to load with ultralytics YOLO...")
            model = YOLO(checkpoint_path)
            model.to(device)
            loaded = True
            print("Successfully loaded with ultralytics YOLO")
        except ImportError as e:
            print(f"ultralytics not available: {e}")
        except Exception as e:
            print(f"ultralytics loading failed: {e}")
        
        # Try torch.hub YOLOv5 as fallback
        if not loaded:
            try:
                print("\nAttempting to load with torch.hub YOLOv5...")
                model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=checkpoint_path, force_reload=False)
                model.to(device)
                loaded = True
                print("Successfully loaded with torch.hub YOLOv5")
            except Exception as e:
                print(f"torch.hub loading failed: {e}")
        
        if not loaded:
            raise RuntimeError("Failed to load model with any method")
        
        # Verify model is not None
        if model is None:
            raise RuntimeError("Model loaded but is None")
        
        # Test inference on dummy image
        print("\nTesting model inference...")
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_results = model(dummy_img)
        print(f"Model inference test passed")
        print(f"Test results type: {type(test_results)}")
        
        # Print model info
        print("\n" + "=" * 60)
        print("MODEL INFO:")
        print("=" * 60)
        print(f"Model type: {type(model)}")
        print(f"Model is None: {model is None}")
        
        if hasattr(model, 'names'):
            print(f"Model classes: {model.names}")
            print(f"Number of classes: {len(model.names)}")
            
            # Verify classes match
            if isinstance(model.names, dict):
                model_class_names = set(model.names.values())
            else:
                model_class_names = set(model.names)
            
            expected_class_names = set(CLASSES)
            
            if model_class_names != expected_class_names:
                print("\n  WARNING: Class name mismatch!")
                print(f"   Model has: {sorted(model_class_names)}")
                print(f"   Expected: {sorted(expected_class_names)}")
                print(f"   This may cause label issues")
        else:
            print("Model has no 'names' attribute")
        
        # Set confidence and IOU thresholds
        if hasattr(model, 'conf'):
            model.conf = 0.25  # Lower threshold for better detection
            print(f"\nConfidence threshold: {model.conf}")
        
        if hasattr(model, 'iou'):
            model.iou = 0.45
            print(f"IOU threshold: {model.iou}")
        
        print("=" * 60)
        print("MODEL LOADED SUCCESSFULLY")
        print("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("MODEL LOADING FAILED")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")
        model = None
        return False

def draw_detections_fast(image, detections):
    """Optimized drawing with OpenCV (faster than PIL)"""
    # Convert PIL to OpenCV if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Color mapping (BGR format for OpenCV)
    colors_bgr = {
        "benchpress": (107, 107, 255),  # Red
        "deadlift": (196, 205, 78),     # Cyan
        "squat": (209, 183, 69),        # Blue
        "leg_ext": (122, 160, 255), # Orange
        "pushup": (200, 216, 152),      # Green
        "shoulder_press": (111, 220, 247) # Yellow
    }
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = det['label']
        conf = det['confidence']
        
        color = colors_bgr.get(label, (255, 255, 255))
        
        # Draw rectangle (faster than PIL)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background and text
        text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # Text
        cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    return image

@torch.no_grad()  # OPTIMIZATION: Disable gradient computation
def detect_objects_fast(image_array, verbose=False):
    """Optimized object detection"""
    if model is None:
        return []
    
    try:
        start_time = time.time()
        
        # OPTIMIZATION: Resize image if too large
        h, w = image_array.shape[:2]
        if max(h, w) > INFERENCE_SIZE:
            scale = INFERENCE_SIZE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image_array
            scale = 1.0
        
        # Run inference with optimizations
        results = model(resized, verbose=False, half=(device.type == 'cuda'))
        
        detections = []
        
        if hasattr(results, '__len__') and len(results) > 0:
            r = results[0]
            
            # Ultralytics format
            if hasattr(r, 'boxes'):
                boxes = r.boxes
                
                for box in boxes:
                    # Extract data
                    if hasattr(box, 'xyxy'):
                        coords = box.xyxy[0].cpu().numpy()
                    else:
                        coords = box.data[0][:4].cpu().numpy()
                    
                    # Scale coordinates back to original size
                    x1, y1, x2, y2 = coords / scale
                    
                    if hasattr(box, 'conf'):
                        conf = float(box.conf[0].cpu().numpy())
                    else:
                        conf = float(box.data[0][4].cpu().numpy())
                    
                    if hasattr(box, 'cls'):
                        cls_id = int(box.cls[0].cpu().numpy())
                    else:
                        cls_id = int(box.data[0][5].cpu().numpy())
                    
                    # Get label
                    if hasattr(model, 'names'):
                        if isinstance(model.names, dict):
                            label = model.names.get(cls_id, f"class_{cls_id}")
                        else:
                            label = model.names[cls_id] if cls_id < len(model.names) else f"class_{cls_id}"
                    else:
                        label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'label': label,
                        'confidence': conf
                    })
            
            # YOLOv5 format
            elif hasattr(r, 'xyxy'):
                pred = r.xyxy[0].cpu().numpy()
                
                for det in pred:
                    x1, y1, x2, y2, conf, cls_id = det
                    
                    # Scale back
                    x1, y1, x2, y2 = np.array([x1, y1, x2, y2]) / scale
                    
                    cls_id = int(cls_id)
                    
                    if hasattr(model, 'names'):
                        label = model.names[cls_id] if cls_id < len(model.names) else f"class_{cls_id}"
                    else:
                        label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'label': label,
                        'confidence': float(conf)
                    })
        
        inference_time = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"Inference: {inference_time:.1f}ms | Detections: {len(detections)}")
        
        return detections
        
    except Exception as e:
        print(f"Detection error: {e}")
        return []

def process_frame_fast(frame, frame_count=0):
    """Optimized frame processing with frame skipping"""
    
    # OPTIMIZATION: Skip frames for performance
    if frame_count % SKIP_FRAMES != 0:
        # Return previous detections or empty frame
        return frame, []
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects
    detections = detect_objects_fast(rgb_frame)
    
    # Draw detections directly on frame
    annotated_frame = draw_detections_fast(rgb_frame, detections)
    
    return annotated_frame, detections

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/webcam_feed")
def webcam_feed():
    """Optimized webcam streaming"""
    def generate():
        if model is None:
            print("ERROR: Model not loaded")
            return
            
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return
        
        # OPTIMIZATION: Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        
        print(f"Webcam started - Processing every {SKIP_FRAMES} frame(s)")
        
        frame_count = 0
        last_detections = []
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                start_time = time.time()
                
                # Process frame (with skipping)
                if frame_count % SKIP_FRAMES == 0:
                    annotated_frame, detections = process_frame_fast(frame, frame_count)
                    last_detections = detections
                else:
                    # Use previous detections for skipped frames
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated_frame = draw_detections_fast(rgb_frame, last_detections)
                
                # OPTIMIZATION: Encode with lower quality for streaming
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), encode_param)
                frame_bytes = buffer.tobytes()
                
                # Calculate FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                avg_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"FPS: {fps:.1f} | Frame time: {avg_time*1000:.1f}ms | Detections: {len(last_detections)}")
                
                frame_count += 1
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # OPTIMIZATION: Limit frame rate
                target_frame_time = 1.0 / TARGET_FPS
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                    
        finally:
            cap.release()
            print("Webcam feed stopped")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """Analyze uploaded image"""
    if model is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 500
    
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Invalid file"}), 400

    try:
        # Read image
        image_bytes = file.read()
        filename = secure_filename(file.filename)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
        
        # Load and process
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image)
        
        # Detect (verbose for images)
        print(f"Analyzing image: {image.size}")
        detections = detect_objects_fast(image_array, verbose=True)
        
        # Draw detections
        annotated_array = draw_detections_fast(image_array.copy(), detections)
        annotated_image = Image.fromarray(annotated_array)
        
        # Save
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], annotated_filename)
        annotated_image.save(annotated_path, quality=95)
        
        # Tips
        tips = {
            "benchpress": "Feet planted, slight arch, shoulder blades retracted; control bar path.",
            "deadlift": "Hinge at hips, bar close to shins, lats tight; push the floor, don't jerk.",
            "squat": "Keep knees tracking over toes; brace your core; maintain neutral spine.",
            "legextension": "Control the movement, don't swing; focus on squeezing the quadriceps.",
            "pushup": "Keep body straight, engage core; lower chest to floor with control.",
            "shoulderpress": "Keep core tight, don't arch back excessively; press straight up."
        }
        
        detected_exercises = list(set([d['label'] for d in detections]))
        exercise_tips = [tips.get(ex, "") for ex in detected_exercises]
        
        return jsonify({
            "ok": True,
            "original_image": url_for("uploaded_file", filename=filename),
            "annotated_image": url_for("uploaded_file", filename=annotated_filename),
            "detections": detections,
            "tips": exercise_tips
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Upload video"""
    if model is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 500
    
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "No video file"}), 400
    
    file = request.files["video"]
    if not file.filename or not allowed_video(file.filename):
        return jsonify({"ok": False, "error": "Invalid video"}), 400
    
    filename = secure_filename(file.filename)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{filename}"
    save_path = os.path.join(app.config["VIDEO_FOLDER"], filename)
    file.save(save_path)
    
    return jsonify({"ok": True, "video_id": filename})

@app.route("/video_feed/<video_id>")
def video_feed(video_id):
    """Optimized video streaming"""
    if model is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 500
        
    video_path = os.path.join(app.config["VIDEO_FOLDER"], video_id)
    
    def generate():
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        last_detections = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process with frame skipping
            if frame_count % SKIP_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detect_objects_fast(rgb_frame)
                annotated_frame = draw_detections_fast(rgb_frame, detections)
                last_detections = detections
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame = draw_detections_fast(rgb_frame, last_detections)
            
            # Encode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), encode_param)
            frame_bytes = buffer.tobytes()
            
            frame_count += 1
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control playback speed
            time.sleep(1.0 / TARGET_FPS)
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Load model on startup
print("\n" + "="*60)
print("FLASK APP STARTING")
print("="*60)
model_loaded = load_model()

if model_loaded:
    print("App ready")
    print(f"Performance settings:")
    print(f"  - Inference size: {INFERENCE_SIZE}x{INFERENCE_SIZE}")
    print(f"  - Skip frames: {SKIP_FRAMES}")
    print(f"  - Target FPS: {TARGET_FPS}")
    print(f"  - JPEG quality: {JPEG_QUALITY}")
    print(f"  - Device: {device}")
else:

    print("Model failed to load")

print("="*60 + "\n")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
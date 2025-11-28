import os
# Force all HF caches to a writable place
_cache = "/data/hf-cache" if os.getenv("HF_SPACE") else os.getenv("HF_CACHE_DIR", "/tmp/hf-cache")
for var in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "HF_CACHE_DIR", "XDG_CACHE_HOME"]:
    os.environ.setdefault(var, _cache)
os.makedirs(_cache, exist_ok=True)

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io
import torch
import cv2
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download
import time
from collections import deque
import shutil

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

# OPTIMIZED Performance settings for CPU
SKIP_FRAMES = 2  # Process every 2nd frame
TARGET_FPS = 20  # Target FPS for video processing
INFERENCE_SIZE = 416  # Optimal balance for YOLOv5
JPEG_QUALITY = 80  # Slightly higher quality
CONF_THRESHOLD = 0.4  # Confidence threshold
IOU_THRESHOLD = 0.45  # NMS IOU threshold

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_times = deque(maxlen=30)
last_frame_cache = None  # Cache for frame skipping

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video(filename: str) -> bool:
    VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in VIDEO_EXTENSIONS

def clear_torch_hub_cache():
    """Clear torch hub cache to resolve compatibility issues"""
    cache_dir = torch.hub.get_dir()
    yolov5_cache = os.path.join(cache_dir, 'ultralytics_yolov5_master')
    if os.path.exists(yolov5_cache):
        print(f"Clearing YOLOv5 cache: {yolov5_cache}")
        try:
            shutil.rmtree(yolov5_cache)
            print("Cache cleared successfully")
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")

def load_model():
    """Load the trained object detection model with CPU optimizations"""
    global model
    
    print("\n" + "=" * 60)
    print("STARTING MODEL LOAD (CPU OPTIMIZED)")
    print("=" * 60)
    
    # CRITICAL: Set environment variables to prevent training
    os.environ['YOLO_VERBOSE'] = 'False'
    os.environ['ULTRALYTICS_AUTOINSTALL'] = 'False'
    
    try:
        # Where to load the checkpoint from
        if os.getenv("HF_SPACE"):
            token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            checkpoint_path = hf_hub_download(
                repo_id="gym-vision/gymvision-model",
                filename="best_v4.pt",
                repo_type="model",
                cache_dir=os.environ["HF_CACHE_DIR"],
                token=token   
            )
        else:
            checkpoint_path = "best_v4.pt"
            print(f"Local mode - Model at: {os.path.abspath(checkpoint_path)}")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model not found: {checkpoint_path}")
        
        print(f"Device: {device}")
        
        # ============ METHOD 1: ultralytics YOLO (preferred) ============
        print("\n[1/2] Loading YOLO model with ultralytics...")
        loaded = False
        try:
            from ultralytics import YOLO

            # Some versions accept task=, some don’t – be defensive
            try:
                model_ = YOLO(checkpoint_path, task="detect")
            except TypeError:
                model_ = YOLO(checkpoint_path)

            model_.to(device)

            # Force eval mode, no gradients
            if hasattr(model_, "model"):
                model_.model.eval()
                model_.model.requires_grad_(False)
                for p in model_.model.parameters():
                    p.requires_grad = False

            # Disable trainer if present
            if hasattr(model_, "trainer"):
                model_.trainer = None

            # Configure inference defaults via overrides if available
            if hasattr(model_, "overrides"):
                model_.overrides.update({
                    "mode": "predict",
                    "task": "detect",
                    "conf": CONF_THRESHOLD,
                    "iou": IOU_THRESHOLD,
                    "max_det": 10,
                    "save": False,
                    "save_txt": False,
                    "save_conf": False,
                    "save_crop": False,
                    "show": False,
                    "plots": False,
                    "half": False,
                    "device": device.type,
                    "imgsz": INFERENCE_SIZE,
                })

            # Optional FP16 on GPU
            if device.type == "cuda" and hasattr(model_, "model"):
                try:
                    model_.model.half()
                    print("  ✓ Enabled FP16 (half precision) for GPU")
                except Exception:
                    print("  ℹ FP16 not available, using FP32")

            model = model_
            loaded = True
            print("✓ Loaded YOLO model with ultralytics (inference mode locked)")
        except Exception as e1:
            print(f"✗ ultralytics YOLO loading failed: {e1}")
            loaded = False

            # ============ METHOD 2: torch.hub YOLOv5 fallback ============
            print("\n[2/2] Trying torch.hub YOLOv5 compatibility mode...")
            try:
                clear_torch_hub_cache()
                model_ = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=checkpoint_path,
                    force_reload=True,
                    device=device.type,
                    trust_repo=True,
                    skip_validation=True,
                )
                print("✓ Loaded with torch.hub (YOLOv5 compatibility)")
                model = model_
                loaded = True
            except Exception as e2:
                print(f"✗ torch.hub failed: {e2}")
                loaded = False

        if not loaded or model is None:
            raise RuntimeError("All loading methods failed. Please check your model file.")

        # Extra safety: set thresholds on YOLOv5-style models
        try:
            if hasattr(model, "overrides"):
                model.overrides.update({
                    "conf": CONF_THRESHOLD,
                    "iou": IOU_THRESHOLD,
                    "max_det": 10,
                    "save": False,
                    "save_txt": False,
                    "save_conf": False,
                    "save_crop": False,
                    "show": False,
                    "plots": False,
                    "half": False,
                })
            elif hasattr(model, "conf"):
                model.conf = CONF_THRESHOLD
                model.iou = IOU_THRESHOLD
                model.max_det = 10
        except Exception as e:
            print(f"Warning: Could not set inference parameters: {e}")
        
        # ---- Warmup inference (simple, version-safe) ----
        print("\nWarming up model (3 iterations)...")
        dummy_img = np.random.randint(0, 255, (INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
        for i in range(3):
            with torch.no_grad():
                try:
                    if hasattr(model, "predict"):
                        # Try with source= first, then positional
                        try:
                            _ = model.predict(source=dummy_img)
                        except TypeError:
                            _ = model.predict(dummy_img)
                    else:
                        _ = model(dummy_img)
                    print(f"  Warmup {i+1}/3 complete")
                except Exception as e:
                    print(f"  Warmup {i+1}/3 warning: {e}")
        
        print("\n" + "=" * 60)
        print("MODEL LOADED & OPTIMIZED")
        print(f"Inference size: {INFERENCE_SIZE}x{INFERENCE_SIZE}")
        print(f"Device: {device}")
        print(f"Skip frames: {SKIP_FRAMES}")
        print(f"Confidence threshold: {CONF_THRESHOLD}")
        print("=" * 60 + "\n")
        
        return True
    
    except Exception as e:
        print("\n" + "=" * 60)
        print("MODEL LOADING FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")
        model = None
        return False

# Pre-define colors for faster lookup (BGR format)
COLORS_BGR = {
    "benchpress": (107, 107, 255),
    "deadlift": (196, 205, 78),
    "squat": (209, 183, 69),
    "leg_ext": (122, 160, 255),
    "pushup": (200, 216, 152),
    "shoulder_press": (111, 220, 247)
}

def draw_detections_fast(image, detections):
    """Optimized drawing with minimal overhead and smart label positioning"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Get image dimensions
    img_h, img_w = image.shape[:2]
    
    # Pre-calculate text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['confidence']
        
        color = COLORS_BGR.get(label, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text = f"{label} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Smart label positioning - avoid going outside frame
        label_margin = 8
        
        # Try above the box first (original position)
        if y1 - text_h - label_margin >= 0:
            # Label fits above - place it there
            label_y1 = y1 - text_h - label_margin
            label_y2 = y1
            text_y = y1 - 4
        # If box is at top, place label below the box
        elif y2 + text_h + label_margin <= img_h:
            # Label fits below - place it there
            label_y1 = y2
            label_y2 = y2 + text_h + label_margin
            text_y = y2 + text_h + 2
        # If both fail, place inside the box at top
        else:
            label_y1 = y1
            label_y2 = y1 + text_h + label_margin
            text_y = y1 + text_h + 2
        
        # Ensure label doesn't extend beyond right edge
        label_x2 = min(x1 + text_w + 4, img_w)
        
        # Background rectangle
        cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
        
        # Text
        cv2.putText(image, text, (x1 + 2, text_y), font, font_scale, (0, 0, 0), thickness)
    
    return image

@torch.no_grad()
def detect_objects_fast(image_array, verbose=False):
    """Highly optimized object detection for CPU/GPU, version-safe."""
    if model is None:
        return []
    
    try:
        start_time = time.time()
        detections = []

        # ---- Ultralytics-style models (YOLOv8/YOLOv5) ----
        if hasattr(model, "predict"):
            # Call predict with as few kwargs as possible; config is in overrides
            try:
                results = model.predict(source=image_array)
            except TypeError:
                # Some versions use positional
                try:
                    results = model.predict(image_array)
                except TypeError:
                    # Fallback: calling the model directly
                    results = model(image_array)

            # Ultralytics v8 usually returns a list of Results
            if results is not None and hasattr(results, "__len__") and len(results) > 0:
                result = results[0]
                if hasattr(result, "boxes") and result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())

                        # Manual confidence filter to decouple from API
                        if conf < CONF_THRESHOLD:
                            continue

                        if hasattr(model, "names"):
                            label = model.names[cls_id] if cls_id < len(model.names) else f"class_{cls_id}"
                        else:
                            label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "label": label,
                            "confidence": conf,
                        })

        # ---- torch.hub YOLOv5 fallback ----
        else:
            try:
                try:
                    results = model(image_array, size=INFERENCE_SIZE, augment=False)
                except TypeError:
                    results = model(image_array)
            except Exception as e:
                print(f"YOLOv5 inference error: {e}")
                results = None

            if results is not None and hasattr(results, "__len__") and len(results) > 0:
                r = results[0]
                if hasattr(r, "xyxy"):
                    pred = r.xyxy[0].cpu().numpy()
                    for det in pred:
                        x1, y1, x2, y2, conf, cls_id = det
                        conf = float(conf)
                        cls_id = int(cls_id)

                        if conf < CONF_THRESHOLD:
                            continue

                        if hasattr(model, "names"):
                            label = model.names[cls_id] if cls_id < len(model.names) else f"class_{cls_id}"
                        else:
                            label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"

                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "label": label,
                            "confidence": conf,
                        })

        inference_time = (time.time() - start_time) * 1000.0
        if verbose:
            print(f"Inference: {inference_time:.1f}ms | Detections: {len(detections)} | Device: {device.type}")
        
        return detections

    except Exception as e:
        print(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_frame_optimized(frame, frame_count=0):
    """Optimized frame processing with intelligent caching"""
    global last_frame_cache
    
    # Skip frames - return cached result
    if frame_count % SKIP_FRAMES != 0 and last_frame_cache is not None:
        return last_frame_cache['annotated'], last_frame_cache['detections']
    
    # Convert BGR to RGB once
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects
    detections = detect_objects_fast(rgb_frame)
    
    # Draw detections
    annotated_frame = draw_detections_fast(rgb_frame.copy(), detections)
    
    # Cache result
    last_frame_cache = {
        'annotated': annotated_frame,
        'detections': detections
    }
    
    return annotated_frame, detections

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/webcam_feed")
def webcam_feed():
    """Optimized webcam streaming with aggressive performance tuning"""
    def generate():
        global last_frame_cache
        last_frame_cache = None
        
        if model is None:
            print("ERROR: Model not loaded")
            return
            
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return
        
        # AGGRESSIVE webcam optimization
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try MJPEG for speed (not all cameras support it)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except:
            pass
        
        print(f"Webcam started - Processing every {SKIP_FRAMES} frame(s)")
        
        frame_count = 0
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                start_time = time.time()
                
                # Process frame with caching
                annotated_frame, detections = process_frame_optimized(frame, frame_count)
                
                # Encode efficiently
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), encode_param)
                frame_bytes = buffer.tobytes()
                
                # Calculate FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                
                if frame_count % 30 == 0:
                    avg_time = sum(frame_times) / len(frame_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"FPS: {fps:.1f} | Frame time: {avg_time*1000:.1f}ms | Detections: {len(detections)}")
                
                frame_count += 1
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Frame rate limiting
                target_frame_time = 1.0 / TARGET_FPS
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                    
        finally:
            cap.release()
            last_frame_cache = None
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
        
        # Detect
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
            "leg_ext": "Control the movement, don't swing; focus on squeezing the quadriceps.",
            "pushup": "Keep body straight, engage core; lower chest to floor with control.",
            "shoulder_press": "Keep core tight, don't arch back excessively; press straight up."
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
    global last_frame_cache
    
    if model is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 500
        
    video_path = os.path.join(app.config["VIDEO_FOLDER"], video_id)
    
    def generate():
        global last_frame_cache
        last_frame_cache = None
        
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process with caching
            annotated_frame, detections = process_frame_optimized(frame, frame_count)
            
            # Encode
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), encode_param)
            frame_bytes = buffer.tobytes()
            
            frame_count += 1
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control playback speed
            time.sleep(1.0 / TARGET_FPS)
        
        cap.release()
        last_frame_cache = None
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Load model on startup
print("\n" + "="*60)
print("FLASK APP STARTING (CPU OPTIMIZED)")
print("="*60)
if __name__ == "__main__":
    model_loaded = load_model()
    if not model_loaded:
        print("✗ Model failed to load - check errors above")
        raise SystemExit(1)

    # Run Flask on local
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        debug=False
    )
import os
# Force all HF caches to a writable place
_cache = "/data/hf-cache" if os.getenv("HF_SPACE") else os.getenv("HF_CACHE_DIR", "/tmp/hf-cache")
for var in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "HF_CACHE_DIR", "XDG_CACHE_HOME"]:
    os.environ.setdefault(var, _cache)
os.makedirs(_cache, exist_ok=True)

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io
import base64
import torch
from datetime import datetime
from model_utils import build_model, get_transforms, CLASSES, IMG_SIZE, ALLOWED_EXTENSIONS
from huggingface_hub import hf_hub_download

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_DIR", "/data/uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Make sure the dir exists
os.makedirs(_cache, exist_ok=True)

# Global variables for model
model = None
device = torch.device("cpu" if os.getenv("HF_SPACE") else ("cuda" if torch.cuda.is_available() else "cpu"))

def load_model():
    """Load the trained model"""
    global model
    try:
        # If you set HF_SPACE, pull from HF Hub, else use local file as before.
        if os.getenv("HF_SPACE"):
            token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            print("HF token present?", bool(token)) 

            checkpoint_path = hf_hub_download(
                repo_id="gym-vision/gymvision-model",
                filename="sbd_best.pt",  
                repo_type="model",
                cache_dir=os.environ["HF_CACHE_DIR"],
                token=token   
            )
        else:
            checkpoint_path = "sbd_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Build model architecture
        state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        keyset = set(state.keys())

        if any(k.startswith("features.") for k in keyset) or "classifier.1.weight" in keyset:
            arch = "efficientnetb0"
        elif any(k.startswith("layer1.") for k in keyset) or "fc.weight" in keyset:
            arch = "resnet18"
        else:
            arch = "efficientnetb0" 

        model = build_model(arch, len(CLASSES))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded arch:", arch, "missing:", missing, "unexpected:", unexpected)
        
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_bytes):
    """Make prediction on image"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        transform = get_transforms()
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
        
        # Get results
        predicted_label = CLASSES[predicted_class.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        class_probs = {CLASSES[i]: float(prob) for i, prob in enumerate(all_probs)}
        
        return {
            "success": True,
            "prediction": predicted_label,
            "confidence": confidence_score,
            "all_probabilities": class_probs
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main prediction endpoint"""
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Read image data
        image_bytes = file.read()
        
        # Save file
        filename = secure_filename(file.filename)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
        
        # Make prediction
        result = predict_image(image_bytes)
        
        if result.get("success"):
            # Generate exercise tips (same as original)
            tips = {
                "benchpress": "Feet planted, slight arch, shoulder blades retracted; control bar path.",
                "deadlift": "Hinge at hips, bar close to shins, lats tight; push the floor, don't jerk.",
                "squat": "Keep knees tracking over toes; brace your core; maintain neutral spine.",
                "legextension": "Control the movement, don't swing; focus on squeezing the quadriceps.",
                "pushup": "Keep body straight, engage core; lower chest to floor with control.",
                "shoulderpress": "Keep core tight, don't arch back excessively; press straight up."
            }
            
            return jsonify({
                "ok": True,
                "filename": filename,
                "image_url": url_for("uploaded_file", filename=filename),  # << add this
                "prediction": {
                    "label": result["prediction"],
                    "confidence": result["confidence"]
                },
                "form": {
                    "ok": result["confidence"] > 0.7,
                    "note": "Good form detected!" if result["confidence"] > 0.7 else "Check your form."
                },
                "tip": tips.get(result["prediction"], "Keep practicing!"),
                "all_probabilities": result["all_probabilities"]
            })
        else:
            return jsonify({"ok": False, "error": result["error"]}), 500

    return jsonify({"ok": False, "error": "Unsupported file type"}), 400

# Load model on startup
load_model()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
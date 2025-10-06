# Complete Flask Application for Gym Exercise Image Classifier
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io
import base64
import torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from model_utils import build_model, get_transforms, CLASSES, IMG_SIZE, ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_DIR", "/data/uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/tmp/hf-cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Global variables for model
model = None
device = torch.device("cpu" if os.getenv("HF_SPACE") else ("cuda" if torch.cuda.is_available() else "cpu"))

def load_model():
    """Load the trained model"""
    global model
    try:
        # If you set HF_SPACE, pull from HF Hub; else use local file as before.
        if os.getenv("HF_SPACE"):
            weights_path = hf_hub_download(
                repo_id="zihinc/gymvision-resnet18",
                filename="sbd_best.pt",  
                repo_type="model",
                cache_dir=HF_CACHE_DIR   
            )
            checkpoint_path = weights_path
        else:
            checkpoint_path = "sbd_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Build model architecture
        model = build_model('efficientnetb0', len(CLASSES))
        
        # Load state dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
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
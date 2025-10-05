# Complete Flask Application for Gym Exercise Image Classifier
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
from datetime import datetime

# Configuration
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CLASSES = ['benchpress', 'deadlift', 'squat', 'legextension', 'pushup', 'shoulderpress']
IMG_SIZE = 224

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variables for model
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(model_name, num_classes):
    """Build model architecture (same as in training)"""
    if model_name.lower() == 'efficientnetb0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def load_model():
    """Load the trained model"""
    global model
    try:
        # Load checkpoint
        checkpoint_path = "sbd_best.pt"  # Your saved model file
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

def get_transforms():
    """Get image transforms for preprocessing"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

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
    app.run(debug=True, host="0.0.0.0", port=5000)
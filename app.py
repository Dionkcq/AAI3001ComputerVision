from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Stub endpoint — replace with your model prediction.
    Returns:
      - predicted exercise label
      - confidence
      - (optional) form-check segmentation stub
    """
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Prefix with timestamp to avoid collisions
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # TODO: call your classifier + (later) segmentation
        # ---- demo stub below ----
        import random
        label = random.choice(["Squat", "Bench Press", "Deadlift"])
        confidence = round(random.uniform(0.78, 0.97), 2)
        form_ok = random.choice([True, False])
        tips = {
            "Squat": "Keep knees tracking over toes; brace your core; maintain neutral spine.",
            "Bench Press": "Feet planted, slight arch, shoulder blades retracted; control bar path.",
            "Deadlift": "Hinge at hips, bar close to shins, lats tight; push the floor, don’t jerk."
        }[label]

        return jsonify({
            "ok": True,
            "filename": filename,
            "prediction": {"label": label, "confidence": confidence},
            "form": {"ok": form_ok, "note": ("Form looks good!" if form_ok else "Some cues to improve.")},
            "tip": tips
        })
    return jsonify({"ok": False, "error": "Unsupported file type"}), 400

if __name__ == "__main__":
    app.run(debug=True)
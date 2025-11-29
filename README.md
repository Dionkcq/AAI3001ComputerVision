---
title: GymVision
emoji: 🏋️
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
---

# GymVision – AAI3001 Computer Vision & Deep Learning Project (Group 15)

Real-time exercise recognition web app built with **YOLOv8**.  
The system detects and classifies 6 exercises from video frames and provides bounding boxes for the person.

**Group 15 – AAI3001 Computer Vision & Deep Learning**

- Gregory Tan  
- Tan Zheng Liang  
- Neo Chuan Zong  
- Cheok Zi Hin  
- Dion Ko

---

## Project Overview

GymVision is a object detection pipeline that recognises common gym exercises from camera/video input.

**Supported exercise classes:**

- Bench press  
- Push up  
- Shoulder press  
- Squat  
- Deadlift  
- Leg extension  

The project includes:

- A **Flask** web application (`app.py`) for image upload and live inference  
- Multiple **training notebooks** (V1 → V4) with dataset curation, ablation study, and evaluation  
- Utility scripts for splitting datasets and testing models

---

## Repository Structure

Key files and folders in this repo:

- `app.py` – Flask web server and inference API (main entry point for the app)  
- `model_utils.py` – Helper functions for loading the YOLO model and running detections  
- `ObjectDetection.ipynb` – Main notebook for **YOLOv8 training**, dataset versions (V1–V4), and ablation studies  
- `ImageClassification.ipynb` – Early image classification baseline / experiments  
- `predict.ipynb` – Offline prediction, visualisation and analysis of model outputs  
- `splitdata.py` – Script used to split the raw dataset into train/val/test folders  
- `static/` – Front-end assets (CSS, JS, icons) used by the Flask app  
- `templates/` – HTML templates for the web UI (e.g. `index.html`)  
- `requirements.txt` – Python dependencies for running the app and model  
- `Split_Dataset.zip` – Compressed version of the final dataset split (train/val/test)  
- `test_images.zip` – Sample images for quick testing of the model  
- `Dockerfile` – Docker configuration for containerised deployment (optional)

> **Note:** The large raw video/image dataset is not tracked in Git due to size; only the split dataset archive and sample images are included here.

---

## Model Versions & Training Summary

We trained several versions of YOLOv8s and iteratively improved both the data and training setup:

### V1 – First sanity check

- Dataset: `Annotated_v1`
- Goal: verify label format, class mapping, and basic performance
- Result: working detector but noisy labels and inconsistent splits

### V2 – Cleaned dataset & ablation study

- Dataset: `Annotated_v2`
- Performed a small **ablation** on:
  - Weight initialisation (COCO vs warm-start from V1)
  - Mosaic augmentation strength
  - Optimiser (AdamW vs Adam vs SGD with same LR)
- Outcome: SGD with a tuned learning rate performed best and became the default for later runs.

### V3 – Main model on combined dataset

- Dataset: `Annotated_v3` 
- Config: YOLOv8s backbone, SGD optimiser, moderate mosaic & geometric augmentations
- This is the **primary model** used by `app.py`.
- Typical performance (on held-out val set):
  - **mAP@50 ≈ 0.85**
  - **mAP@[50:95] ≈ 0.53**
  - Stronger performance on squat / leg extension, weaker on push up / shoulder press.

### V4 – Targeted push up & shoulder press improvement

- Problem: confusion matrix showed **push up** and **shoulder press** were the weakest classes.
- Approach:
  - Collected additional images for these two classes from separate pools (`pool_dataset`, `pool_dataset_cz`).
  - Ensured **no image is re-used** from V2/V3 by excluding any filename stem already seen in previous splits.
  - Used the **best V3 model** to pseudo-label these new images.
- Impact:
  - Improved per-class mAP for push up and shoulder press.
  - Slight trade-offs on some other classes, but overall mAP remained strong.

For full experimental details, see `ObjectDetection.ipynb`.

---

## Explainability & Analysis

To understand what the model looks at:

- Visualised **intermediate feature maps** (e.g. P3/P5) for different exercises.
- Used **Occlusion sensitivity** (Captum) on YOLO predictions:
  - Slide a window over the input image and measure how the detection confidence drops.
  - Confirmed that the model focuses on semantically meaningful regions (barbell, hips, shoulders, core) rather than background artefacts.
- Analysed **confusion matrices** to find common failure cases (e.g. push up misclassified as deadlift in certain camera angles).

These analyses are documented in `ObjectDetection.ipynb` and `predict.ipynb`.

---

## Online Demo (Hugging Face Spaces)

You can try the model directly in the browser on Hugging Face:

**GymVision Space:** https://huggingface.co/spaces/<your-username>/<your-space-name>

**How to use:**

1. Open the Space in your browser.
2. Upload a test image (or select one of the provided sample gym images).
3. Click **Detect**.
4. The app will return:
   - the image with bounding boxes, and
   - the predicted exercise class + confidence score.

---

## How to Run the Web Application

### 1. Clone the repository

```bash
git clone https://github.com/Dionkcq/AAI3001ComputerVision.git
cd AAI3001ComputerVision

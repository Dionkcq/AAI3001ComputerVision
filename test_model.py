import torch
import cv2
import numpy as np
from PIL import Image
import os

print("Testing YOLO model...")

# Test 1: Check if model file exists
model_path = "yolo_best.pt"
print(f"\n1. Model file check:")
print(f"   Path: {os.path.abspath(model_path)}")
print(f"   Exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    print(f"   Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

# Test 2: Load model
print(f"\n2. Loading model...")
try:
    from ultralytics import YOLO
    model = YOLO(model_path)
    print("   Loaded with ultralytics")
except Exception as e:
    print(f"   ultralytics failed: {e}")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        print("   Loaded with torch.hub")
    except Exception as e2:
        print(f"   torch.hub failed: {e2}")
        exit(1)

# Test 3: Check model properties
print(f"\n3. Model properties:")
print(f"   Type: {type(model)}")
print(f"   Has 'names': {hasattr(model, 'names')}")
if hasattr(model, 'names'):
    print(f"   Class names: {model.names}")
    print(f"   Number of classes: {len(model.names)}")

# Test 4: Create test image
print(f"\n4. Creating test image...")
test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
print(f"   Image shape: {test_img.shape}")
print(f"   Image dtype: {test_img.dtype}")

# Test 5: Run inference
print(f"\n5. Running inference...")
try:
    results = model(test_img)
    print(f"   Inference successful")
    print(f"   Results type: {type(results)}")
    print(f"   Results: {results}")
    
    # Try to access detections
    if hasattr(results, '__len__'):
        print(f"   Number of result objects: {len(results)}")
        if len(results) > 0:
            r = results[0]
            print(f"   First result type: {type(r)}")
            print(f"   Has 'boxes': {hasattr(r, 'boxes')}")
            if hasattr(r, 'boxes'):
                print(f"   Number of boxes: {len(r.boxes)}")
    
except Exception as e:
    print(f"   Inference failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test with real image (if you have one)
test_image_path = "test_exercise.jpg"  # Replace with your test image
if os.path.exists(test_image_path):
    print(f"\n6. Testing with real image: {test_image_path}")
    img = cv2.imread(test_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"   Image shape: {img_rgb.shape}")
    
    try:
        results = model(img_rgb)
        print(f"   Inference successful")
        
        # Parse results
        if hasattr(results, '__len__') and len(results) > 0:
            r = results[0]
            if hasattr(r, 'boxes'):
                boxes = r.boxes
                print(f"   Detected {len(boxes)} objects")
                
                for i, box in enumerate(boxes):
                    if hasattr(box, 'xyxy'):
                        coords = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy() if hasattr(box, 'conf') else 0
                        cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                        print(f"   Box {i}: coords={coords}, conf={conf:.3f}, class={cls}")
    except Exception as e:
        print(f"   Failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n6. No test image found at {test_image_path}")

print("\n" + "="*50)
print("Test complete!")
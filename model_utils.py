"""
Model utilities for object detection
"""
import torch
import torchvision.transforms as transforms

# Exercise classes
CLASSES = [
    "benchpress",
    "deadlift", 
    "squat",
    "leg_ext",
    "pushup",
    "shoulder_press"
]

# Image parameters
IMG_SIZE = 640  # Standard YOLO input size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def build_detection_model(model_type='yolov5', num_classes=6):
    """
    Build object detection model
    
    Args:
        model_type: Type of model ('yolov5', 'yolov8', 'fasterrcnn')
        num_classes: Number of exercise classes
    
    Returns:
        Detection model
    """
    if model_type == 'yolov5':
        # YOLOv5 model from ultralytics
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # Modify for custom classes if needed
        model.model[-1].nc = num_classes
        return model
        
    elif model_type == 'yolov8':
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        return model
        
    elif model_type == 'fasterrcnn':
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_transforms(is_train=False):
    """
    Get image transforms for detection
    
    Args:
        is_train: Whether transforms are for training
    
    Returns:
        Transform pipeline
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def nms_boxes(boxes, scores, threshold=0.45):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes
    
    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        scores: Confidence scores for each box
        threshold: IOU threshold for suppression
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    
    keep = torch.ops.torchvision.nms(boxes, scores, threshold)
    return keep.tolist()

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IOU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

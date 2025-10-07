import torch.nn as nn
from torchvision import models, transforms

# Configuration
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CLASSES = ['benchpress', 'deadlift', 'squat', 'legextension', 'pushup', 'shoulderpress']
IMG_SIZE = 224

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

def get_transforms():
    """Get image transforms for preprocessing"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from datasets import load_dataset
import numpy as np
from PIL import Image

# Metric computation functions
def compute_iou(pred_mask, true_mask, num_classes):
    """Computes Intersection over Union (IoU) for each class."""
    iou_per_class = []
    for cls in range(num_classes):
        intersection = torch.sum((pred_mask == cls) & (true_mask == cls)).item()
        union = torch.sum((pred_mask == cls) | (true_mask == cls)).item()
        if union > 0:
            iou_per_class.append(intersection / union)
    return sum(iou_per_class) / len(iou_per_class)

def compute_pixel_accuracy(pred_mask, true_mask):
    """Computes Pixel Accuracy."""
    correct = torch.sum(pred_mask == true_mask).item()
    total = torch.numel(true_mask)
    return correct / total

# Function to evaluate a model on a dataset
def evaluate_model(model_name, dataset, num_classes):
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    
    iou_scores = []
    pixel_accuracies = []
    
    # Iterate through the dataset
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            # Preprocess image
            image = sample["image"]
            inputs = processor(images=image, return_tensors="pt")
            true_mask = torch.tensor(np.array(sample["annotation"]))  # Ground truth mask
            
            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits  # (batch_size, num_classes, height, width)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0)  # Predicted mask
            
            # Compute metrics
            iou_scores.append(compute_iou(pred_mask, true_mask, num_classes))
            pixel_accuracies.append(compute_pixel_accuracy(pred_mask, true_mask))
    
    # Return average metrics
    avg_iou = np.mean(iou_scores)
    avg_pixel_acc = np.mean(pixel_accuracies)
    return avg_iou, avg_pixel_acc

# Load ADE20K dataset (test split)
print("Loading dataset...")
dataset = load_dataset("scene_parse_150", split="test[:20]")  # Use 20 samples for evaluation
num_classes = 150  # Number of classes in ADE20K

# Evaluate pre-trained models
print("Evaluating models...")
vit_model_1 = "nvidia/segformer-b0-finetuned-ade-512-512"
vit_model_2 = "nvidia/segformer-b1-finetuned-ade-512-512"

iou_vit1, acc_vit1 = evaluate_model(vit_model_1, dataset, num_classes)
iou_vit2, acc_vit2 = evaluate_model(vit_model_2, dataset, num_classes)

# Print results
print(f"Model 1 ({vit_model_1}) - IoU: {iou_vit1:.4f}, Pixel Accuracy: {acc_vit1:.4f}")
print(f"Model 2 ({vit_model_2}) - IoU: {iou_vit2:.4f}, Pixel Accuracy: {acc_vit2:.4f}")

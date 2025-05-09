import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from transformers.models.vit.modeling_vit import ViTConfig
from model_utils import ModifiedViTModel, ModifiedViTLayer
import matplotlib.pyplot as plt
from PIL import Image
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "himanshu")))

from hi_main import CIFAR100Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import importlib


# Fix the seed for the random library
random.seed(42)

module_name = "model_utils"  # Use dot notation instead of relative paths
module = importlib.import_module(module_name)
ModifiedViTModel = module.ModifiedViTModel

# Threshold values
sim_threshold = 0.9
mlp_threshold = 0.5
num_patches = 196           # 14x14 patches for ViT
data_path = '../himanshu/data'
pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"
test_size = 1000
num_workers = 16
test_batch_size = 128

# Load the model
model_path = "../himanshu/models/2025-02-27_21-25-32_fixing_loss_weight-1.5_vit_model_utils_loss-cosine^classification_lr-0.001^0.0001_st-0.9_mt-0.5_bs-64_trs-10000_tes-1000_nw-16_prevmodel-2025-02-27_19-58-51_fixing_loss_weight-1.pth"
pretrained_cifar_model = AutoModelForImageClassification.from_pretrained(pretrained_cifar_model_name)
config = ViTConfig.from_pretrained(pretrained_cifar_model_name)
config.num_labels = len(config.id2label)
model = ModifiedViTModel(config, sim_threshold, mlp_threshold)
model.load_state_dict(torch.load(model_path))
model.eval()

cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Create a directory for skipped patches visualizations
skipped_patch_dir = "skipped_patches_blackout"
os.makedirs(skipped_patch_dir, exist_ok=True)

# Number of layers in the model
num_layers = len(model.encoder.layer)

# Number of random images to visualize - strictly limit to 10
num_of_images_to_visualise = 30

# Function to blacken out skipped patches on an image
def blacken_skipped_patches(image_tensor, skipped_patches):
    """
    Create visualization with skipped patches blacked out
    
    Args:
        image_tensor: Original image tensor (C, H, W)
        skipped_patches: Binary mask of skipped patches (14, 14)
    
    Returns:
        Image with blackened patches as numpy array (H, W, C)
    """
    # Convert image tensor to numpy array and transpose to (H, W, C)
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Make a copy of the image to avoid modifying the original
    result = image.copy()
    
    # Ensure image values are between 0 and 1 for proper display
    if result.max() > 1.0:
        result = result / 255.0
    
    # For CIFAR-100, images are 32x32, and we have 14x14 patches
    h, w = result.shape[:2]
    patch_size_h, patch_size_w = h // 14, w // 14
    
    # Black out skipped patches
    for i in range(14):
        for j in range(14):
            if skipped_patches[i, j]:
                y_start, y_end = i * patch_size_h, (i + 1) * patch_size_h
                x_start, x_end = j * patch_size_w, (j + 1) * patch_size_w

                # result[y_start:y_end, x_start:x_end] =   # Red color for skipped patches
                # Red color for skipped_patches
                result[y_start:y_end, x_start:x_end] = [1.0, 0.0, 0.0]
    
    return result

# Disable gradient computation for inference
with torch.no_grad():
    # Process only one random batch
    random_batch_idx = random.randint(0, len(test_loader) - 1)
    
    for batch_idx, batch in enumerate(test_loader):
        # Skip until we reach the random batch
        if batch_idx != random_batch_idx:
            continue
            
        inputs, labels = batch
        batch_size = inputs.shape[0]
        
        # Strictly limit to exactly 10 images (or less if batch is smaller)
        if batch_size > num_of_images_to_visualise:
            img_indices = random.sample(range(batch_size), num_of_images_to_visualise)
        else:
            img_indices = list(range(batch_size))
            num_of_images_to_visualise = batch_size
            
        print(f"Selected {len(img_indices)} images from batch {batch_idx}")
        
        # Move inputs to device
        inputs = inputs.to(device)
        
        # Collect information at each layer
        layer_skipped_patches = {layer_idx: [] for layer_idx in range(num_layers)}
        
        # Forward pass through the model
        outputs = model(inputs, compute_cosine=True)
        
        # Extract skipped patches information from each layer
        for layer_idx, layer in enumerate(model.encoder.layer):
            if isinstance(layer, ModifiedViTLayer):
                # Get the pred_labels tensor which indicates skipped patches (0)
                pred_labels = layer.pred_labels.cpu().numpy()
                print(f"Layer {layer_idx} pred_labels shape: {pred_labels.shape}")
                
                # Check if pred_labels needs reshaping (if it's flattened)
                if len(pred_labels.shape) == 1:
                    # If it's flattened to (batch_size * num_patches), reshape it
                    pred_labels = pred_labels.reshape(batch_size, num_patches)
                    print(f"Reshaped pred_labels to: {pred_labels.shape}")
                
                # Mark skipped patches (where pred_labels == 0)
                skipped_patches = (pred_labels == 0)
                
                # For each selected image, extract and reshape its skipped patches
                for i, img_idx in enumerate(img_indices):
                    # Extract this image's skipped patches
                    img_skipped = skipped_patches[img_idx]
                    
                    # Reshape to 14x14 grid (patch grid)
                    img_skipped_grid = img_skipped.reshape(14, 14)
                    
                    # Store for visualization
                    layer_skipped_patches[layer_idx].append(img_skipped_grid)
        
        # Create visualizations ONLY for the selected images (limit to num_of_images_to_visualise)
        for i, img_idx in enumerate(img_indices):
            if i >= num_of_images_to_visualise:
                break  # Strict enforcement of image limit
                
            # Create a figure with subplots for each layer
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 3x4 grid for 12 layers
            axes = axes.flatten()
            
            # Get the original image
            original_img = inputs[img_idx].cpu()
            
            # Plot each layer's skipped patches
            for layer_idx in range(num_layers):
                if layer_idx >= len(axes):
                    continue
                    
                ax = axes[layer_idx]
                
                if layer_idx not in layer_skipped_patches or i >= len(layer_skipped_patches[layer_idx]):
                    # If no data for this layer/image, just show the original image
                    img_display = original_img.numpy().transpose(1, 2, 0)
                    # Normalize if needed
                    if img_display.max() > 1.0:
                        img_display = img_display / 255.0
                else:
                    # Get skipped patches mask for this layer and image
                    skipped_patches_mask = layer_skipped_patches[layer_idx][i]
                    
                    # Apply blackout to skipped patches
                    img_display = blacken_skipped_patches(original_img, skipped_patches_mask)
                    
                    # Add info about number of patches skipped
                    skip_count = skipped_patches_mask.sum()
                    skip_percentage = (skip_count / num_patches) * 100
                    ax.set_title(f"Layer {layer_idx}: {skip_count} skipped ({skip_percentage:.1f}%)")
                
                # Display the image
                ax.imshow(img_display)
                ax.axis('off')
            
            # Adjust spacing and save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(skipped_patch_dir, f"image_{i}_all_layers.png"))
            plt.close(fig)
        
        print(f"Saved visualizations for exactly {min(len(img_indices), num_of_images_to_visualise)} images in '{skipped_patch_dir}' directory.")
        break  # Process only the selected random batch

# Create a summary chart showing average skipped patches per layer
try:
    # Only calculate for layers that have data
    valid_layers = [layer_idx for layer_idx in range(num_layers) if layer_idx in layer_skipped_patches and layer_skipped_patches[layer_idx]]
    
    if valid_layers:
        # Calculate average number of skipped patches per layer
        avg_skipped_per_layer = []
        for layer_idx in valid_layers:
            # Only include images for which we have data
            valid_images = [i for i in range(min(len(img_indices), num_of_images_to_visualise)) 
                          if i < len(layer_skipped_patches[layer_idx])]
            
            if valid_images:
                layer_skips = [layer_skipped_patches[layer_idx][i].sum() for i in valid_images]
                avg_skipped = sum(layer_skips) / len(layer_skips)
                avg_skipped_per_layer.append((layer_idx, avg_skipped))
        
        # Sort by layer index for plotting
        avg_skipped_per_layer.sort(key=lambda x: x[0])
        
        # Plot average skipped patches per layer
        plt.figure(figsize=(10, 6))
        x_values = [x[0] for x in avg_skipped_per_layer]
        y_values = [x[1] for x in avg_skipped_per_layer]
        plt.bar(x_values, y_values)
        plt.xlabel("Layer Index")
        plt.ylabel("Average Number of Skipped Patches")
        plt.title("Average Skipped Patches per Layer")
        plt.xticks(x_values)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(skipped_patch_dir, "average_skipped_patches_per_layer.png"))
        plt.close()
        
        print("Created summary visualization of average skipped patches per layer.")
except Exception as e:
    print(f"Error creating summary visualization: {e}")
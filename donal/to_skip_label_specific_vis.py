import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from transformers.models.vit.modeling_vit import ViTConfig
from model_utils import ModifiedViTModel, ModifiedViTLayer
from PIL import Image
import random
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "himanshu")))

from hi_main import CIFAR100Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Fix the seed for reproducibility
random.seed(42)

# Reload the module using importlib
module_name = "model_utils"  # Use dot notation
module = importlib.import_module(module_name)
ModifiedViTModel = module.ModifiedViTModel

# Configurable parameters
sim_threshold = 0.9
mlp_threshold = 0.5
num_patches = 196           # 14x14 patches for ViT
data_path = '../himanshu/data'
pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"
test_size = 1000
num_workers = 16
test_batch_size = 128

# Define the target CIFAR-100 label for visualization (change as needed)
target_label = 0

# Load the pretrained model and configuration
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

# Create directory for visualizations
to_skip_patch_dir = "to_skip_patches_blackout"
os.makedirs(to_skip_patch_dir, exist_ok=True)

# Number of layers (from the encoder)
num_layers = len(model.encoder.layer)

# Function to blacken out to_skip patches on an image representation
def blacken_to_skip_patches(image_tensor, to_skip_patches):
    """
    Black out (red overlay) the to_skip patches in the image representation.
    
    Args:
        image_tensor: Tensor of shape (C, H, W) representing the image.
        to_skip_patches: Binary mask (14x14) indicating patches to skip.
    
    Returns:
        Numpy array of the image with blackened (red) patches.
    """
    # Convert to numpy array with shape (H, W, C)
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    result = image.copy()
    
    # Ensure image values are between 0 and 1
    if result.max() > 1.0:
        result = result / 255.0
    
    h, w = result.shape[:2]
    patch_size_h, patch_size_w = h // 14, w // 14
    
    # Black out patches by setting them red
    for i in range(14):
        for j in range(14):
            if to_skip_patches[i, j]:
                y_start, y_end = i * patch_size_h, (i + 1) * patch_size_h
                x_start, x_end = j * patch_size_w, (j + 1) * patch_size_w
                result[y_start:y_end, x_start:x_end] = [1.0, 0.0, 0.0]
    
    return result

# --- Filter the test set by target label ---
filtered_inputs = []
for inputs, labels in test_loader:
    mask = (labels == target_label)
    if mask.sum() > 0:
        selected = inputs[mask]
        filtered_inputs.append(selected)

if len(filtered_inputs) == 0:
    print(f"No images found with label {target_label}.")
    exit()

filtered_inputs = torch.cat(filtered_inputs, dim=0)
print(f"Found {filtered_inputs.shape[0]} images with label {target_label}.")

filtered_inputs = filtered_inputs.to(device)

# Containers for per-layer data (for each image)
layer_to_skip_patches = {layer_idx: [] for layer_idx in range(num_layers)}
layer_images = {layer_idx: [] for layer_idx in range(num_layers)}

# --- Forward pass: Call the encoder directly to get hidden states and masks ---
with torch.no_grad():
    encoder_outputs, _ = model.encoder(filtered_inputs, compute_cosine=True, output_hidden_states=True, output_mask=True)
    # encoder_outputs is a BaseModelOutput with hidden_states attribute
    hidden_states = encoder_outputs.hidden_states  # tuple: (embeddings, layer1, layer2, ..., layerN)
    batch_size = filtered_inputs.shape[0]

    # For each ModifiedViTLayer, extract the true_labels (for to_skip patches)
    for layer_idx, layer in enumerate(model.encoder.layer):
        true_labels = layer.true_labels.cpu().numpy()
        if len(true_labels.shape) == 1:
            true_labels = true_labels.reshape(batch_size, num_patches)
        to_skip_patches_arr = (true_labels == 0)
        
        # Save the to_skip mask for each image
        for img_idx in range(batch_size):
            img_to_skip = to_skip_patches_arr[img_idx].reshape(14, 14)
            layer_to_skip_patches[layer_idx].append(img_to_skip)
    
    # For the image representation at each layer, use the hidden states.
    # hidden_states[0] corresponds to the patch embeddings,
    # and hidden_states[1:] correspond to each layer’s output.
    for layer_idx in range(num_layers):
        rep = hidden_states[layer_idx + 1]  # shape: (batch_size, seq_length, hidden_size)
        for img_idx in range(batch_size):
            # Exclude the CLS token; take patch tokens (indices 1:), shape (196, hidden_size)
            patch_tokens = rep[img_idx, 1:, :]
            # Compute a simple representation by averaging over the hidden dimension -> (196,)
            token_avg = patch_tokens.mean(dim=1)
            # Reshape into a 14x14 grid
            grid = token_avg.reshape(14, 14)
            # Normalize grid to [0,1]
            grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
            # Convert single-channel to 3-channel (stack the same channel)
            img_rep = torch.stack([grid, grid, grid], dim=0)  # shape: (3, 14, 14)
            layer_images[layer_idx].append(img_rep)

# --- Visualization for each filtered image ---
for img_idx in range(filtered_inputs.shape[0]):
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        # Get the intermediate representation and corresponding to_skip mask
        rep_img = layer_images[layer_idx][img_idx]  # shape: (3,14,14)
        to_skip_mask = layer_to_skip_patches[layer_idx][img_idx]
        
        # Apply blackout: note that our rep_img is small (14x14), so visualization will be coarse.
        img_display = blacken_to_skip_patches(rep_img, to_skip_mask)
        
        # Count patches marked to skip and compute percentage
        skip_count = to_skip_mask.sum()
        skip_percentage = (skip_count / num_patches) * 100
        ax.set_title(f"Layer {layer_idx}: {skip_count} to_skip ({skip_percentage:.1f}%)")
        ax.imshow(img_display, interpolation='nearest')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(to_skip_patch_dir, f"filtered_image_{img_idx}_all_layers.png"))
    plt.close(fig)

print(f"Saved visualizations for all images with label {target_label} in '{to_skip_patch_dir}' directory.")

# --- Create summary chart: Average number of to_skip patches per layer ---
try:
    avg_to_skip_per_layer = []
    for layer_idx in range(num_layers):
        skips = [layer_to_skip_patches[layer_idx][i].sum() for i in range(len(layer_to_skip_patches[layer_idx]))]
        avg_to_skip = sum(skips) / len(skips)
        avg_to_skip_per_layer.append((layer_idx, avg_to_skip))
    
    avg_to_skip_per_layer.sort(key=lambda x: x[0])
    
    plt.figure(figsize=(10, 6))
    x_values = [x[0] for x in avg_to_skip_per_layer]
    y_values = [x[1] for x in avg_to_skip_per_layer]
    plt.bar(x_values, y_values)
    plt.xlabel("Layer Index")
    plt.ylabel("Average Number of to_skip Patches")
    plt.title("Average to_skip Patches per Layer (for images with target label)")
    plt.xticks(x_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(to_skip_patch_dir, "average_to_skip_patches_per_layer.png"))
    plt.close()
    
    print("Created summary visualization of average to_skip patches per layer.")
except Exception as e:
    print(f"Error creating summary visualization: {e}")

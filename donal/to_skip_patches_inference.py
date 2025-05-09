import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
from transformers.models.vit.modeling_vit import ViTConfig
import numpy as np
import torch
from model_utils import ModifiedViTModel, ModifiedViTLayer
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "himanshu")))

from hi_main import CIFAR100Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import importlib

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

# Create a directory for to_skip patch heatmaps
to_skip_heatmap_dir = "to_skip_heatmaps"
os.makedirs(to_skip_heatmap_dir, exist_ok=True)

# Initialize dictionary to store per-layer to_skip patch counts
layer_to_skip_patch_counts = {layer_idx: np.zeros(num_patches, dtype=int) for layer_idx in range(len(model.encoder.layer))}

# Disable gradient computation for inference
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        inputs, labels = batch
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs, compute_cosine=True)

        # Extract per-patch skip information for each layer
        for layer_idx, layer in enumerate(model.encoder.layer):
            if isinstance(layer, ModifiedViTLayer):
                # Get the pred_labels tensor which should contain information about to_skip patches
                true_labels = layer.true_labels.cpu().numpy()  # Shape: (batch_size, 196)
                
                #  to_skip patches are marked with 0 in pred_labels
                to_skip_patches = (true_labels == 0)
                
                batch_size = inputs.shape[0]
                to_skip_patches = to_skip_patches.reshape(batch_size, num_patches)
                
                # Sum over the batch dimension to count per-patch skips
                layer_to_skip_patch_counts[layer_idx] += to_skip_patches.sum(axis=0)
        
        # Optional: Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * test_batch_size} samples...")

# Normalize and create heatmaps for to_skip patches
for layer_idx, to_skip_count in layer_to_skip_patch_counts.items():
    # Normalize by the total number of samples
    normalized_to_skip = to_skip_count / test_size
    
    # Reshape into 14x14 grid for visualization
    skip_map = normalized_to_skip.reshape(14, 14)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(skip_map, annot=False, cmap="Blues", linewidths=0.5)
    plt.title(f"Patch Skip Frequency Heatmap - Layer {layer_idx}")
    plt.xlabel("Patch Index (X-axis)")
    plt.ylabel("Patch Index (Y-axis)")
    
    # Save the heatmap
    heatmap_path = os.path.join(to_skip_heatmap_dir, f"layer_{layer_idx}_to_skip_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

# Create combined visualization (optional) - to_skip vs. total patches
print(f"Saved all to_skip patch heatmaps in '{to_skip_heatmap_dir}' directory.")

# Optional: Calculate and print summary statistics
for layer_idx, to_skip_count in layer_to_skip_patch_counts.items():
    total_skips = to_skip_count.sum()
    avg_skips_per_patch = total_skips / num_patches
    max_skips = to_skip_count.max()
    max_skip_patch = np.argmax(to_skip_count)
    max_skip_patch_coords = (max_skip_patch % 14, max_skip_patch // 14)
    
    print(f"Layer {layer_idx} statistics:")
    print(f"  Total to_skip patches: {total_skips}")
    print(f"  Average skips per patch: {avg_skips_per_patch:.2f}")
    print(f"  Most frequently to_skip patch: {max_skip_patch} at position {max_skip_patch_coords} with {max_skips} skips")
    print()
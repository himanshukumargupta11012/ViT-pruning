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


cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)   # Initialize the image processor for CIFAR-100 data pre-processing
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True)


# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Inference

# Create a directory for heatmaps
heatmap_dir = "misclassified_heatmaps"
os.makedirs(heatmap_dir, exist_ok=True)

# Initialize dictionary to store per-layer patch error counts
layer_patch_error_counts = {layer_idx: np.zeros(num_patches, dtype=int) for layer_idx in range(len(model.encoder.layer))}

# Disable gradient computation for inference
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch  # Unpacking the tuple
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs, compute_cosine=True)

        # Extract per-patch misclassification information for each layer
        for layer_idx, layer in enumerate(model.encoder.layer):  
            if isinstance(layer, ModifiedViTLayer):
                true_labels = layer.true_labels.cpu().numpy()  # Shape: (batch_size, 196)
                pred_labels = layer.pred_labels.cpu().numpy()  # Shape: (batch_size, 196)

                misclassified = (true_labels != pred_labels)    # 1D array of shape (batch_size * 196,)

                # Reshape misclassified array to match (batch_size, 196) before summing
                batch_size = inputs.shape[0]  # Get batch size dynamically
                misclassified = misclassified.reshape(batch_size, 196)  # Reshape to (batch_size, 196)

                # Sum over the batch dimension to count per-patch errors
                layer_patch_error_counts[layer_idx] += misclassified.sum(axis=0)



# Normalize and save heatmaps
for layer_idx, patch_error_count in layer_patch_error_counts.items():
    # Normalize values to stay below 1
    patch_error_count = patch_error_count / test_size

    # Reshape into 14x14 grid for visualization
    error_map = patch_error_count.reshape(14, 14)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(error_map, annot=False, cmap="Reds", linewidths=0.5)
    plt.title(f"Patch Misclassification Heatmap - Layer {layer_idx}")
    plt.xlabel("Patch Index (X-axis)")
    plt.ylabel("Patch Index (Y-axis)")

    # Save the heatmap
    heatmap_path = os.path.join(heatmap_dir, f"layer_{layer_idx}_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

print(f"Saved all heatmaps in '{heatmap_dir}' directory.")





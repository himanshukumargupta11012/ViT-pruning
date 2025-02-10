from transformers.models.vit.modeling_vit import ViTModel, ViTAttention, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import ViTForImageClassification,  AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import torch.optim as optim
import os
from himanshu_miniproject.himanshu_main import CIFAR100Dataset

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from tqdm import tqdm
import torch.utils.data as data_utils
from ptflops import get_model_complexity_info
import random

print("--------- Random Pruning Model ---------- ")


class RandomPruningViTLayer(ViTLayer):
    def __init__(self, config, num_pruned_patches):
        super().__init__(config)
        self.num_pruned_patches = num_pruned_patches
        self.threshold = 0.05  # Similar to the threshold in ModifiedViTLayer
        self.loss = 0
        self.skip_ratio = 0
        self.mlp_accuracy_arr = None

    def forward(
        self, hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        compute_cosine: bool = False
    ):
        batch_size, num_patches, hidden_size = hidden_states.shape
        num_pruned_patches = min(self.num_pruned_patches, num_patches - 1)  # Ensure we don't prune more than available patches

        # Preserve the CLS token
        cls_token = hidden_states[:, :1, :]
        
        # Randomly select patches to keep, excluding the CLS token
        boolean_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=hidden_states.device)
        for i in range(batch_size):
            keep_indices = random.sample(range(1, num_patches), num_patches - num_pruned_patches)
            boolean_mask[i, keep_indices] = True

        # Ensure CLS token is always included
        boolean_mask[:, 0] = True
        output = hidden_states.clone()

        # Apply mask and process only unpruned patches
        for i in range(batch_size):
            output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)

        # Calculate skip ratio
        self.skip_ratio = torch.sum(~boolean_mask).item() / (batch_size * num_patches)

        return (output, )


class RandomPruningViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig, num_pruned_patches_per_layer):
        super().__init__(config)
        self.layer = nn.ModuleList([RandomPruningViTLayer(config, num_pruned_patches=num_pruned_patches) 
                                    for num_pruned_patches in num_pruned_patches_per_layer])

class RandomPruningViTModel(ViTModel):
    def __init__(self, config: ViTConfig, num_pruned_patches_per_layer):
        super().__init__(config)
        self.encoder = RandomPruningViTEncoder(config, num_pruned_patches_per_layer)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        # Call the superclass forward to handle encoding, but now using RandomPruningViTEncoder
        outputs = super().forward(pixel_values, output_attentions=output_attentions, 
                                  output_hidden_states=output_hidden_states, return_dict=return_dict)
        # Add a classifier layer to compute logits from CLS token
        logits = self.classifier(outputs['last_hidden_state'][:, 0])
        output = lambda: None
        setattr(output, 'logits', logits)

        return output




# Initialize the RandomPruningViTModel with avg_pruning_counts
num_labels = 100  # Assuming CIFAR-100 dataset
avg_pruning_counts = [0, 0, 0, 8, 20, 46, 149, 174, 154, 157, 7, 5]  # Define number of patches to prune per layer
config = ViTConfig.from_pretrained("Ahmed9275/Vit-Cifar100")
config.num_labels = num_labels
random_pruning_vit_model = RandomPruningViTModel(config, num_pruned_patches_per_layer=avg_pruning_counts)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cifar_processor = AutoImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100")   # Initialize the image processor for CIFAR-100 data pre-processing
pretrained_cifar_model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")


# Here, instead of defining the custom dataset class for CIFAR-100, we  import and use it


# Function to evaluate the random pruning model
def test_random_pruning(model, dataloader, progress=True):
    model.eval()
    total_correct = 0
    each_layer_skip = torch.zeros(len(model.encoder.layer))
    
    data_size = len(dataloader.dataset)
    with torch.no_grad():
        if progress:
            dataloader = tqdm(dataloader, desc="Testing Random Pruning", leave=False)
        
        for (inputs, labels) in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            logits = outputs.logits
            predicted_class_indices = logits.argmax(dim=-1)

            # Track the skip ratios from each layer for analysis
            for i, vit_layer in enumerate(model.encoder.layer):
                each_layer_skip[i] += vit_layer.skip_ratio.item()
            
            total_correct += torch.sum(predicted_class_indices == labels).item()
    
    print(f"Random Pruning Skip ratio per layer: {each_layer_skip / data_size}")
    accuracy = total_correct / data_size
    print(f"Accuracy for Random Pruning Model: {accuracy:.4f}")
    return accuracy



# Define the testing parameters
load_pretrained = False            # Flag to load a pre-trained model
loss_type = 'cosine'
lr = 2e-2
data_path = 'data'
batch_size = 64
train_size = 10000
test_size = 1000
pretrained_model_path = os.path.join('models', 'mlp_class_10000.pth')


# Load the pretrained weights as with the modified model
if not load_pretrained:
    new_state_dict = {}
    for key in pretrained_cifar_model.state_dict().keys():
        new_key = key.replace('vit.', '')
        new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

    random_pruning_vit_model.load_state_dict(new_state_dict, strict=False)
else:
    random_pruning_vit_model.load_state_dict(torch.load(pretrained_model_path))

random_pruning_vit_model.to(device)


# Initialize datasets and dataloaders (same as for modified model)
train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Run the test on the random pruning model
test_accuracy = test_random_pruning(random_pruning_vit_model, test_loader, progress=True)

print(f"Test accuracy of the Random Pruning Model: {test_accuracy:.4f}")

# Calculating FLOPs for both models
input_size = (3, 224, 224)
flops_random, params_random = get_model_complexity_info(random_pruning_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)

print(f"FLOPs for Random Pruning Model: {flops_random}")
print(f"FLOPs for Original Pruning Model: {flops_ori}")



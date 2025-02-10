from transformers.models.vit.modeling_vit import ViTModel, ViTAttention, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import ViTForImageClassification,  AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import torch.optim as optim
import os

from torch.utils.data import Dataset, DataLoader, Subset
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
        self.skip_ratio = torch.tensor(0.0)  # Initialize as a tensor
        self.mlp_accuracy_arr = None

    def forward(
        self, hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        compute_cosine: bool = False
    ):
        batch_size, num_patches, hidden_size = hidden_states.shape
        num_pruned_patches = min(self.num_pruned_patches, num_patches - 1)  # Ensure we don't prune more than available patches, also ensure that if we 
        # Preserve the CLS token
        cls_token = hidden_states[:, :1, :]
        
        # Randomly select patches to keep, excluding the CLS token
        boolean_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=hidden_states.device)
        for i in range(batch_size):
            # Ensure there's at least one patch to keep in addition to the CLS token
            num_to_keep = max(1, num_patches - num_pruned_patches - 1)
            keep_indices = random.sample(range(1, num_patches), num_to_keep)
            boolean_mask[i, keep_indices] = True

        # Ensure CLS token is always included
        boolean_mask[:, 0] = True
        output = hidden_states.clone()

        # Apply mask and process only unpruned patches
        for i in range(batch_size):
            output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)

        # Calculate skip ratio
        self.skip_ratio = torch.sum(~boolean_mask).float() / (batch_size * num_patches)

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
num_labels = 100                                                     # Assuming CIFAR-100 dataset

# Number of patches to prune per layer (obtained from our pruning method for threshold = 0.05)
avg_pruning_counts = [0, 0, 0, 8, 20, 46, 149, 174, 154, 157, 7, 5]  
config = ViTConfig.from_pretrained("Ahmed9275/Vit-Cifar100")
config.num_labels = num_labels
random_pruning_vit_model = RandomPruningViTModel(config, num_pruned_patches_per_layer=avg_pruning_counts)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cifar_processor = AutoImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100")   # Initialize the image processor for CIFAR-100 data pre-processing
pretrained_cifar_model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")


# Custom dataset class for CIFAR-100 that integrates with AutoImageProcessor for processing
class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, size=None, processor=None):
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True)           # Load CIFAR-100 dataset from torchvision
        if size is not None:
            self.dataset = Subset(self.dataset, torch.arange(size))                                  # Optionally, load only count number of samples
        
        self.processor = processor                                                                    # Store the processor (used to preprocess images before passing to the model)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx] 
        
        # Preprocess the image and convert it to tensor format suitable for the model
        inputs = self.processor(images=image, return_tensors="pt")
        # Return the processed pixel values and label
        return inputs['pixel_values'].squeeze(0), label 

# Modify the test function to average out the random pruning effect over multiple runs
def test_random_pruning(model, dataloader, num_runs=5, progress=True):
    total_accuracy = 0.0
    total_skip_ratios = torch.zeros(len(model.encoder.layer))
    data_size = len(dataloader.dataset)  # Store dataset length before applying tqdm

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        model.eval()
        run_total_correct = 0
        run_layer_skip_ratios = torch.zeros(len(model.encoder.layer))
        
        with torch.no_grad():
            if progress:
                dataloader_iter = tqdm(dataloader, desc=f"Testing Random Pruning Run {run + 1}", leave=False)
            else:
                dataloader_iter = dataloader
            
            for (inputs, labels) in dataloader_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                logits = outputs.logits
                predicted_class_indices = logits.argmax(dim=-1)

                # Track the skip ratios for each layer
                for i, vit_layer in enumerate(model.encoder.layer):
                    run_layer_skip_ratios[i] += vit_layer.skip_ratio.item()
                
                # Count the correct predictions
                run_total_correct += torch.sum(predicted_class_indices == labels).item()
        
        # Calculate accuracy for this run
        run_accuracy = run_total_correct / data_size
        print(f"Accuracy for Run {run + 1}: {run_accuracy:.4f}")
        print(f"Layer-wise Skip Ratios for Run {run + 1}: {run_layer_skip_ratios / len(dataloader)}")
        
        # Accumulate accuracy and skip ratios
        total_accuracy += run_accuracy
        total_skip_ratios += run_layer_skip_ratios / len(dataloader)

    # Calculate the average accuracy and skip ratios over all runs
    avg_accuracy = total_accuracy / num_runs
    avg_skip_ratios = total_skip_ratios / num_runs
    print(f"\nAverage Accuracy over {num_runs} runs: {avg_accuracy:.4f}")
    print(f"Average Skip Ratios per layer over {num_runs} runs: {avg_skip_ratios}")
    
    return avg_accuracy, avg_skip_ratios



# Define the testing parameters
load_pretrained = False            # Flag to load a pre-trained model
data_path = 'data'
test_size = 1000
batch_size = 64
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
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Run the test on the random pruning model with multiple runs to average out randomness
num_test_runs = 5  # Set the number of runs for averaging
avg_accuracy, avg_skip_ratios = test_random_pruning(random_pruning_vit_model, test_loader, num_runs=num_test_runs, progress=True)
print(f"\nFinal Average Accuracy of the Random Pruning Model: {avg_accuracy:.4f}")
print(f"Final Average Skip Ratios per layer: {avg_skip_ratios}")


# Calculating FLOPs for both models
input_size = (3, 224, 224)
flops_random, params_random = get_model_complexity_info(random_pruning_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
print(f"FLOPs for Random Pruning Model: {flops_random}")
print(f"FLOPs for Original Pruning Model: {flops_ori}")



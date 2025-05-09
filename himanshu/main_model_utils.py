import time
import numpy as np
from ptflops import get_model_complexity_info
from torch import nn
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
from torchvision.datasets import ImageFolder
import pandas as pd

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        alpha: Balancing factor for class imbalance (0 < alpha < 1)
        gamma: Focusing parameter (higher = more focus on hard samples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        # Compute standard Binary Cross-Entropy Loss
        bce_loss = nn.BCELoss(reduction='none')(probs, targets)
        
        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t = p for y=1, (1-p) for y=0
        
        # Apply Focal Loss scaling factor
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        
        # Apply class balancing weight (alpha)
        loss = self.alpha * targets * loss + (1 - self.alpha) * (1 - targets) * loss

        return loss.mean()
        


class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, size=None, processor=None):
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True)           # Load CIFAR-100 dataset from torchvision
        if size is not None:
            indices = torch.randperm(len(self.dataset))[:size]
            self.dataset = Subset(self.dataset, indices)
        
        self.processor = processor      # Store the processor (used to preprocess images before passing to the model)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx] 
        
        # Preprocess the image and convert it to tensor format suitable for the model
        inputs = self.processor(images=image, return_tensors="pt")
        # Return the processed pixel values and label
        return inputs['pixel_values'].squeeze(0), label  



class TinyImageNetDataset(Dataset):
    def __init__(self, root, train=True, size=None, processor=None):
        """
        root: path to the `tiny-imagenet-200` directory
        train: if True, loads from root/train; else from root/val
        size: optional int, limits dataset to first `size` samples
        processor: a Hugging-Face image processor, e.g. ViTFeatureExtractor
        """
        split = "train" if train else "val"
        folder = os.path.join(root, split)

        self.dataset = ImageFolder(root=folder)

        if size is not None:
            indices = torch.randperm(len(self.dataset))[:size]
            self.dataset = Subset(self.dataset, indices)

        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        else:
            pixel_values = transforms.ToTensor()(image)

        return pixel_values, label
    


# Training function to train the model over multiple epochs
def train(model, train_loader, test_loader, device, log_file, save_path, num_epochs=10, loss_type='both', lr=1e-4):
    model.train()
    criterion = nn.CrossEntropyLoss()
    cosine_loss_ratio = 1

    best_val_accuracy = 0.0
    best_model = None

    if loss_type == "cosine":
        model.mlp_train()
    elif loss_type == "classification":
        model.vit_train()
    elif loss_type == "both":
        model.vit_mlp_train()

    if save_path:
        model_save_path = f"models/{save_path}.pth"
    

    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)

    temp_train_loader = train_loader
    for epoch in range(num_epochs):
        model.train()  # Re-enable training mode at the start of each epoch

        if loss_type == "alternate":
            if epoch % 3 == 0:
                model.mlp_train()
            else:
                model.vit_train()
        
        running_loss = 0.0

        tqdm_update_step = len(train_loader) / 100
        train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False, ncols=100)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)                

            logits = model(inputs).logits

            if loss_type in ['classification', 'both']:
                classification_loss = criterion(logits, labels)

            # Accumulate cosine similarity losses from each DHSLayer
            if loss_type in ['cosine', 'both']:
                cosine_loss = 0.0
                for i, vit_layer in enumerate(model.encoder.layer):
                    cosine_loss += vit_layer.loss

            if loss_type == "alternate":
                total_loss = 0
                if epoch % 3 == 0:
                    for i, vit_layer in enumerate(model.encoder.layer):
                        total_loss += vit_layer.loss
                else:
                    total_loss = criterion(logits, labels)

            
            # Calculate the total loss as a weighted sum of classification and cosine similarity losses
            if loss_type == 'classification':
                total_loss = classification_loss
            elif loss_type == 'cosine':
                total_loss = cosine_loss
            elif loss_type == 'both':
                total_loss = classification_loss + cosine_loss_ratio * cosine_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate running loss for this batch
            running_loss += total_loss.item()

            # Update the progress bar with current batch loss
            if (batch_idx + 1) % tqdm_update_step == 0:
                train_loader.update(tqdm_update_step)
            # train_loader.set_postfix({'batch_loss': total_loss.item()})

        val_accuracy, mlp_val_accuracy = test(model, test_loader, device, log_file, full_testing=True)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            if save_path:
                torch.save(best_model, model_save_path)

        # write_N_print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
        write_N_print(f"Test accuracy after {epoch + 1} epochs: {val_accuracy:.2%}\n", log_file)

    # accuracy, mlp_accuracy = test(model, temp_train_loader, full_testing=True)
    write_N_print(f"Best accuracy: {best_val_accuracy * 100}%\n", log_file)


def get_complexity(model, data_loader, processor, device):
    total_flops = 0
    total_time = 0.0
    model.eval()

    with torch.no_grad():
        for inp, tar in tqdm(data_loader, desc="Evaluating", ncols=100):
            batch = processor(inp, return_tensors="pt")
            images = batch['pixel_values'].to(device)
            labels = tar.to(device)

            start_time = time.time()

            # Loop through each image individually becasuse of ptflops structure
            for img in images:
                img = img.unsqueeze(0)  # add batch dim -> shape: (1, C, H, W)
                input_res = tuple(img.shape[1:])  # (C, H, W)

                input_constructor = lambda x: {'pixel_values': img}
                flops, params = get_model_complexity_info(
                    model, 
                    input_res, 
                    input_constructor=input_constructor,
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False
                )
                total_flops += flops

            elapsed = time.time() - start_time
            total_time += elapsed

    gflops = total_flops / 1e9
    gflops_per_sec = gflops / total_time if total_time > 0 else 0

    print(f"\nTotal Inference Time: {total_time:.2f} seconds")
    print(f"Total FLOPs: {gflops:.2f} GFLOPs")
    print(f"Throughput: {gflops_per_sec:.2f} GFLOPS")


# Function to evaluate the model on the test data
def test(model, dataloader, device, log_file, full_testing=False):
    model.eval()
    total_correct = 0
    if full_testing:
        total_correct_mlp = torch.zeros(len(model.encoder.layer), 2, 2)

    data_size = len(dataloader.dataset)
    with torch.no_grad():             # Disable gradient computation for efficiency
        dataloader = tqdm(dataloader, desc=f"Testing", leave=False, ncols=100)
        for (inputs, labels) in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if full_testing:
                outputs = model(inputs, compute_cosine=True)
            else:
                outputs = model(inputs)
            logits = outputs.logits

            predicted_class_indices = logits.argmax(dim=-1)    # Get the class index with the highest logit score

            # Track the skip ratios from each layer
            for i, vit_layer in enumerate(model.encoder.layer):
                if full_testing and hasattr(vit_layer, 'mlp_accuracy_arr'):
                    total_correct_mlp[i] += vit_layer.mlp_confusion_matrix
                    
            total_correct += torch.sum(predicted_class_indices == labels).item()

    if full_testing:
        each_layer_skip = (total_correct_mlp.sum(dim=[1], keepdim=True) / total_correct_mlp.sum(dim=[1, 2], keepdim=True))[:, 0, 0]
        
        mlp_accuracy = (torch.sum(total_correct_mlp[:, 1, 1]) + torch.sum(total_correct_mlp[:, 0, 0])) / (torch.sum(total_correct_mlp) + 1e-16)
        mlp_confusion_mat = (total_correct_mlp / (total_correct_mlp.sum(dim=[-2, -1], keepdim=True) + 1e-16)).tolist()
        mlp_accuracy_arr = (total_correct_mlp[:, 1, 1] + total_correct_mlp[:, 0, 0]) / total_correct_mlp.sum(dim=[1, 2])

        write_N_print(f"Skip %: {each_layer_skip.mean():.2%}\nOverall accuracy of MLP: {mlp_accuracy:.2%}", log_file)

        # skip ratio and mlp accuracy for each layer
        results_df = pd.DataFrame([np.array(each_layer_skip), np.array(mlp_accuracy_arr)], index=["Skip ratio", "MLP accuracy"])
        results_df.columns = [f"L {i}" for i in range(len(mlp_accuracy_arr))]
        results_df.index = ["Skip ratio", "MLP accuracy"]
        results_df = (results_df * 100).round(1)
        write_N_print(results_df.to_string(), log_file)

        # confusion matrix for each layer
        mlp_confusion_mat = np.array(mlp_confusion_mat)
        new_shape = (mlp_confusion_mat.shape[0] * 2 - 1, mlp_confusion_mat.shape[1], mlp_confusion_mat.shape[2])
        result = np.full(new_shape, np.nan)
        result[::2] = mlp_confusion_mat

        result = result.transpose(1, 0, 2)  # Transpose to (C, H, W)
        result = result.reshape(result.shape[0], -1)
        conf_mat_df = pd.DataFrame(result)
        conf_mat_df = np.trunc(conf_mat_df * 1000) / 1000
        conf_mat_df = conf_mat_df.astype('Float64').astype(str)
        conf_mat_df.replace("<NA>", "    ", inplace=True)

        write_N_print(f"\nConfusion matrix for each layer:\n", log_file)
        write_N_print(conf_mat_df.iloc[:, :22].to_string(index=False, header=False) + "\n", log_file)
        write_N_print(conf_mat_df.iloc[:, 24:].to_string(index=False, header=False), log_file)

        accuracy = total_correct / data_size
        write_N_print(f"Overall accuracy: {accuracy:.2%}\n", log_file)
        return accuracy, mlp_accuracy
    
    return total_correct / data_size



def write_N_print(string, log_file):
    print(string)
    log_file.write(string + "\n")
    log_file.flush()
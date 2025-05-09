import os, sys
import time
from transformers.models.vit.modeling_vit import ViTConfig
from torch import nn
import torch
from datetime import datetime
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torchvision
from tqdm import tqdm
import torch.optim as optim
from ptflops import get_model_complexity_info
import importlib
import random

module_name = "model_utils"
module = importlib.import_module(module_name)
ModifiedViTModel = module.ModifiedViTModel


seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print("CUDA device:", os.getenv("CUDA_VISIBLE_DEVICES"), ' '.join(sys.argv))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor
import kagglehub

# Download and get path to Tiny ImageNet dataset
path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
data_root = os.path.join(path, "tiny-imagenet-200")  # ensure full dataset path

print("Path to dataset files:", data_root)

# Load the ViT processor
processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

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
            indices = torch.arange(min(size, len(self.dataset))).tolist()
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

# Initialize datasets
train_dataset = TinyImageNetDataset(
    root=data_root,
    train=True,
    size=None,
    processor=processor
)

test_dataset = TinyImageNetDataset(
    root=data_root,
    train=False,
    size=None,
    processor=processor
)


# Custom dataset class for CIFAR-100 that integrates with AutoImageProcessor for processing
class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, size=None, processor=None):
        self.dataset = torchvision.datasets.ImageFolder(root=root)           # Load CIFAR-100 dataset from torchvision
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

def write_N_print(string):
    print(string)
    log_file.write(string + "\n")
    log_file.flush()


# Training function to train the model over multiple epochs
def train(model, train_loader, test_loader, save_path, num_epochs=10, loss_type='both', lr=1e-4):
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
        train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

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

        val_accuracy, mlp_val_accuracy = test(model, test_loader, full_testing=True)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            if save_path:
                torch.save(best_model, model_save_path)

        # write_N_print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
        write_N_print(f"Test accuracy after {epoch + 1} epochs: {val_accuracy:.2%}\n")

    # accuracy, mlp_accuracy = test(model, temp_train_loader, full_testing=True)
    write_N_print(f"Best accuracy: {best_val_accuracy * 100}%\n")


def get_complexity(model,test_loader, processor, device):
    total_flops = 0
    total_time = 0.0
    model.eval()

    with torch.no_grad():
        for inp, tar in tqdm(train_loader, desc="Evaluating"):
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
def test(model, dataloader, full_testing=False):
    model.eval()
    total_correct = 0
    if full_testing:
        total_correct_mlp = torch.zeros(len(model.encoder.layer), 2, 2)

    data_size = len(dataloader.dataset)
    with torch.no_grad():             # Disable gradient computation for efficiency
        dataloader = tqdm(dataloader, desc=f"Testing", leave=False)
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
        write_N_print(f"Skip ratio: {each_layer_skip}\nSkip %: {each_layer_skip.mean():.2%}")
        mlp_accuracy = (torch.sum(total_correct_mlp[:, 1, 1]) + torch.sum(total_correct_mlp[:, 0, 0])) / (torch.sum(total_correct_mlp) + 1e-16)
        mlp_confusion_mat = (total_correct_mlp / (total_correct_mlp.sum(dim=[-2, -1], keepdim=True) + 1e-16)).tolist()
        mlp_accuracy_arr = (total_correct_mlp[:, 1, 1] + total_correct_mlp[:, 0, 0]) / total_correct_mlp.sum(dim=[1, 2])
        formatted_lst = [[[round(num, 4) for num in row] for row in col]  for col in mlp_confusion_mat]

        for i in range(len(mlp_accuracy_arr)):
            write_N_print(f"Correct prediction ratio for layer {i}:\n{formatted_lst[i]}\t{mlp_accuracy_arr[i]:.2%}\n")
        write_N_print(f"Overall accuracy of MLP: {mlp_accuracy:.2%}")

        return total_correct / data_size, mlp_accuracy
    
    return total_correct / data_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the model')
    # parser.add_argument('--num_epochs', type=int, nargs='+', help='Number of epochs to train the model for', required=True)
    # parser.add_argument('--loss_type', type=str, nargs='+', help='Type of loss to use for training', required=True)
    # parser.add_argument('--lr', type=float, nargs='+', help='Learning rate for training', required=True)
    # parser.add_argument('train_size', type=int, help='Size of training dataset', required=True)
    # parser.add_argument('test_size', type=int, help='Size of testing dataset', required=True)
    parser.add_argument('--model_desc', '-d', type=str, help='Desc of model', required=True)

    args = parser.parse_args()

    # Define the training parameters
    pretrained_model_path = "/home/ai21btech11012/ViT-pruning/himanshu/models/2025-02-27_19-58-51_fixing_loss_weight-1.5_mlp_model_utils_loss-cosine^alternate_lr-0.001^0.0001_st-0.9_mt-0.5_bs-64_trs-10000_tes-1000_nw-16.pth"
    pretrained_model_path = None
    train_type = "both"  # mlp or vit or both
    loss_type = ["cosine", "classification"]
    num_epochs = [10, 10]
    lr = [1e-3, 1e-5]
    train_batch_size = 64
    test_batch_size = 128
    train_size = 10000
    test_size = 1000
    sim_threshold = 0.9
    mlp_threshold = 0.5
    avg_threshold = 0
    num_workers = 16
    data_path = './data'
    pretrained_cifar_model_name = "google/vit-base-patch16-224"
    csv_path = f"stats.csv"

    combined_lr = "^".join(map(str, lr))
    combined_loss = "^".join(loss_type)

    save_path = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.model_desc}_{train_type}_{module_name}_loss-{combined_loss}_lr-{combined_lr}_st-{sim_threshold}_mt-{mlp_threshold}_at-{avg_threshold}_bs-{train_batch_size}_trs-{train_size}_tes-{test_size}_nw-{num_workers}"

    # if (train_type == "vit") == bool(pretrained_model_path):
    #     if pretrained_model_path:
    #         save_path = f"{save_path}_prevmodel-{pretrained_model_path.split('/')[-1].split('.')[0]}"
    # else:
    #     print("Not correct configuration. Exiting...")
    #     exit()

    cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)   # Initialize the image processor for CIFAR-100 data pre-processing
    pretrained_cifar_model = AutoModelForImageClassification.from_pretrained(pretrained_cifar_model_name)
    config = ViTConfig.from_pretrained(pretrained_cifar_model_name)
    for name, param in pretrained_cifar_model.named_parameters():
        print(f"{name}: {param.shape}")
    config.num_labels = len(config.id2label)
    modified_vit_model = ModifiedViTModel(config, sim_threshold, mlp_threshold, avg_threshold)

    # -----------   Copying Weights    ------------  #
    if not pretrained_model_path:
        new_state_dict = {}
        for key in pretrained_cifar_model.state_dict().keys():
            new_key = key.replace('vit.', '')
            new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

        modified_vit_model.load_state_dict(new_state_dict, strict=False)
    else:
        modified_vit_model.load_state_dict(torch.load(pretrained_model_path))

    # modified_vit_model = nn.DataParallel(modified_vit_model)
    modified_vit_model.to(device)
    # modified_vit_model = modified_vit_model.module


    # train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
    # test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)

    train_indices = random.sample(range(len(train_dataset)), 10000)
    train_subset = Subset(train_dataset, train_indices)

    # Randomly select 1,000 indices for testing
    test_indices = random.sample(range(len(test_dataset)), 1000)
    test_subset = Subset(test_dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True)

    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    log_file = open(f"logs/{save_path}.txt", "w")

    if train_type not in ['mlp', 'vit', 'both']:
        print("Train_type variable not correct. Exiting...")
        exit()
    
    # ----   Training and testing started   ----
    write_N_print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True)}')

    if train_type in ['mlp', 'both']:
        train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs[0], loss_type=loss_type[0], lr=lr[0], save_path=save_path)
    
    if train_type in ['vit', 'both']:
        train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs[1], loss_type=loss_type[1], lr=lr[1], save_path=None)

    # ---------------------------------------------------

    accuracy, mlp_accuracy = test(modified_vit_model, test_loader, full_testing=True)
    if os.path.exists(csv_path):
        stats_file = open(csv_path, "a")
    else:
        stats_file = open(csv_path, "w")
        stats_file.write("Model, Accuracy, MLP Accuracy\n")
    stats_file.write(f"{save_path}, {accuracy}, {mlp_accuracy}\n")
    stats_file.close()

    # # calculating flops
    input_size = (3, 224, 224)
    flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
    flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
    print(flops, flops_ori)

    log_file.close()


"""# pretrained model accuracy 89.85"""

















################### ------------------------------------------------------------------------------------------------------------------############
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)

# config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
# pretrained_vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
# pretrained_classification_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# modified_vit_model.load_state_dict(pretrained_vit_model.state_dict(), strict=False)
# modified_vit_model.classifier.weight.data = pretrained_classification_model.classifier.weight.data.clone()
# modified_vit_model.classifier.bias.data = pretrained_classification_model.classifier.bias.data.clone()

# from PIL import Image
# import os
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from transformers import ViTFeatureExtractor
# import torch
# import requests


# transform = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# images_list = []
# images_list2 = []
# for image_name in os.listdir('images'):
#     img = Image.open(os.path.join('images', image_name)).convert('RGB')
#     images_list2.append(img)
#     img_tensor = transform(img)
#     images_list.append(img_tensor)

# batch_tensor = torch.stack(images_list)

# # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# # inputs = feature_extractor(images=images_list2, return_tensors="pt", padding=True)


# with torch.no_grad():
#     # out1 = pretrained_vit_model(batch_tensor)['pooler_output']
#     logits = modified_vit_model(batch_tensor)


# predicted_class_indices = logits.argmax(dim=-1).tolist()

# url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# response = requests.get(url)
# class_names = response.json()

# for idx, class_index in enumerate(predicted_class_indices):
#     print(f"Predicted class index for image {idx + 1}: {class_index}")
#     print(class_names[class_index])

# plt.imshow(inputs['pixel_values'].squeeze(0).permute(1, 2, 0))
# print(inputs['pixel_values'].max(), inputs['pixel_values'].min())
# plt.show()
# test_dataset[1][0].show()
# plt.imshow(test_dataset[1][0])
# plt.show()
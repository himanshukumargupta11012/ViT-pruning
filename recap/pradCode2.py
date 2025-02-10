# -*- coding: utf-8 -*-

# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)


from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Optional
from transformers import ViTForImageClassification
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# class ModifiedViTLayer(ViTLayer):
#     def __init__(self, config, threshold=0.02, beta=0.5, similarity_threshold=0.5, mlp_threshold=0.5):
#         super().__init__(config)
#         self.hidden_size = config.hidden_size
        
#         # Define the MLP layers with a 768 intermediate dimension
#         layer_sizes = [self.hidden_size, 32, 768, 32, 1]

#         mlp_layers = []
#         for i in range(len(layer_sizes) - 1):
#             if i < len(layer_sizes) - 2:
#                 mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
#             else:
#                 mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])                

#         self.mlp_layer = nn.Sequential(*mlp_layers)
#         self.threshold = threshold
#         self.loss = 0
#         self.skip_ratio = 0

#         # Additional hyperparameters for loss calculation
#         self.beta = beta
#         self.similarity_threshold = similarity_threshold
#         self.mlp_threshold = mlp_threshold


#     def forward(self, hidden_states: torch.Tensor,
#                 head_mask: Optional[torch.Tensor] = None,
#                 output_attentions: bool = False, compute_cosine=True, output_mask=False):

#         # Forward pass through the first two MLP layers
#         layer1_output = self.mlp_layer[1](self.mlp_layer[0](hidden_states[:, 1:]))
#         middle_layer_output = self.mlp_layer[3](self.mlp_layer[2](layer1_output))
#         layer3_output = self.mlp_layer[5](self.mlp_layer[4](middle_layer_output+hidden_states[:, 1:]))
#         mlp_output = self.mlp_layer[7](self.mlp_layer[6](layer3_output))
        
#         boolean_mask = (mlp_output >= self.threshold).squeeze(-1)
#         cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)

#         boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)
#         output = hidden_states.clone()
#         batch_size = hidden_states.shape[0]

#         ones = torch.ones((middle_layer_output.size(0), 1, middle_layer_output.size(2)), device=middle_layer_output.device)
#         middle_layer_output_new = torch.cat((ones, middle_layer_output+hidden_states[:,1:]), dim=1)
# # torch.Size([196, 768])
#         # print(middle_layer_output_new)
#         for i in range(batch_size):
            
#             output[i][boolean_mask[i]] =super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)
            
#             if (~boolean_mask[i]).sum() > 0: 
#                 output[i][~boolean_mask[i]] = (middle_layer_output_new[i])[~boolean_mask[i]]

#         if self.training or compute_cosine:
#                 # Compute the "real" output using the original ViTLayer
#                 real_output = super().forward(hidden_states[:, 1:])[0]
                
#                 # Cosine similarity between the real output and the hidden states
#                 cos_similarity = (F.cosine_similarity(real_output, middle_layer_output+hidden_states[:,1:], dim=-1) + 1) / 2
            
#                 # cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
#                 # euclidean_dist = torch.sum((real_output - middle_layer_output) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
#                 euclidean_dist = torch.sum((real_output - (middle_layer_output+hidden_states[:,1:])) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
#                 dist_similarity = 1 / (1 + euclidean_dist)
                
#                 # Weighted similarity value
#                 alpha = 0.8
#                 similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 

#                 # Binary Cross-Entropy Loss
#                 bce_loss = nn.BCELoss()(mlp_output.squeeze(-1), (similarity_val < self.similarity_threshold).float())

#                 # MSE Loss: Compute the MSE loss between input (hidden_states[:, 1:]) and the middle layer (middle_layer_output)
#                 mse_loss = F.mse_loss(middle_layer_output+hidden_states[:,1:],real_output) 
#                 # Final loss as a weighted sum of BCE and MSE losses
#                 beta=0.8
#                 self.loss = beta * bce_loss + (1 -beta) * mse_loss
#                 # self.loss = self.beta * bce_loss 

#                 # MLP accuracy: Determining the number of accurate predictions based on the threshold
#                 self.mlp_accuracy_arr = ((self.similarity_threshold - similarity_val) * 
#                                         (mlp_output.squeeze(-1) - self.mlp_threshold) > 0)
            
#                 del real_output
#         else:
#             self.loss = 0
        
#         # torch.cuda.empty_cache()

#         # Skip ratio: Number of skipped positions in the mask
#         self.skip_ratio = torch.sum(~boolean_mask).item()

#         # Optionally return the boolean mask as well
#         if output_mask:
#             return (output, boolean_mask)
#         else:
#             return (output,)



class ModifiedViTLayer(ViTLayer):
    def __init__(self, config, threshold=0.02, beta=0.5, similarity_threshold=0.5, mlp_threshold=0.5):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        
        # Define the MLP layers with a 768 intermediate dimension
        layer_sizes = [self.hidden_size, 64, 1]
        layer_sizes2 = [self.hidden_size,128, 768]

        mlp_layers = []
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
            else:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])                

        self.mlp_layer = nn.Sequential(*mlp_layers)
        
        mlp_layers2 = []
        for i in range(len(layer_sizes2) - 1):
            if i < len(layer_sizes2) - 2:
                mlp_layers2.extend([nn.Linear(layer_sizes2[i], layer_sizes2[i + 1]), nn.ReLU()])
            else:
                mlp_layers2.extend([nn.Linear(layer_sizes2[i], layer_sizes2[i + 1]), nn.Sigmoid()])                

        self.mlp_layer2 = nn.Sequential(*mlp_layers2)
        


        self.threshold = threshold
        self.loss = 0
        self.skip_ratio = 0

        # Additional hyperparameters for loss calculation
        self.beta = beta
        self.similarity_threshold = similarity_threshold
        self.mlp_threshold = mlp_threshold


    def forward(self, hidden_states: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False, compute_cosine=True, output_mask=False):

        # # Forward pass through the first two MLP layers
        # layer1_output = self.mlp_layer[1](self.mlp_layer[0](hidden_states[:, 1:]))
        # middle_layer_output = self.mlp_layer[3](self.mlp_layer[2](layer1_output))
        # layer3_output = self.mlp_layer[5](self.mlp_layer[4](middle_layer_output+hidden_states[:, 1:]))
        # mlp_output = self.mlp_layer[7](self.mlp_layer[6](layer3_output))
 
        mlp_output = self.mlp_layer(hidden_states[:, 1:])
        mlp_update = self.mlp_layer2(hidden_states[:, 1:])

        # newCos = F.cosine_similarity(mlp_update+hidden_states[:, 1:], hidden_states[:, 1:], dim=-1)
        # thresholdVar = 0.7
        # print(newCos)
        # mask = newCos <= (thresholdVar)
        # topK=40
        # _, topK_indices = torch.topk(newCos, topK, dim=1, largest=True, sorted=False)
        # mask = torch.ones_like(newCos, dtype=torch.bool)
        # for i in range(newCos.size(0)):  # Loop over each row (16 rows in your case)
        #     mask[i, topK_indices[i]] = False  # Set the top K values to True

    
        ones = torch.ones((mlp_update.size(0), 1, mlp_update.size(2)), device=mlp_update.device)
        middle_layer_output_new = torch.cat((ones, mlp_update+hidden_states[:,1:]), dim=1)
        boolean_mask = (mlp_output >= self.threshold).squeeze(-1)
        # boolean_mask=mask
        cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
        # print(boolean_mask.shape)
        boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)
        output = hidden_states.clone()
        batch_size = hidden_states.shape[0]

        for i in range(batch_size):
            output[i][boolean_mask[i]] =super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)
            if (~boolean_mask[i]).sum() > 0: 
                output[i][~boolean_mask[i]] = (middle_layer_output_new[i])[~boolean_mask[i]]

        if self.training or compute_cosine:
                # Compute the "real" output using the original ViTLayer
                real_output = super().forward(hidden_states[:, 1:])[0]
                
                # Cosine similarity between the real output and the hidden states
                cos_similarity = (F.cosine_similarity(real_output, mlp_update+hidden_states[:,1:], dim=-1) + 1) / 2
            #  mlp_update+hidden_states[:,1:]
                # cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
                euclidean_dist = torch.sum((real_output - (mlp_update+hidden_states[:,1:])) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
                # euclidean_dist = torch.sum((real_output - (mlp_update+hidden_states[:,1:])) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
                dist_similarity = 1 / (1 + euclidean_dist)
                alpha = 1
                similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 
                bce_loss = nn.BCELoss()(mlp_output.squeeze(-1), (similarity_val < self.similarity_threshold).float())
                mse_loss = F.mse_loss(mlp_update+hidden_states[:,1:],real_output) 
                # Final loss as a weighted sum of BCE and MSE losses
                beta=0.5
                self.loss = beta * bce_loss + (1 -beta) * mse_loss
                # self.loss = self.beta * bce_loss 

                # MLP accuracy: Determining the number of accurate predictions based on the threshold
                self.mlp_accuracy_arr = ((self.similarity_threshold - similarity_val) * 
                                        (mlp_output.squeeze(-1) - self.mlp_threshold) > 0)
            
                del real_output
        else:
            self.loss = 0
        
        # torch.cuda.empty_cache()

        # Skip ratio: Number of skipped positions in the mask
        self.skip_ratio = torch.sum(~boolean_mask).item()

        # Optionally return the boolean mask as well
        if output_mask:
            return (output, boolean_mask)
        else:
            return (output,)

class ModifiedViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig, threshold=0.02):
        super().__init__(config)
        self.layer = nn.ModuleList([ModifiedViTLayer(config, threshold) for _ in range(config.num_hidden_layers)])

class ModifiedViTModel(ViTModel):
    def __init__(self, config: ViTConfig, threshold=0.02):
        super().__init__(config)
        self.encoder = ModifiedViTEncoder(config, threshold)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        outputs = super().forward(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = self.classifier(outputs['last_hidden_state'][:, 0])
        output = lambda: None
        setattr(output, 'logits', logits)

        return output



from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torchvision
from tqdm import tqdm
import os
import torch.optim as optim
from ptflops import get_model_complexity_info


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


# Function to evaluate the model on the test data
def test(model, dataloader, full_testing=False, progress=True):
    model.eval()
    total_correct = 0
    each_layer_skip = torch.zeros(len(model.encoder.layer))
    if full_testing:
        total_correct_mlp = torch.zeros(len(model.encoder.layer))
        total_mlp_count = torch.zeros(len(model.encoder.layer))
    data_size = len(dataloader.dataset)
    with torch.no_grad():             # Disable gradient computation for efficiency
        if progress:
            dataloader = tqdm(dataloader, desc=f"Testing", leave=False)
        for (inputs, labels) in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if full_testing:
                outputs = model(inputs)
            else:
                outputs = model.module(inputs)
            logits = outputs.logits

            predicted_class_indices = logits.argmax(dim=-1)    # Get the class index with the highest logit score

            # Track the skip ratios from each layer
            for i, vit_layer in enumerate(model.encoder.layer):
                each_layer_skip[i] += vit_layer.skip_ratio#.item()
                if full_testing:
                    total_correct_mlp[i] += (vit_layer.mlp_accuracy_arr).sum().item()
                    total_mlp_count[i] += vit_layer.mlp_accuracy_arr.numel()
            total_correct += torch.sum(predicted_class_indices == labels).item()

    if full_testing:
        print(f"correct prediction ratio: {total_correct_mlp / total_mlp_count} and {torch.sum(total_correct_mlp) / torch.sum(total_mlp_count)}")
    print(f"Skip ratio: {each_layer_skip / data_size} and {torch.sum(each_layer_skip) / data_size / len(model.encoder.layer)}")
    return total_correct / data_size


# Training function to train the model over multiple epochs
def train(model, train_loader, test_loader, num_epochs=10, loss_type='both', lr=1e-4, progress=True):
    model.train()
    criterion = nn.CrossEntropyLoss()
    cosine_loss_ratio = 1.5

    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
    temp_train_loader = train_loader
    for epoch in range(num_epochs):
            model.train()  # Re-enable training mode at the start of each epoch
            running_loss = 0.0
            # total_correct_mlp = torch.zeros(len(model.encoder.layer))
            # total_mlp_count = torch.zeros(len(model.encoder.layer))
            if progress:
                train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)                

                # Forward pass to get the classification logits, loss calculation
                logits = model(inputs).logits

                if loss_type in ['classification', 'both']:
                    classification_loss = criterion(logits, labels)

                # Accumulate cosine similarity losses from each DHSLayer
                if loss_type in ['cosine', 'both']:
                    cosine_loss = 0.0
                    for i, vit_layer in enumerate(model.encoder.layer):
                        # total_correct_mlp[i] += (vit_layer.mlp_accuracy_arr).sum().item()
                        # total_mlp_count[i] += vit_layer.mlp_accuracy_arr.numel()
                        cosine_loss += vit_layer.loss
                
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
                if progress:
                    if (batch_idx + 1) % 5 == 0:
                        train_loader.update(5)
                    train_loader.set_postfix({'batch_loss': total_loss.item()})

            print(f"")
            print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
            print(f"Test accuracy after {epoch + 1} epochs: {test(model, test_loader, full_testing=True, progress=progress)}")
            # print(f"Train accuracy after {epoch + 1} epochs: {test(model, temp_train_loader, full_testing=True, progress=progress)}")


# Define the training parameters
load_pretrained = False            # Flag to load a pre-trained model
pretrained_model_path = os.path.join('models', 'new_model.pth')
loss_type = 'cosine'
num_epochs = 10
lr = 1e-3
data_path = 'data'
batch_size = 16
train_size = 10000
test_size = 1000

threshold = 0.02
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"
print(device)

cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)   # Initialize the image processor for CIFAR-100 data pre-processing
pretrained_cifar_model = AutoModelForImageClassification.from_pretrained(pretrained_cifar_model_name)
config = ViTConfig.from_pretrained(pretrained_cifar_model_name)
config.num_labels = len(config.id2label)
modified_vit_model = ModifiedViTModel(config, threshold)

# -----------   Copying Weights    ------------  #
if not load_pretrained:
    new_state_dict = {}
    for key in pretrained_cifar_model.state_dict().keys():
        new_key = key.replace('vit.', '')
        new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

    modified_vit_model.load_state_dict(new_state_dict, strict=False)
else:
    modified_vit_model.load_state_dict(torch.load(pretrained_model_path))

modified_vit_model = nn.DataParallel(modified_vit_model)
modified_vit_model.to(device)
modified_vit_model = modified_vit_model.module


# Making grad true or false depending on training type
for param in modified_vit_model.parameters():
    param.requires_grad = False

if loss_type in ['cosine', 'both']:
    for layer in modified_vit_model.encoder.layer:
        for param in layer.mlp_layer.parameters():
            param.requires_grad = True

if loss_type in ['classification', 'both']:
    for param in modified_vit_model.classifier.parameters():
        param.requires_grad = True


train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True, progress=True)}')

train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs, loss_type=loss_type, lr=lr, progress=True)
torch.save(modified_vit_model.state_dict(), os.path.join('models', 'prad_model.pth'))

# calculating flops
input_size = (3, 224, 224)
flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
print(flops, flops_ori)

"""# pretrained model accuracy 89.85"""

















################### ------------------------------------------------------------------------------------------------------------------############


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
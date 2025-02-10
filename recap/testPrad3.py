# -*- coding: utf-8 -*-

# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)


from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torchvision
from tqdm import tqdm
import os
import torch.optim as optim
from ptflops import get_model_complexity_info

from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Optional
from transformers import ViTForImageClassification
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

class CNNNetwork(nn.Module):
    def __init__(self, hidden_size=768, mlp_size=16, conv_out_channels=64):
        super(CNNNetwork, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),  # 768 -> 16
            nn.ReLU(),
            nn.Linear(mlp_size, mlp_size),  # You can stack more layers if needed
            nn.ReLU()
        )
        
        self.conv1 = nn.Conv2d(1, conv_out_channels, kernel_size=3, padding=1)  # 56x56 -> 56x56
        self.conv2 = nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1)  # 56x56 -> 
        
        self.fc_out = nn.Linear(conv_out_channels * 56 * 56, 196)  # Flattened to 196 dimensions
    
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        reduced_states = self.mlp(hidden_states)  # (64, 196, 16)
        reduced_states = reduced_states.view(batch_size, seq_len, 16)  # (64, 196, 16)
        reshaped_states = reduced_states.view(batch_size, 14, 4, 16)  # Reshape (64, 14, 4, 16)
        reshaped_states = reshaped_states.permute(0, 3, 1, 2)  # (64, 16, 14, 4) for Conv layers
        conv_out = F.relu(self.conv1(reshaped_states))  # (64, conv_out_channels, 14, 4)
        conv_out = F.relu(self.conv2(conv_out))  # (64, conv_out_channels, 14, 4)
        conv_out = conv_out.view(batch_size, -1)  # Flatten to (64, conv_out_channels * 56 * 56)
        output = self.fc_out(conv_out)  # (64, 196)
        
        return output


class ModifiedViTLayer(ViTLayer):
    def __init__(self, config, threshold=0.05):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.no_patchs=196
        self.cnnsize=16
        self.patch_size = int(self.cnnsize ** 0.5)
        # Calculate the grid size (e.g., 14x14 if no_patchs = 196)
        self.grid_size = int(self.no_patchs ** 0.5)  

        self.ScoreNetMLP=nn.Sequential(
            nn.Linear(self.hidden_size, self.cnnsize),  # 768 -> 16
            nn.ReLU(),
            nn.Linear(self.cnnsize, self.cnnsize),  # Further transformation to 16 dimensions
            nn.ReLU()
        )
        self.ScoreNetReshape = nn.Sequential(
            nn.Unflatten(1,(self.grid_size*self.patch_size, self.grid_size*self.patch_size))
        )
        self.ScoreNetconv1= nn.Conv2d(in_channels=1,out_channels=32,kernel_size=4,stride=2,padding=2)
        self.relu=nn.ReLU()
        self.ScoreNetPool1=nn.MaxPool2d(kernel_size=2, stride=2) 
        self.ScoreNetconv2= nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=2, padding=2)  
        self.ScoreNetPool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat=nn.Flatten()
        self.ScoreNetMLP2=nn.Linear(128,196)
        self.sigmoid=nn.Sigmoid()

        
        self.hidden_size = config.hidden_size
        # self.threshold = 0.95
        # self.mlpthreshold=0.5
        self.similarity_threshold = 0.9
        self.mlp_threshold = 0.5
        
        self.loss = 0
        self.skip_ratio = 0

    def forward(self, hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False, compute_cosine = True):


        input1 = self.ScoreNetMLP(hidden_states[:,1:]) 
        input1=input1.view(hidden_states[:,1:].shape[0], -1)
        input2=self.ScoreNetReshape(input1)
        input2=input2.unsqueeze(1)
        x = self.ScoreNetconv1(input2)
        x = self.relu(x)
        x = self.ScoreNetPool1(x)
        x = self.ScoreNetconv2(x)
        x = self.relu(x)
        x = self.ScoreNetPool2(x)
        ScoreNetOutput=self.sigmoid(self.ScoreNetMLP2(self.flat(x)))


        mlp_output=ScoreNetOutput.unsqueeze(-1)
        # print(mlp_output)
        


        boolean_mask = (mlp_output >= self.mlp_threshold).squeeze(-1)
        cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
        boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

        output = hidden_states.clone()
        batch_size = hidden_states.shape[0]


        for i in range(batch_size):
            output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)


        if self.training or compute_cosine:
            # print(hidden_states[:, 1:].shape)
            

            real_output = super().forward(hidden_states[:, 1:])[0]
            
            cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
            euclidean_dist = torch.sum((real_output - hidden_states[:, 1:]) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
            dist_similarity = 1 / (1 + euclidean_dist)
            alpha = 0.5
            similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 
            self.loss = nn.BCELoss()(mlp_output.squeeze(-1), (similarity_val < self.similarity_threshold).float())
            self.mlp_accuracy_arr = ((self.similarity_threshold - similarity_val) * (mlp_output.squeeze(-1) - self.mlp_threshold) > 0)
        
            # similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 
            # target_label=(similarity_val >self.threshold)
            # self.loss = F.kl_div(mlp_output.squeeze(-1),target_label.float(), reduction='batchmean')

            # self.mlp_accuracy_arr = ( (1 - cos_similarity - (1-self.threshold)) * (mlp_output.squeeze(-1) - (1-self.threshold)) > 0)
        else:
            self.loss = 0
            
        self.skip_ratio = torch.sum(~boolean_mask)
        return (output, )

class ModifiedViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig, threshold=0.05):
        super().__init__(config)
        self.layer = nn.ModuleList([ModifiedViTLayer(config, threshold) for _ in range(config.num_hidden_layers)])

class ModifiedViTModel(ViTModel):
    def __init__(self, config: ViTConfig, threshold=0.05):
        super().__init__(config)
        self.encoder = ModifiedViTEncoder(config, threshold)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        outputs = super().forward(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = self.classifier(outputs['last_hidden_state'][:, 0])
        output = lambda: None
        setattr(output, 'logits', logits)

        return output

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
                each_layer_skip[i] += vit_layer.skip_ratio.item()
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
                # if loss_type == 'classification':
                #     total_loss = classification_loss
                # elif loss_type == 'cosine':
                #     total_loss = cosine_loss
                # elif loss_type == 'both':
                #     total_loss = classification_loss + cosine_loss_ratio * cosine_loss
                total_loss = cosine_loss
                
                optimizer.zero_grad()
                # print('loss now',total_loss)
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
lr = 5*1e-3
data_path = 'data'
batch_size = 64
train_size = 10000
test_size = 1000

threshold = 0.95
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

for layer in modified_vit_model.encoder.layer:
    # print(layer.parameters())
    for param in layer.ScoreNetMLP.parameters():
        param.requires_grad = True
    for param in layer.ScoreNetconv1.parameters():
        param.requires_grad = True
    for param in layer.ScoreNetconv2.parameters():
        param.requires_grad = True
    for param in layer.ScoreNetMLP2.parameters():
        param.requires_grad = True

train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True, progress=True)}')

train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs, loss_type=loss_type, lr=lr, progress=True)
torch.save(modified_vit_model.state_dict(), os.path.join('models', 'new_model.pth'))

# calculating flops
input_size = (3, 224, 224)
flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
print(flops, flops_ori)

"""# pretrained model accuracy 89.85"""
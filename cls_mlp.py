from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Optional
from transformers import ViTForImageClassification
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
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
        return inputs['pixel_values'].squeeze(0), label  
    

class ModifiedViTLayer(ViTLayer):
    def __init__(self, config):
        super().__init__(config)
        
        self.hidden_size = config.hidden_size
        # print('####')
        # print(self.hidden_size)
        # print('----')
        layer_sizes = [self.hidden_size*2, 64, 1]

        mlp_layers = []
        for i in range(len(layer_sizes) - 1):
            if i < len(mlp_layers) - 2:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
            else:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])                

        self.mlp_layer = nn.Sequential(*mlp_layers)
        self.threshold = 0.05
        self.loss = 0
        self.skip_ratio = 0


    def forward(self, hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False, compute_cosine = True):
        # print('####')
        # print(self.hidden_size)
        # print('----')

        # Step 1: Extract hidden_states[:, 1:] which is of shape (batch_size, 196, dim)
        tokens = hidden_states[:, 1:]  # Shape: (batch_size, 196, dim)

        # Step 2: Extract hidden_states[:, 0], which is of shape (batch_size, dim)
        cls_token = hidden_states[:, 0].unsqueeze(1)  # Shape: (batch_size, 1, dim)

        # Step 3: Concatenate cls_token with each of the 196 tokens in tokens
        # We need to concatenate along the feature dimension (dim), so we'll do this:
        concatenated = torch.cat((tokens, cls_token.expand(-1, 196, -1)), dim=-1)

        # The resulting shape will be (batch_size, 196, dim*2)
        # print(concatenated.shape)  # Should output: torch.Size([batch_size, 196, dim*2])
        mlp_output = self.mlp_layer(concatenated)
        
        # mlp_output = self.mlp_layer(torch.concat((hidden_states[:, 1:],hidden_states[:,0:1]),dim=1))
        boolean_mask = (mlp_output >= self.threshold).squeeze(-1)
        cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
        boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

        output = hidden_states.clone()
        batch_size = hidden_states.shape[0]
        for i in range(batch_size):
            output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)

        if self.training or compute_cosine:
            real_output = super().forward(hidden_states[:, 1:])[0]
            cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
            
            self.loss = nn.MSELoss()(cos_similarity, 1 - mlp_output.squeeze(-1))
            self.mlp_accuracy_arr = ((1 - cos_similarity - self.threshold) * (mlp_output.squeeze(-1) - self.threshold) > 0)
        else:
            self.loss = 0
            
        self.skip_ratio = torch.sum(~boolean_mask)
        return (output, )


class ModifiedViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([ModifiedViTLayer(config) for _ in range(config.num_hidden_layers)])


class ModifiedViTModel(ViTModel):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.encoder = ModifiedViTEncoder(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        outputs = super().forward(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = self.classifier(outputs['last_hidden_state'][:, 0])
        output = lambda: None
        setattr(output, 'logits', logits)

        return output


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
config = ViTConfig.from_pretrained("Ahmed9275/Vit-Cifar100")
config.num_labels = len(config.id2label)
modified_vit_model = ModifiedViTModel(config).to(device)
cifar_processor = AutoImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100")   # Initialize the image processor for CIFAR-100 data pre-processing
pretrained_cifar_model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")


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
                outputs = model(inputs)
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
        print(f"correct prediction ratio: {total_correct_mlp / total_mlp_count}")
    print(f"Skip ratio: {each_layer_skip / data_size}")
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
loss_type = 'cosine'
num_epochs = 5
lr = 2e-2
data_path = 'data'
batch_size = 64
train_size = 10000
test_size = 1000
pretrained_model_path = os.path.join('models', 'mlp_class_10000.pth')

# -----------   Copying Weights    ------------  #
if not load_pretrained:
    new_state_dict = {}
    for key in pretrained_cifar_model.state_dict().keys():
        new_key = key.replace('vit.', '')
        new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

    modified_vit_model.load_state_dict(new_state_dict, strict=False)
else:
    modified_vit_model.load_state_dict(torch.load(pretrained_model_path))

modified_vit_model.to(device)


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
torch.save(modified_vit_model.state_dict(), os.path.join('models', 'new_model.pth'))

# calculating flops
input_size = (3, 224, 224)
flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
print(flops, flops_ori)
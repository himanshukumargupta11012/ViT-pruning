# -*- coding: utf-8 -*-

# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)


from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPooling
)


class ModifiedViTLayer(ViTLayer):
    def __init__(self, config, threshold=0.05, mlp_needed=True):
        super().__init__(config)
        self.mlp_needed = mlp_needed

        self.loss = 0
        self.skip_ratio = 0
        if not self.mlp_needed:
            return
        self.hidden_size = config.hidden_size
        layer_sizes = [self.hidden_size, 64, 1]

        mlp_layers = []
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
            else:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])                

        self.mlp_layer = nn.Sequential(*mlp_layers)
        self.similarity_threshold = threshold
        self.mlp_threshold = 0.5


    def forward(self, hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False, compute_cosine = True, output_mask=False):

        boolean_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool).to(hidden_states.device)
        if not self.mlp_needed:
            if output_mask:
                return (super().forward(hidden_states)[0], boolean_mask)
            else:
                return super().forward(hidden_states)
        mlp_output = self.mlp_layer(hidden_states[:, 1:])
        boolean_mask = (mlp_output >= self.mlp_threshold).squeeze(-1)
        cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
        boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

        output = hidden_states.clone()
        batch_size = hidden_states.shape[0]
        for i in range(batch_size):
            output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)

        if self.training or compute_cosine:
            real_output = super().forward(hidden_states[:, 1:])[0]
            cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
            # cos_similarity = torch.abs(F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1))
            euclidean_dist = torch.sum((real_output - hidden_states[:, 1:]) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
            dist_similarity = 1 / (1 + euclidean_dist)
            alpha = .5
            similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 
            self.loss = nn.BCELoss()(mlp_output.squeeze(-1), (similarity_val < self.similarity_threshold).float())
            self.mlp_accuracy_arr = ((self.similarity_threshold - similarity_val) * (mlp_output.squeeze(-1) - self.mlp_threshold) > 0)
        else:
            self.loss = 0
            
        self.skip_ratio = torch.sum(~boolean_mask).item()
        if output_mask:
            return (output, boolean_mask)
        else:
            return (output, )

class ModifiedViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig, threshold=0.05):
        super().__init__(config)
        mlp_needed_arr = torch.ones(config.num_hidden_layers, dtype=torch.bool)
        mlp_needed_arr[8] = False
        # indices_to_change = [6, 7, 8]
        # mlp_needed_arr[indices_to_change] = True

        self.layer = nn.ModuleList([ModifiedViTLayer(config, threshold, mlp_needed_arr[_]) for _ in range(config.num_hidden_layers)])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        output_mask: bool = False
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_boolean_mask = () if output_mask else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, output_mask=output_mask)

            hidden_states = layer_outputs[0]


            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_mask:
                all_boolean_mask = all_boolean_mask + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        ), all_boolean_mask
    
class ModifiedViTModel(ViTModel):
    def __init__(self, config: ViTConfig, threshold=0.05):
        super().__init__(config)
        self.encoder = ModifiedViTEncoder(config, threshold)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_mask: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs, boolean_masks = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_mask=output_mask
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        outputs = BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        logits = self.classifier(outputs['last_hidden_state'][:, 0])
        output = lambda: None
        setattr(output, 'logits', logits)
        setattr(output, 'boolean_masks', boolean_masks)
        
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
                outputs = model(inputs)
            logits = outputs.logits

            predicted_class_indices = logits.argmax(dim=-1)    # Get the class index with the highest logit score

            # Track the skip ratios from each layer
            for i, vit_layer in enumerate(model.encoder.layer):
                each_layer_skip[i] += vit_layer.skip_ratio
                if full_testing and hasattr(vit_layer, 'mlp_accuracy_arr'):
                    total_correct_mlp[i] += (vit_layer.mlp_accuracy_arr).sum().item()
                    total_mlp_count[i] += vit_layer.mlp_accuracy_arr.numel()
            total_correct += torch.sum(predicted_class_indices == labels).item()

    if full_testing:
        print(f"correct prediction ratio: {total_correct_mlp / (total_mlp_count + 1e-16)} and {torch.sum(total_correct_mlp) / (torch.sum(total_mlp_count) + 1e-16)}")
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
load_pretrained = True            # Flag to load a pre-trained model
pretrained_model_path = os.path.join('models', 'final_model.pth')
loss_type = 'cosine'
num_epochs = 5
lr = 1e-3
data_path = 'data'
batch_size = 64
train_size = 10000
test_size = 1000
threshold = 0.9
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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
        if not hasattr(layer, 'mlp_layer'):
            continue
        for param in layer.mlp_layer.parameters():
            param.requires_grad = True

if loss_type in ['classification', 'both']:
    for param in modified_vit_model.classifier.parameters():
        param.requires_grad = True


train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

# print(modified_vit_model)
# if __name__ == '__main__':
#     print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True, progress=True)}')

    # train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs, loss_type=loss_type, lr=lr,progress=True)
    # torch.save(modified_vit_model.state_dict(), os.path.join('models', 'new_model3.pth'))

"""# pretrained model accuracy 89.85"""


import numpy as np
import torchvision
def layerOutputs(inputs):

    # Generate the model output
    output = modified_vit_model(inputs.to(device), output_mask=True)
    patch_size = 16

    # Convert the input to grayscale and resize it

    to_pil = transforms.ToPILImage()
    pil_image = to_pil(inputs.squeeze(0).cpu())

    # Convert to grayscale and resize
    bw_image = np.array(pil_image.convert("L").resize((224, 224)))
    
    # bw_image = np.array(inputs.squeeze(0).cpu().convert("L").resize((224, 224)))
    print(bw_image.shape)
    # Reshape the image into patches
    patches = bw_image.reshape(14, patch_size, 14, patch_size).transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)

    images = []

    for i, mask in enumerate(output.boolean_masks):
        curr_mask = np.array(mask.to("cpu").squeeze(0)[1:])
        patch = patches.copy()

        # Apply the mask to the patches
        patch[~curr_mask] = 0
        
        # Reconstruct the image from the patches
        out = patch.reshape(14, 14, patch_size, patch_size).transpose(0, 2, 1, 3).reshape(224, 224)
        print(out)

        # Add the label to the image
        label = f"Layer-{i+1}" if i < 6 else f"Image-{i-5}"  # Adjust label based on index
        labeled_image = add_label_to_image(out, label)
        
        # Append the labeled image to the images list
        images.append((label, labeled_image))

    # Return the labeled images along with the model output
    return output, images



import streamlit as st
import random
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from torchvision import transforms
def pil_to_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor.
    This will handle images with different modes (e.g., RGBA, RGB, L).
    """
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    # Define a transform to convert PIL images to tensors
    transformp2t = transforms.Compose([
        transforms.ToTensor(),  # Converts the PIL image to a tensor and scales pixel values to [0, 1]
        transforms.Resize((224, 224))
    ])
    
    # Apply the transform to the PIL image
    tensor_image = transformp2t(pil_image)
    
    return tensor_image
from torchvision.transforms import ToPILImage
# Function to add label to an image or create a blank image with a label
def add_label_to_image(image, label, size=(224, 224)):
    if image is None:
        # Create a blank image if no image is uploaded
        img = Image.new("RGB", size, color="gray")
    else:
        if(torch.is_tensor(image)):
            print(image.shape)
            image=image.squeeze(0)
            print(image.shape)
            trans = ToPILImage()
            image = trans(image)
        img = image.copy()
        img = img.resize(size)

    draw = ImageDraw.Draw(img)
    draw.text((10, 10), label, fill="white")
    return img

# Streamlit UI
st.title("Image Processor with Random Outputs")

# Drag-and-drop file upload
uploaded_image = st.file_uploader(
    "Drag and drop an image here (PNG, JPG, JPEG)",
    type=[],
)

inputImage = None
if uploaded_image:
    try:
        # Load the uploaded image
        inputImage = Image.open(uploaded_image)
        st.image(inputImage, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading the image: {e}")

# Submit button
if st.button("Submit"):
    # Generate a random number
    # random_number = random.randint(1, 100)
    # print(inputImage)
    inputImage=pil_to_tensor(inputImage)
    # print(inputImage)
    inputImage=inputImage.to(device)
    inputImage=inputImage.unsqueeze(0)

    outputs = modified_vit_model(inputImage)
    
    st.subheader(f"Random Number: {outputs.logits.argmax(dim=-1).item()}")

    # Generate 12 labeled images
    st.subheader("Generated Images:")
    cols = st.columns(6)
    # images = []



    # for i in range(1, 13):
    #     label = f"Layer-{i}" if i <= 6 else f"Image-{i-6}"
    #     labeled_image = add_label_to_image(inputImage, label)
    #     images.append((label, labeled_image))

    # Display 12 images
    # Assuming `inputs` is your input image tensor
    output, images = layerOutputs(inputImage)

    # # Display the images using Streamlit
    # display_images(images)

    for idx, (label, img) in enumerate(images):
        cols[idx % 6].image(img, caption=label, use_column_width=True)

    # Generate 12 random numbers for bar chart
    random_numbers = np.random.randint(1, 101, 12)

    # Display bar chart
    st.subheader("Bar Chart of Random Numbers:")
    st.bar_chart(random_numbers)

# If no image is uploaded, show blank images when the button is clicked
elif not uploaded_image:
    st.info("No image uploaded. Blank images will be generated when submitted.")


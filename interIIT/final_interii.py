import streamlit as st
from PIL import Image
import pandas as pd
# Set the page configuration


from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPooling
)

from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torchvision
from tqdm import tqdm
import os
import torch.optim as optim
from ptflops import get_model_complexity_info

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTConfig
import streamlit as st



import numpy as np
import torchvision
import matplotlib.pyplot as plt

import streamlit as st
import random
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from torchvision.transforms import ToPILImage
from torchvision import transforms

st.set_page_config(
    page_title="Cosine Similarity in Vision Transformers",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set the theme to light mode using Streamlit configuration
st.markdown(
    "<style>body { background-color: white; color: black; }</style>",
    unsafe_allow_html=True
)

# Title of the application
st.title("Patch Pruning in Vision Transformers (ViTs)")
with st.expander("Ideation", expanded=True):
    st.markdown("""
    Vision Transformers (ViTs) excel in visual tasks by modeling global context via self-attention. 
    However, this leads to high computational and memory costs, especially when processing redundant image patches. 
    The quadratic growth of Multi-Head Self-Attention (MHSA) computations with the number of patches exacerbates these inefficiencies. 

    This work introduces a novel pruning approach for ViTs that:
    - Balances computation and accuracy
    - Enhances the model’s explainability by identifying and skipping redundant patches
    """)

with st.expander("Prototype Description", expanded=True):
    st.markdown("""
    Our model optimizes Vision Transformers by dynamically pruning redundant patches using cosine similarity. 

    Key Features:
    - A lightweight MLP predicts pruning scores to balance computation and accuracy.
    - During **training**, pruning labels are generated and refined.
    - At **inference**, patches are skipped or retained based on the pruning scores, ensuring adaptive and efficient performance.

    This approach ensures both model efficiency and explainability by focusing on only relevant patches, reducing unnecessary computation.
    """)

# Combined Observations for Cosine Similarity Distributions
with st.expander("Background and Motivation", expanded=False):
    st.markdown(
        "We observed that, the average cosine similarity between the patches withing a layer is first decreasing and then rising to large values such as 0.6, 0.7, Also, we can observe that, as we go deep into the layers, the patches at same location have very close similarity, approaching nearly 0.9. This redundancy can be resolved by selectively pruning similarly positioned patches across the layers."

    ) 
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/cosineSim.png", caption="Layer-wise Cosine Similarity Distribution", use_column_width=False, width=350)
    with col2:
        st.image("images/cosineSim2.png", caption="Cosine Similarity Between Adjacent Layers", use_column_width=False, width=300)


# Dataset and Experiment Setup
with st.expander("Experiment Setup", expanded=False):
    st.markdown(
        "### Dataset: CIFAR-100\n"
        "CIFAR-100 is a widely used benchmark dataset in machine learning, consisting of 60,000 32x32 color\n" 
        "images categorized into 100 classes. It is divided into 50,000 training images and 10,000 \n"
        "test images. The dataset provides a diverse set of images with 600 images per class, designed to evaluate classification performance.\n"
    )
    st.image("images/cifar100.png", caption="CIFAR100 Dataset", use_column_width=False, width=300)


with st.expander("Literature Review", expanded=False):
    st.markdown(
    """
    Previously Done Work That Helped Build Our Approach:

    - **<u>ViT with Patch Diversification</u>:** Deeper layers in Vision Transformers (ViTs) exhibit high cosine similarity among patch embeddings. The approach introduces novel loss functions to reduce cosine similarity and promote patch diversity.

    - **<u>Patch Slimming</u>:** Identifies effective patches in the final layer to guide pruning in preceding layers, evaluating their contributions to output features and discarding less impactful patches.

    - **<u>IARED² (Interpretability-Aware Redundancy Reduction)</u>:** A model-agnostic and task-agnostic approach that removes uncorrelated patches dynamically using a policy network. Achieves up to 1.4x speed-up with less than 0.7% accuracy loss for DeiT.

    - **<u>Anti-Oversmoothing in Deep Vision Transformers</u>:** Addresses attention collapse in deep ViTs, where patch embeddings become too similar, resulting in coarse representations and loss of finer features.

    - **<u>Skip Attention: Improving Vision Transformers by Paying Less Attention</u>:** Proposes a method to skip attention in less critical areas, improving computational efficiency without sacrificing accuracy.

    - **<u>No Token Left Behind: Efficient Vision Transformer via Dynamic Token Idling</u>:** Introduces dynamic token idling, allowing unselected tokens to be reintroduced in subsequent layers, preventing the permanent loss of crucial information.
    """, unsafe_allow_html=True
)


with st.expander("Our Approach", expanded=False):
    st.markdown(
    """
    Using MLP to predict patch importance and dynamically prune patches to improve efficiency and performance.
    ###  
    - **Dynamic Pruning:** At each layer, scaled cosine similarity is computed to identify redundant (unevolving) patches. A lightweight MLP predicts a pruning score for each patch, which is compared against a predefined threshold to make a binary decision:  
        - **Keep Patch (Label = 1):** If the pruning score exceeds the threshold, the patch is retained for further computation.  
        - **Skip Patch (Label = 0):** If the pruning score is below the threshold, the patch is bypassed for the current layer.  
    - **Reintroduction of Skipped Patches:** Unlike traditional pruning methods, skipped patches are not permanently dropped. They are reintroduced in subsequent layers for reevaluation to ensure no critical information is lost.  
    <p style="color:red;"><b>Novality:</b> Dynamically skipping and reusing the patches</p>
    
    """,unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/train.png", caption="Training Workflow", use_column_width=False, width=400)
    with col2:
        st.image("images/test.png", caption="Inference Workflow", use_column_width=False, width=400)


with st.expander("Modification", expanded=False):
    st.write("### Some of the novel modifications we made to the original model:")

    st.markdown("1. Modified Attention Mechanism: The bypassing of patches was based on the fact that they are not updating \n"
        "themselves, but unpruned patches may still need to attend to all patches, including the bypassed ones\n" "while updating themselves. To address this, we had to modify PyTorch's Attention function (Q, K, V)\n" "accordingly to properly ensure that the unpruned patches can fully leverage the context from all patches")
            
    st.markdown("2. Using different architectures for finding patch score: The MLP in our current architectures predicts the patch relevance, using just the patch. This could sometime mislead that, the need of current patch updating is dependednt on other pactes, making the MLP not much efficient to predict. We then approached this, by taking the complete 196 patches and their 768 dimension embedding, converting these embeddings to 32 dim vector, and processing using Convolution layers. This way, we are considering the updating of current patch, by considering other patches too, making in better predictions. Finally we return the 196 patch scores, and decide which patches to skip and which pacthes to pass.")

with st.expander("Results", expanded=False):

    # Data for the first table
    data1 = {
        "# epochs": [10, 10],
        "# Patches Pruned": [40, 68],
        "Threshold": [0.95, 0.9],
        "FLOPs decrease %": ["~ 20%", "~ 35%"],
        "Accuracy": ["85.2%", "77%"]
    }

    # Data for the second table
    data2 = {
        "# epochs": [10],
        "# Patches Pruned": [40],
        "Threshold": [0.95],
        "FLOPs decrease %": ["~ 16%"],
        "Accuracy": ["85.4%"]
    }

    # Convert data to DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Streamlit app
    st.write("Accuracy of VIT and Patch Pruning at Different Thresholds")

    # col1, col2 = st.columns(2)
    # with col1:
            
    #     st.write("Accuracy of VIT and Average patches pruned at different thresholds")
    #     st.table(df1)
    # with col2:
    #     st.write("Accuracy of VIT and Average patches pruned at threshold = 0.95 using modified attention")
    #     st.table(df2)


    # col1, col2 = st.columns(2)
    # with col1:
            
    st.write("Accuracy of VIT and Average patches pruned at different thresholds")
    st.table(df1)
    st.write("Accuracy of VIT and Average patches pruned at threshold = 0.95 using modified attention")
    st.table(df2)


with st.expander("Future Work", expanded=False):

    st.markdown("1. In our methodology, unlike many other pruning methods, we are not permanently dropping the patches, but we are skipping them. This reintroduction of patches leads us to further usage of our method, in Semantic segmentation. We are now intrested in using our methodology for tasks which are very focused on minor details, like semantic segmentation.  \n")
    # st.markdown("2. ")

    st.markdown("2. A further update to the model, which focuses on using LSTM for the patches, to see if we can get better results. \n")

with st.expander("Conclusion", expanded=False):
    st.markdown(
        "This project highlights the importance of patch stability analysis as a key factor for understanding the dynamics of Vision \n""Transformers and guiding pruning methodologies. The results underscore the potential of similarity based metrics in identifying\n" "redundant or less-informative or idle patches across layers. The study and work opens doors for improving efficiency and\n" "performance in dense prediction tasks."
    )


# -*- coding: utf-8 -*-

# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)


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

# st.set_page_config(layout="wide")

with st.expander("Model Prototype", expanded=False):
    st.title("PATCH PRUNING IN VISION TRANSFORMERS")
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

    # Initialization function to ensure certain steps run only once
    def initialize():
        global device, pretrained_cifar_model_name, cifar_processor, pretrained_cifar_model, config, modified_vit_model

        if not hasattr(initialize, "initialized"):  # Check if already initialized
            # Define the training parameters
            device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
            # print(device)

            pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"

            # Initialize components
            cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)
            pretrained_cifar_model = AutoModelForImageClassification.from_pretrained(pretrained_cifar_model_name)
            config = ViTConfig.from_pretrained(pretrained_cifar_model_name)
            config.num_labels = len(config.id2label)
            modified_vit_model = ModifiedViTModel(config, threshold)

            initialize.initialized = True  # Set the flag to prevent reinitialization

    # Call the initialization function
    initialize()

    # -----------   Copying Weights    ------------  #
    if not load_pretrained:
        new_state_dict = {}
        for key in pretrained_cifar_model.state_dict().keys():
            new_key = key.replace('vit.', '')
            new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

        modified_vit_model.load_state_dict(new_state_dict, strict=False)
    else:
        # Use a flag to ensure this runs only once
        if not hasattr(modified_vit_model, "weights_loaded"):
            modified_vit_model.load_state_dict(torch.load(pretrained_model_path))
            modified_vit_model.weights_loaded = True  # Set the flag to prevent reloading

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

    # Function to load datasets (caching enabled)
    @st.cache_data
    def load_datasets(data_path, train_size, test_size, _processor, batch_size):
        """Load and preprocess CIFAR datasets."""
        train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=_processor)
        test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=_processor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        return train_loader, test_loader

    # Load datasets
    train_loader, test_loader = load_datasets(
        data_path=data_path,
        train_size=train_size,
        test_size=test_size,
        _processor=cifar_processor,  # Use `_processor` here
        batch_size=batch_size
    )

    # # Define the training parameters

    # load_pretrained = True            # Flag to load a pre-trained model
    # pretrained_model_path = os.path.join('models', 'final_model.pth')
    # loss_type = 'cosine'
    # num_epochs = 5
    # lr = 1e-3
    # data_path = 'data'
    # batch_size = 64
    # train_size = 10000
    # test_size = 1000
    # threshold = 0.9
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"
    # print(device)

    # cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)   # Initialize the image processor for CIFAR-100 data pre-processing
    # pretrained_cifar_model = AutoModelForImageClassification.from_pretrained(pretrained_cifar_model_name)
    # config = ViTConfig.from_pretrained(pretrained_cifar_model_name)
    # config.num_labels = len(config.id2label)
    # modified_vit_model = ModifiedViTModel(config, threshold)



    # # -----------   Copying Weights    ------------  #
    # if not load_pretrained:
    #     new_state_dict = {}
    #     for key in pretrained_cifar_model.state_dict().keys():
    #         new_key = key.replace('vit.', '')
    #         new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

    #     modified_vit_model.load_state_dict(new_state_dict, strict=False)
    # else:
    #     modified_vit_model.load_state_dict(torch.load(pretrained_model_path))

    # modified_vit_model = nn.DataParallel(modified_vit_model)
    # modified_vit_model.to(device)
    # modified_vit_model = modified_vit_model.module


    # # Making grad true or false depending on training type
    # for param in modified_vit_model.parameters():
    #     param.requires_grad = False

    # if loss_type in ['cosine', 'both']:
    #     for layer in modified_vit_model.encoder.layer:
    #         if not hasattr(layer, 'mlp_layer'):
    #             continue
    #         for param in layer.mlp_layer.parameters():
    #             param.requires_grad = True

    # if loss_type in ['classification', 'both']:
    #     for param in modified_vit_model.classifier.parameters():
    #         param.requires_grad = True


    # train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
    # test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # print(modified_vit_model)
    # if __name__ == '__main__':
    #     print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True, progress=True)}')

        # train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs, loss_type=loss_type, lr=lr,progress=True)
        # torch.save(modified_vit_model.state_dict(), os.path.join('models', 'new_model3.pth'))

    # """# pretrained model accuracy 89.85"""

    def layerOutputs(inputImg):

        # data = torchvision.datasets.CIFAR100(root='data', train=False, download=True)[201][0]
        # inputs = cifar_processor(images=data, return_tensors="pt")['pixel_values']
        # input = test_dataset.__getitem__(0)[0]
        output = modified_vit_model(inputImg.to(device), output_mask=True)
        patch_size = 16

        to_pil = transforms.ToPILImage()
        pil_image = to_pil(inputImg.squeeze(0).cpu())

        # Convert to grayscale and resize
        bw_image = np.array(pil_image.convert("L").resize((224, 224)))

        patches = bw_image.reshape(14, patch_size, 14, patch_size).transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)
        # plt.figure(figsize=(12, 4))
        images=[]
        skipConnections = []
        for i, mask in enumerate(output.boolean_masks):
            curr_mask = np.array(mask.to("cpu").squeeze(0)[1:])
            patch = patches.copy()
            # print(curr_mask.shape, patch.shape)
            skipConnections.append(196-np.sum(curr_mask))
            
            patch[~curr_mask] = 0

            out = patch.reshape(14, 14, patch_size, patch_size).transpose(0, 2, 1, 3).reshape(224, 224)
            # plt.subplot(2, 6, i + 1)
            # plt.imshow(out, cmap='inferno')
            # plt.title(f"{i + 1}th layer")
            # plt.axis('off')
            images.append(out)
        # plt.suptitle("MLP mask at each layer")
        # plt.tight_layout()
        # plt.show()
        # print(len(images))
        # print(images[0].shape)
        return images,skipConnections

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

    # Function to add label to an image or create a blank image with a label
    def add_label_to_image(image, label, size=(224, 224)):
        if image is None:
            # Create a blank image if no image is uploaded
            img = Image.new("RGB", size, color="gray")
        else:
            if(torch.is_tensor(image)):
                # print(image.shape)
                image=image.squeeze(0)
                # print(image.shape)
                trans = ToPILImage()
                image = trans(image)
            img = image.copy()
            img = img.resize(size)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label, fill="white")
        return img

    def display_images_with_labels(images):
        """
        Display a list of images with corresponding labels in a 6-column grid layout.
        :param images: List of tuples, where each tuple is (label, image)
        """
        # Create 6 columns for displaying images
        cols = st.columns(4)
        
        for idx, (label, img) in enumerate(images):
            # Get the corresponding column for the image
            col_idx = idx % 4
            
            # Display the image in the corresponding column with the label
            cols[col_idx].image(img, caption=label, use_column_width=True)

    # Assuming you have `images` as a list of (label, image) tuples
    # Example:  # `layerImgs` is your list of 12 images




    # Streamlit UI


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
            st.image(inputImage, caption="Uploaded Image", use_column_width=False, width=300)

            # st.image(inputImage, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading the image: {e}")

    # Submit button
    # if uploaded_image:
    #     st.

    if st.button("Submit") and uploaded_image:

        # Generate a random number
        # random_number = random.randint(1, 100)
        # print(inputImage)
        inputImage=pil_to_tensor(inputImage)
        # print(inputImage)
        inputImage=inputImage.to(device)
        inputImage=inputImage.unsqueeze(0)
        layerImgs,skipConnections=layerOutputs(inputImage)
        outputs = modified_vit_model(inputImage)
        # Mapping of class indices (0-99) to CIFAR-100 class names
        class_map = {
            0: 'beaver', 1: 'dolphin', 2: 'otter', 3: 'seal', 4: 'whale',
            5: 'aquarium fish', 6: 'flatfish', 7: 'ray', 8: 'shark', 9: 'trout',
            10: 'orchids', 11: 'poppies', 12: 'roses', 13: 'sunflowers', 14: 'tulips',
            15: 'bottles', 16: 'bowls', 17: 'cans', 18: 'cups', 19: 'plates',
            20: 'apples', 21: 'mushrooms', 22: 'oranges', 23: 'pears', 24: 'sweet peppers',
            25: 'clock', 26: 'computer keyboard', 27: 'lamp', 28: 'telephone', 29: 'television',
            30: 'bed', 31: 'chair', 32: 'couch', 33: 'table', 34: 'wardrobe',
            35: 'bee', 36: 'beetle', 37: 'butterfly', 38: 'caterpillar', 39: 'cockroach',
            40: 'bear', 41: 'leopard', 42: 'lion', 43: 'tiger', 44: 'wolf',
            45: 'bridge', 46: 'castle', 47: 'house', 48: 'road', 49: 'skyscraper',
            50: 'cloud', 51: 'forest', 52: 'mountain', 53: 'plain', 54: 'sea',
            55: 'camel', 56: 'cattle', 57: 'chimpanzee', 58: 'elephant', 59: 'kangaroo',
            60: 'fox', 61: 'porcupine', 62: 'possum', 63: 'raccoon', 64: 'skunk',
            65: 'crab', 66: 'lobster', 67: 'snail', 68: 'spider', 69: 'worm',
            70: 'baby', 71: 'boy', 72: 'girl', 73: 'man', 74: 'woman',
            75: 'crocodile', 76: 'dinosaur', 77: 'lizard', 78: 'snake', 79: 'turtle',
            80: 'hamster', 81: 'mouse', 82: 'rabbit', 83: 'shrew', 84: 'squirrel',
            85: 'maple', 86: 'oak', 87: 'palm', 88: 'pine', 89: 'willow',
            90: 'bicycle', 91: 'bus', 92: 'motorcycle', 93: 'pickup truck', 94: 'train',
            95: 'lawn-mower', 96: 'rocket', 97: 'streetcar', 98: 'tank', 99: 'tractor'
        }

    # # Example usage: getting the class label for class index 35
    # class_index = 35
    # class_label = class_map[class_index]
    # print(f"The class label for index {class_index} is {class_label}.")

        st.subheader(f"Pruning at each layer: {class_map[outputs.logits.argmax(dim=-1).item()]}")

        # Generate 12 labeled images
        # st.subheader("Generated Images:")
        # cols = st.columns(6)
        # layerImgs -> list of 12 images of size 224x224

        images = [(f"Layer-{i+1}", image) for i, image in enumerate(layerImgs)]
        display_images_with_labels(images)


        # for i in range(1, 13):
        #     label = f"Layer-{i}" if i <= 6 else f"Image-{i-6}"
        #     labeled_image = add_label_to_image(layerImgs[i], label)
        #     images.append((label, labeled_image))




        # Display 12 images
        # for idx, (label, img) in enumerate(images):
        #     cols[idx % 6].image(img, caption=label, use_column_width=True)

        # Generate 12 random numbers for bar chart
        # print(skipConnections)

        # Display bar chart
        st.subheader("Each layer prunned patches :")
        # st.bar_chart(skipConnections)
        # Plot using Matplotlib# Create columns for layout
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the width ratios as needed

        # Use col2 (center column) to display the bar chart
        with col2:
            st.write("### Bar Chart Example")

            # Create a smaller bar chart
            fig, ax = plt.subplots(figsize=(6, 4))  # Control figure size here
            ax.bar(range(12), skipConnections, color='skyblue')

            # Customize the x-axis and labels
            ax.set_xticks(range(12))
            ax.set_xticklabels([f'Layer {i+1}' for i in range(12)])

            # Rotate the x-axis labels to avoid overlap
            plt.xticks(rotation=45)

            # Add legend with custom label
            ax.legend(["Patches Pruned"], loc="upper right")

            # Display the plot in the center column
            st.pyplot(fig)

            # ax.imshow(img, cmap='inferno')

    # If no image is uploaded, show blank images when the button is clicked
    else :
        st.warning("Upload Image")


from transformers.models.vit.modeling_vit import ViTModel, ViTAttention, ViTLayer, ViTEncoder, ViTConfig, ViTSelfAttention
from torch import nn
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import ViTForImageClassification,  AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from tqdm import tqdm
import torch.utils.data as data_utils
from ptflops import get_model_complexity_info


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Set



class ModifiedViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, patch_indices_to_keep):
        
        # print('----')
        # print(hidden_states.shape)

        # # print(patch_indices_to_keep.shape)
        # print('----')
        # Create Query, Key, Value vectors
        
        queries = self.query(hidden_states)  # (batch_size, num_patches, hidden_size)
        keys = self.key(hidden_states)       # (batch_size, num_patches, hidden_size)
        values = self.value(hidden_states)   # (batch_size, num_patches, hidden_size)

        # Reshape to (batch_size, num_heads, num_patches, head_size)
        queries = queries.view(hidden_states.size(0), -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        keys = keys.view(hidden_states.size(0), -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        values = values.view(hidden_states.size(0), -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Prune the Query vectors based on provided indices
        # # print(patch_indices_to_keep)
        # if queries.size(2) != patch_indices_to_keep.size(0):
        #     # Adjust the patch_indices_to_keep mask to match the number of patches in `queries`
        #     patch_indices_to_keep = patch_indices_to_keep[:queries.size(2)]
                
        pruned_queries = self.prune_queries(queries, patch_indices_to_keep)

        # Calculate the attention scores (Q @ K^T)

        # print(f" Q:{queries.shape}")
        # print(f" K:{keys.shape}")
        # print(f" V:{values.shape}")
        # print(f" Q':{pruned_queries.shape}")
        queries=pruned_queries
        if(len(queries.shape)!=4):
            queries=queries.squeeze()
            
        # Calculate attention scores and apply softmax
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of values
        context = torch.matmul(attention_probs, values)

        # Concatenate heads and reshape to [batch, seq_len, hidden_size]
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, num_heads, head_size]
        new_context_shape = context.size()[:-2] + (self.all_head_size,)  # [batch, seq_len, hidden_size]
        context = context.view(new_context_shape)
        # print(context.shape)
        return context


    def prune_queries(self,queries,patch_indices_to_keep):
        # Select patches where patch_indices_to_keep is True
        # # print('apple')
        # # print(queries.shape)
        # # print(patch_indices_to_keep.shape)
        # # print("##")
        # # print(queries.shape)
        # # print('banana')
        
        pruned_queries = queries[:, :, patch_indices_to_keep, :]
        return pruned_queries


class ModifiedViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class ModifiedViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = ModifiedViTSelfAttention(config)
        # self.attention2 = ViTSelfAttention(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # Mimic output layer
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        patch_indices_to_keep=None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # Get attention outputs
        self_outputs = self.attention(hidden_states, patch_indices_to_keep)
        # self_outputs2 = self.attention2(hidden_states, patch_indices_to_keep)
        
        # # # print(self_outputs.shape)
        # # # print((self_outputs2[0].shape))
        attention_output=self_outputs

        # Mimic behavior of self.output using inline operations
        # attention_output = self_outputs[0]  # Get the main output from self.attention
        attention_output = self.dense(attention_output)  # Linear transformation
        attention_output = self.dropout(attention_output)  # Apply dropout
        
        # Return the output in the expected format
        # # print(attention_output.shape)
        # # print(self_outputs.shape)
        outputs = attention_output + self_outputs  # add attentions if we output them
        return (outputs,)

class ModifiedViTLayer(nn.Module):
    def __init__(self, config: ViTConfig) -> None:  
        super().__init__()
        self.attention = ModifiedViTAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        patch_indices_to_keep=None
    ) -> torch.Tensor:
        # # print(patch_indices_to_keep)
        attention_output = self.attention(hidden_states, head_mask, output_attentions, patch_indices_to_keep=patch_indices_to_keep)
        return attention_output

class DHSLayer(ModifiedViTLayer):
    def __init__(self, config, use_random_pruning=False, pruning_counts=None, layer_index=0):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.mlp_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        ## use_random_pruning argument (default False)
        self.use_random_pruning = use_random_pruning
        self.pruning_counts = pruning_counts if pruning_counts is not None else [0] * 12  # Default to no pruning
        self.layer_index = layer_index  # Layer index to access the correct count
        
        self.loss = 0
        self.skip_ratio = 0

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        
        keep_cls = True
        threshold = 0.05        # The MLP output threshold above which a token is retained for processing
        batch_size = hidden_states.shape[0]
        num_patches = hidden_states.shape[1] - 1  # Exclude CLS token
        
        if self.use_random_pruning:
            # Get the number of patches to drop for this specific layer from the pruning_counts array
            num_to_drop = self.pruning_counts[self.layer_index]

            # Ensure the number to drop is within the range of available patches
            num_to_drop = min(num_to_drop, num_patches)

            # Random boolean mask for dropping
            boolean_mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
            for i in range(batch_size):
                drop_indices = torch.randperm(num_patches)[:num_to_drop]
                boolean_mask[i, drop_indices] = False

            # Add CLS token to the mask
            cls_col = torch.ones((batch_size, 1), dtype=torch.bool).to(boolean_mask.device)
            boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

        else:
            # Keep cls means no need to apply the threshold to the first token(i.e the CLS token)
            if keep_cls:
                mlp_output = self.mlp_layer(hidden_states[:, 1:])
                boolean_mask = (mlp_output >= threshold).squeeze(-1)
                cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
                boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)
            else:
                mlp_output = self.mlp_layer(hidden_states)
                boolean_mask = (mlp_output >= threshold).squeeze(-1)
        # print(f" grape: {mlp_output.shape}")
        # Applies the ViTLayer superclass forward pass only on tokens selected by boolean_mask
        output = hidden_states.clone()
        for i in range(batch_size):
            # # print("map is")
            # # print(boolean_mask[i])
            # # print("map end")
            # print(f"apple: {boolean_mask[i].shape}")
            # print('!!!!!!!!!!!')
            # print(hidden_states.shape)
            # print('!!!!!!!!!!!!!!!')
            # output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0),patch_indices_to_keep=boolean_mask[i])[0].squeeze(0)
        
            output[i][boolean_mask[i]] = super().forward(hidden_states[i].unsqueeze(0),patch_indices_to_keep=boolean_mask[i])[0].squeeze(0)
        

        # Calculate the loss and accuracy during training
        if self.training:
            if keep_cls:
                real_output = super().forward(hidden_states[:, 1:])[0]
                cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
            else:
                real_output = super().forward(hidden_states)[0]
                cos_similarity = (F.cosine_similarity(real_output, hidden_states, dim=-1) + 1) / 2

            self.loss = nn.MSELoss()(cos_similarity, 1 - mlp_output.squeeze(-1))
            self.mlp_accuracy = ((1 - cos_similarity - threshold) * (mlp_output.squeeze(-1) - threshold) > 0).sum().item() / cos_similarity.numel()
        
        # Skip Ratio (During Inference) : Counts the total number of skipped tokens across the batch, allowing for tracking efficiency improvements.
        else:
            self.loss = 0
            self.skip_ratio = torch.sum(~boolean_mask)
        # # print(boolean_mask)
        return (output, )



# class DHSLayer(ViTLayer):
#     def __init__(self, config, use_random_pruning=False, pruning_counts=None, layer_index=0):
#         super().__init__(config)
#         self.hidden_size = config.hidden_size
#         self.mlp_layer = nn.Sequential(
#             nn.Linear(self.hidden_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
#         ## use_random_pruning argument (default False)
#         self.use_random_pruning = use_random_pruning
#         self.pruning_counts = pruning_counts if pruning_counts is not None else [0] * 12  # Default to no pruning
#         self.layer_index = layer_index  # Layer index to access the correct count
        
#         self.loss = 0
#         self.skip_ratio = 0

#     def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        
#         keep_cls = True
#         threshold = 0.05        # The MLP output threshold above which a token is retained for processing
#         batch_size = hidden_states.shape[0]
#         num_patches = hidden_states.shape[1] - 1  # Exclude CLS token

#         if self.use_random_pruning:
#             # Get the number of patches to drop for this specific layer from the pruning_counts array
#             num_to_drop = self.pruning_counts[self.layer_index]

#             # Ensure the number to drop is within the range of available patches
#             num_to_drop = min(num_to_drop, num_patches)

#             # Random boolean mask for dropping
#             boolean_mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
#             for i in range(batch_size):
#                 drop_indices = torch.randperm(num_patches)[:num_to_drop]
#                 boolean_mask[i, drop_indices] = False

#             # Add CLS token to the mask
#             cls_col = torch.ones((batch_size, 1), dtype=torch.bool).to(boolean_mask.device)
#             boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

#         else:
#             # Keep cls means no need to apply the threshold to the first token(i.e the CLS token)
#             if keep_cls:
#                 mlp_output = self.mlp_layer(hidden_states[:, 1:])
#                 boolean_mask = (mlp_output >= threshold).squeeze(-1)
#                 cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
#                 boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)
#             else:
#                 mlp_output = self.mlp_layer(hidden_states)
#                 boolean_mask = (mlp_output >= threshold).squeeze(-1)

#         # Applies the ViTLayer superclass forward pass only on tokens selected by boolean_mask
#         output = hidden_states.clone()
#         for i in range(batch_size):
#             output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)
        

#         # Calculate the loss and accuracy during training
#         if self.training:
#             if keep_cls:
#                 real_output = super().forward(hidden_states[:, 1:])[0]
#                 cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
#             else:
#                 real_output = super().forward(hidden_states)[0]
#                 cos_similarity = (F.cosine_similarity(real_output, hidden_states, dim=-1) + 1) / 2

#             self.loss = nn.MSELoss()(cos_similarity, 1 - mlp_output.squeeze(-1))
#             self.mlp_accuracy = ((1 - cos_similarity - threshold) * (mlp_output.squeeze(-1) - threshold) > 0).sum().item() / cos_similarity.numel()
        
#         # Skip Ratio (During Inference) : Counts the total number of skipped tokens across the batch, allowing for tracking efficiency improvements.
#         else:
#             self.loss = 0
#             self.skip_ratio = torch.sum(~boolean_mask)
#         # # print(boolean_mask)
#         return (output, )

class ModifiedViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig):
        super().__init__(config)             # Initialize the parent ViTEncoder class with the given config
        
        # Replace the standard transformer layers with DHSLayer to perform conditional token processing
        self.layer = nn.ModuleList([DHSLayer(config) for _ in range(config.num_hidden_layers)])

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
# ViTLayer
# class PrunedAttention(nn.Module):
#     def __init__(self, config, pruning_ratio=0.5):
#         super().__init__()
#         self.config = config
#         self.pruning_ratio = pruning_ratio

#     def prune_queries(self, queries, patch_indices_to_keep):
#         # queries: (batch_size, num_heads, num_patches, hidden_size)
#         # patch_indices_to_keep: Tensor or list of shape (batch_size, num_heads, num_patches_to_keep)
#         # # print(f"patches to keep {patch_indices_to_keep}")
#         # Ensure patch_indices_to_keep is a tensor for easy indexing
#         if isinstance(patch_indices_to_keep, list):
#             patch_indices_to_keep = torch.tensor(patch_indices_to_keep, dtype=torch.long, device=queries.device)

#         # Gather the pruned queries based on the provided indices
#         pruned_queries = queries[torch.arange(queries.size(0)).unsqueeze(1), :, patch_indices_to_keep, :]

#         return pruned_queries

#     def forward(self, hidden_states, attention_mask=None, output_attentions=False, patch_indices_to_keep=None):
#         # hidden_states: (batch_size, num_heads, num_patches, hidden_size)

#         # Prune the Query vectors based on provided indices
#         queries = self.prune_queries(hidden_states, patch_indices_to_keep)

#         # The Key and Value vectors remain unchanged
#         keys = hidden_states  # (batch_size, num_heads, num_patches, hidden_size)
#         values = hidden_states  # (batch_size, num_heads, num_patches, hidden_size)

#         # Calculate the attention scores (Q @ K^T)
#         # # print(f" Q:{queries.shape}")
#         # # print(f" K:{keys.shape}")
#         # # print(f" V:{values.shape}")
#         attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=torch.float32))

#         if attention_mask is not None:
#             attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(2)

#         # Apply softmax to get attention probabilities
#         attention_probs = F.softmax(attention_scores, dim=-1)

#         # Take the weighted sum of the Value vectors using the attention probabilities
#         context = torch.matmul(attention_probs, values)

#         if output_attentions:
#             return context, attention_probs
#         else:
#             return context
# # class ModifiedViTModel2(ViTModel):
# #     def __init__(self, config: ViTConfig):
# #         super().__init__(config)
        
# #         # Replace the standard encoder with ModifiedViTEncoder (which uses DHSLayers)
# #         self.encoder = ModifiedViTEncoder(config)

# #         # Replace the standard attention layer with PrunedAttention
# #         self.pruned_attention = PrunedAttention(config)

# #         # Define a classifier for 100 classes
# #         self.classifier = nn.Linear(config.hidden_size, 100)

#     def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True, patch_indices_to_keep=None):
#         # Call the original forward method to get hidden states
#         outputs = super().forward(
#             pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict
#         )

#         # Get the last hidden states from the outputs
#         hidden_states = outputs.last_hidden_state  # (batch_size, num_patches, hidden_size)

#         # Reshape hidden_states for multi-head attention
#         batch_size, num_patches, hidden_size = hidden_states.size()
#         num_heads = self.config.num_attention_heads
#         hidden_states = hidden_states.view(batch_size, num_heads, num_patches, hidden_size // num_heads)

#         # Use the pruned attention mechanism with the specified patch indices
#         context = self.pruned_attention(hidden_states, output_attentions=output_attentions, patch_indices_to_keep=patch_indices_to_keep)

#         # Use the CLS token's hidden state (first token) for classification
#         logits = self.classifier(context[:, 0])  # Take the context of the first token for classification

#         if return_dict:
#             return {'logits': logits}
#         else:
#             return logits


class ModifiedViTModel(ViTModel):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        
        # Replace the standard encoder with ModifiedViTEncoder (which uses DHSLayers)
        self.encoder = ModifiedViTEncoder(config)
        
        # Define a classifier that maps the hidden size of the final CLS token output to 100 classes, to be used for classification
        self.classifier = nn.Linear(config.hidden_size, 100)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        # Call the forward method of the original ViTModel (now with the modified encoder)
        outputs = super().forward(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # Use the CLS token's hidden state (first token) from the last layer for classification
        logits = self.classifier(outputs['last_hidden_state'][:, 0])
        
        # Define a simple object to hold the logits (simulating a dictionary output)
        output = lambda: None
        setattr(output, 'logits', logits)

        return output                               # Return the output object containing logits


# Imports a ViTConfig from a pre-trained configuration
config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
# Instantiate the modified model with the given configuration
modified_vit_model = ModifiedViTModel(config)
# modified_vit_model = ModifiedViTModel2(config)


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# # print(device)

cifar_processor = AutoImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100")   # Initialize the image processor for CIFAR-100 data pre-processing



############ ------- Custom dataset class for CIFAR-100 that integrates with AutoImageProcessor for processing ---------- ##########
class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, count=None, processor=None):
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True)           # Load CIFAR-100 dataset from torchvision
        if count is not None:
            self.dataset = data_utils.Subset(self.dataset, torch.arange(count))                       # Optionally, load only count number of samples
        
        self.processor = processor                                                                    # Store the processor (used to preprocess images before passing to the model)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx] 
        
        # Preprocess the image and convert it to tensor format suitable for the model
        inputs = self.processor(images=image, return_tensors="pt")
        # Return the processed pixel values and label
        return inputs['pixel_values'].squeeze(0), label        


data_path = 'data'
train_dataset = CIFAR100Dataset(root=data_path, train=True, count=10000, processor=cifar_processor)
test_dataset = CIFAR100Dataset(root=data_path, train=False, count=1000, processor=cifar_processor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


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
# def train(model, train_loader, test_loader, num_epochs=10, loss_type='both', lr=1e-4, progress=True):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     cosine_loss_ratio = 1.5

#     optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
#     temp_train_loader = train_loader
#     for epoch in range(num_epochs):
#             model.train()  # Re-enable training mode at the start of each epoch
#             running_loss = 0.0
#             # total_correct_mlp = torch.zeros(len(model.encoder.layer))
#             # total_mlp_count = torch.zeros(len(model.encoder.layer))
#             if progress:
#                 train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

#             for batch_idx, (inputs, labels) in enumerate(train_loader):
#                 inputs, labels = inputs.to(device), labels.to(device)                

#                 # Forward pass to get the classification logits, loss calculation
#                 logits = model(inputs).logits

#                 if loss_type in ['classification', 'both']:
#                     classification_loss = criterion(logits, labels)

#                 # Accumulate cosine similarity losses from each DHSLayer
#                 if loss_type in ['cosine', 'both']:
#                     cosine_loss = 0.0
#                     for i, vit_layer in enumerate(model.encoder.layer):
#                         # total_correct_mlp[i] += (vit_layer.mlp_accuracy_arr).sum().item()
#                         # total_mlp_count[i] += vit_layer.mlp_accuracy_arr.numel()
#                         cosine_loss += vit_layer.loss
                
#                 # Calculate the total loss as a weighted sum of classification and cosine similarity losses
#                 if loss_type == 'classification':
#                     total_loss = classification_loss
#                 elif loss_type == 'cosine':
#                     total_loss = cosine_loss
#                 elif loss_type == 'both':
#                     total_loss = classification_loss + cosine_loss_ratio * cosine_loss
                
#                 optimizer.zero_grad()
#                 total_loss.backward()
#                 optimizer.step()

#                 # Accumulate running loss for this batch
#                 running_loss += total_loss.item()

#                 # Update the progress bar with current batch loss
#                 if progress:
#                     if (batch_idx + 1) % 5 == 0:
#                         train_loader.update(5)
#                     train_loader.set_postfix({'batch_loss': total_loss.item()})

#             print(f"")
#             print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
#             print(f"Test accuracy after {epoch + 1} epochs: {test(model, test_loader, full_testing=True, progress=progress)}")
#             # print(f"Train accuracy after {epoch + 1} epochs: {test(model, temp_train_loader, full_testing=True, progress=progress)}")










# Function to evaluate the model on the test data
def test(model, dataloader, progress=True):
    model.eval()
    total_correct = 0
    total_skip = 0
    data_size = len(dataloader.dataset)

    with torch.no_grad():             # Disable gradient computation for efficiency
        if progress:
            dataloader = tqdm(dataloader, desc=f"Testing", leave=False)
        for (inputs, labels) in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            logits = outputs.logits

            predicted_class_indices = logits.argmax(dim=-1)    # Get the class index with the highest logit score

            # Track the skip ratios from each layer            
            for layer in model.encoder.layer:
                total_skip += layer.skip_ratio
            total_correct += torch.sum(predicted_class_indices == labels).item()

    # print(total_skip / data_size / len(model.encoder.layer))
    return total_correct / data_size


# Training function to train the model over multiple epochs
def train(model, train_loader, test_loader, num_epochs=10, lr=1e-4, progress=True):
    model.train()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)

    for epoch in range(num_epochs):
            model.train()  # Re-enable training mode at the start of each epoch
            running_loss = 0.0
            
            if progress:
                train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass to get the classification logits, loss calculation
                logits = model(inputs).logits
                classification_loss = criterion(logits, labels)

                # Accumulate cosine similarity losses from each DHSLayer
                cosine_loss = 0.0
                for i in range(len(model.encoder.layer)):
                    vit_layer = model.encoder.layer[i]
                    cosine_loss += vit_layer.loss                   # `loss` calculated within each DHSLayer
                
                # Calculate the total loss as a weighted sum of classification and cosine similarity losses
                total_loss = classification_loss + 1.5 * cosine_loss
                
                # Backward pass to compute gradients and update weights
                total_loss.backward()
                optimizer.step()

                # Accumulate running loss for this batch
                running_loss += total_loss.item()

                # Update the progress bar with current batch loss
                if progress:
                    if (batch_idx + 1) % 5 == 0:
                        train_loader.update(5)
                    train_loader.set_postfix({'batch_loss': total_loss.item()})

            print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
            print(f"Test accuracy after {epoch + 1} epochs: {test(model, test_loader, progress)}")



load_pretrained = False            # Flag to load a pre-trained model (currently set to False)

# -----------   Copying Weights    ------------  #
if not load_pretrained:
    pretrained_cifar_model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")

    new_state_dict = {}
    for key in pretrained_cifar_model.state_dict().keys():
        new_key = key.replace('vit.', '')
        new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

    modified_vit_model.load_state_dict(new_state_dict, strict=False)
else:
    modified_vit_model.load_state_dict(torch.load('model2.pth'))

modified_vit_model.to(device)


# Set all parameters in the model to have no gradient updates by default
for param in modified_vit_model.parameters():
    param.requires_grad = False

# Enable gradients for the MLP (feed-forward) layers in each Transformer layer in the encoder
for layer in modified_vit_model.encoder.layer:
    for param in layer.mlp_layer.parameters():
        param.requires_grad = True

# Enable gradients for the classifier layer as it directly impacts class predictions
for param in modified_vit_model.classifier.parameters():
    param.requires_grad = True


modified_vit_model.to(device)

# Define the training parameters
num_epochs = 15
lr = 2e-3

input_size = (3, 224, 224)


flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True,  print_per_layer_stat=True)
flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True,   print_per_layer_stat=True)

print(flops, flops_ori)


print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, False)}')
train(modified_vit_model, train_loader, test_loader, num_epochs, lr, progress=True)

torch.save(modified_vit_model.state_dict(), 'model4.pth')

"""# pretrained model accuracy 89.85"""
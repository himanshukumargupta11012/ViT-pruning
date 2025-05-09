# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)

import os, sys
from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import confusion_matrix

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print("CUDA device:", os.getenv("CUDA_VISIBLE_DEVICES"), ' '.join(sys.argv))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPooling
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
# Assuming ViTLayer is imported correctly from transformers.models.vit.modeling_vit
from transformers.models.vit.modeling_vit import ViTLayer, ViTConfig # Make sure to import ViTLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
# Assuming ViTLayer is imported correctly from transformers.models.vit.modeling_vit
from transformers.models.vit.modeling_vit import ViTLayer, ViTConfig # Make sure to import ViTLayer
# Need sklearn for confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np # Import numpy

# Dummy ViTConfig for testing if needed
# config = ViTConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072)

class ModifiedViTLayer(ViTLayer):
    def __init__(self, config, sim_threshold=0.9, mlp_threshold=0.1, mlp_needed=True):
        """
        Args:
            config (ViTConfig): Configuration object for ViT.
            sim_threshold (float): Threshold for similarity-based 'ground truth' for CM.
            mlp_threshold (float): Threshold on the predicted attention score (MLP output)
                                   to decide whether to process a patch token and for CM 'predicted label'.
            mlp_needed (bool): If False, behaves like a standard ViTLayer.
        """
        super().__init__(config)

        self.mlp_needed = mlp_needed
        self.config = config # Store config for easy access

        self.loss = torch.tensor(0.0) # Initialize loss as a tensor
        self.mlp_confusion_matrix = None # Initialize confusion matrix

        if not self.mlp_needed:
            return

        self.hidden_size = config.hidden_size
        # MLP input is concatenation of [CLS, Patch_i], so 2 * hidden_size
        # Output is a single score (predicted attention)
        mlp_input_dim = 2 * self.hidden_size
        # Example MLP structure: [2*D, 64, 1] - adjust as needed
        layer_sizes = [mlp_input_dim, 64, 1]

        mlp_layers = []
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
            else:
                # Final layer outputs a score between 0 and 1
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])

        self.mlp_layer = nn.Sequential(*mlp_layers)
        self.mlp_threshold = mlp_threshold
        self.sim_threshold = sim_threshold # Store sim_threshold for CM
        # Use MSELoss for regressing predicted score towards target attention score
        self.loss_fct = nn.MSELoss()
        # Store predicted scores and target scores for analysis if needed
        self.mlp_predicted_scores = None
        self.target_attention_scores = None


    # Note: Kept the original forward signature as requested
    def forward(self, hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False, compute_cosine = False, output_mask=False): # compute_cosine currently unused internally but kept in signature
        top_k=150
        batch_size, seq_len, dim = hidden_states.shape
        device = hidden_states.device

        # --- Standard Layer Behavior (if mlp_needed is False) ---
        if not self.mlp_needed:
            # Ensure output format matches expected tuple structure
            layer_output = super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
            # Standard ViTLayer output is (hidden_states,) or (hidden_states, attention_probs)
            final_output = layer_output[0]
            if output_mask:
                 # Create a mask where everything is selected
                 boolean_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
                 # Ensure output tuple matches expected structure from modified path
                 return (final_output, boolean_mask) + layer_output[1:] # Appends attn if present
            else:
                 # Ensure output tuple matches expected structure from modified path
                 return (final_output,) + layer_output[1:] # Appends attn if present
      

        # --- MLP Prediction Phase ---
        cls_token = hidden_states[:, 0:1, :]    # Shape: [B, 1, D]
        patch_tokens = hidden_states[:, 1:, :]  # Shape: [B, N, D] (N = seq_len - 1)
        num_patch_tokens = patch_tokens.shape[1]

        # Expand CLS token to match the number of patch tokens
        expanded_cls_token = cls_token.expand(-1, num_patch_tokens, -1) # Shape: [B, N, D]

        # Concatenate CLS and each patch token
        mlp_input = torch.cat((expanded_cls_token, patch_tokens), dim=-1) # Shape: [B, N, 2*D]
        # print(attention_probs.shape)
        # print(attention_probs[0])

        # Get predicted scores from MLP
        # mlp_input shape is [B, N, 2*D] -> mlp_output shape is [B, N, 1]
        mlp_output_scores = self.mlp_layer(mlp_input).squeeze(-1) # Shape: [B, N]

        pred_labels = mlp_output_scores

        # print(pred_labels.shape, true_labels.shape)

        self.mlp_predicted_scores = mlp_output_scores # Store for potential analysis
        topk_values, topk_indices = torch.topk(mlp_output_scores, k=top_k, dim=1)

        # Now create the mask
        mask = torch.zeros_like(mlp_output_scores, dtype=torch.bool)

        # Scatter True at topk_indices
        mask.scatter_(1, topk_indices, True)

        # --- Mask Creation ---
        # Create mask based on thresholding MLP scores
        # patch_mask = (mlp_output_scores >= self.mlp_threshold) # Shape: [B, N]
        patch_mask = mask # Shape: [B, N]
        # Always include the CLS token
        cls_mask_col = torch.ones((batch_size, 1), dtype=torch.bool, device=device) # Shape: [B, 1]

        # Combine CLS mask and patch mask
        boolean_mask = torch.cat((cls_mask_col, patch_mask), dim=1) # Shape: [B, N+1] (or [B, seq_len])


        # --- Selective Forward Pass ---
        # Initialize output as a copy of input (tokens not processed retain original values)
        output = hidden_states.clone()

        # Process batch by batch using the mask
        for i in range(batch_size):
            # Get indices of tokens to process for this batch item
            selected_indices = boolean_mask[i].nonzero(as_tuple=False).squeeze(-1)

            if len(selected_indices) > 0: # Check if any tokens are selected
                # Select the tokens for this batch item
                tokens_to_process = hidden_states[i][selected_indices].unsqueeze(0) # Add batch dim back [1, num_selected, D]

                # Apply the standard ViTLayer forward pass ONLY to the selected tokens
                processed_tokens_output = super().forward(
                    tokens_to_process,
                    head_mask=head_mask, # Apply head_mask if provided (applied internally by super().forward)
                    output_attentions=False # No need for attentions here
                )[0].squeeze(0) # Get hidden states output, remove batch dim [num_selected, D]

                # Place the processed tokens back into the correct positions in the output tensor
                output[i][selected_indices] = processed_tokens_output


        # --- Loss and Confusion Matrix Calculation (only during training) ---
        attention_probs = None # Initialize attention_probs for potential return later
        if self.training:
            # Perform a FULL forward pass on the ORIGINAL hidden_states to get target attention scores
            # and the full output needed for similarity calculation
            with torch.no_grad(): # No need to compute gradients for the target/similarity calculation step
                 full_layer_output_tuple = super().forward(
                     hidden_states, # Use original full input
                     head_mask=head_mask,
                     output_attentions=True # Need attentions for loss target
                 )
                 real_output = full_layer_output_tuple[0] # Full output needed for similarity: [B, N+1, D]
                 attention_probs = full_layer_output_tuple[1] # Attentions: [B, H, N+1, N+1]

                 # 1. Calculate Target Attention Scores (for MSE Loss)
                 cls_to_patch_attention = attention_probs[:, :, 0, 1:] # [B, H, N]
                 target_attention_scores = cls_to_patch_attention.mean(dim=1) # [B, N]
                 self.target_attention_scores = target_attention_scores # Store for potential analysis

                 # 2. Calculate Similarity (for Confusion Matrix 'True Labels')
                #  input_patches = hidden_states[:, 1:]   # [B, N, D]
                #  output_patches = real_output[:, 1:]    # [B, N, D]


            layer_output = super().forward(hidden_states, head_mask=head_mask, output_attentions=True)
                # Standard ViTLayer output is (hidden_states,) or (hidden_states, attention_probs)
            # print(len(layer_output))
            # final_output = layer_output[0]
            # print(len(final_output))
            hidden_states, attention_probs = layer_output
            cls_to_patches = attention_probs[:, :, 0, 1:]  # shape: [batch, num_heads, 196]

            # Step 2: Average across heads
            avg_cls_attention = cls_to_patches.mean(dim=1)  # shape: [batch, 196]


            true_labels=avg_cls_attention
            
            # Calculate MSE loss: MLP predicted score vs. Target Attention Score
            # pred_labels = (mlp_output_scores > self.mlp_threshold).int().detach().cpu().flatten() # [B*N]
            # current_loss = self.loss_fct(pred_labels, true_labels)
            current_loss = F.mse_loss(pred_labels, true_labels, reduction='mean')

            self.loss = current_loss

            # pred_labels = (mlp_output_scores > self.mlp_threshold).int().detach().cpu().flatten() # [B*N]

            # Clip and compute confusion matrix
            # true_labels = np.clip(true_labels, 0, 1)
            # pred_labels = np.clip(pred_labels, 0, 1)
            # self.mlp_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

        else:
            # Ensure loss is zero and CM is None during inference
            self.loss = torch.tensor(0.0, device=device)
            self.mlp_confusion_matrix = None
            self.target_attention_scores = None


        # --- Prepare Final Output ---
        # The main output is the 'output' tensor containing selectively processed tokens
        final_outputs = (output,)

        # Append the boolean mask if requested
        if output_mask:
            final_outputs += (boolean_mask,)

        # Append attention probabilities if requested AND available (i.e., during training)
        # Note: attention_probs is from the *full* pass calculated during training
        if output_attentions and attention_probs is not None:
            final_outputs += (attention_probs,)
        # If output_attentions is True but we are not training or mlp not needed,
        # the standard ViTLayer `super().forward` call at the beginning (or inside the loop)
        # might have produced attentions. We need to ensure consistent tuple length.
        # However, the current structure only calculates `attention_probs` during training's full pass.
        # For simplicity and consistency with the request to modify inside the function,
        # we only return attentions calculated during the training's full pass.
        # If inference-time attentions are needed with MLP active, the selective pass
        # would need modification.

        return final_outputs # (output_hidden_states, [optional boolean_mask], [optional attention_probs])

    def get_loss(self):
        """Helper method to retrieve the computed loss (MSE)."""
        return self.loss

    def get_scores(self):
        """Helper method to retrieve predicted and target scores (useful for debugging/analysis)."""
        return self.mlp_predicted_scores, self.target_attention_scores

    def get_confusion_matrix(self):
        """Helper method to retrieve the computed confusion matrix (during training)."""
        return self.mlp_confusion_matrix





# Dummy ViTConfig for testing if needed
# config = ViTConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072)

# class ModifiedViTLayer(ViTLayer):
#     def __init__(self, config,sim_threshold=0.9, mlp_threshold=0.1, mlp_needed=True):
#         """
#         Args:
#             config (ViTConfig): Configuration object for ViT.
#             mlp_threshold (float): Threshold on the predicted attention score
#                                    to decide whether to process a patch token.
#             mlp_needed (bool): If False, behaves like a standard ViTLayer.
#         """
#         super().__init__(config)
        
#         self.mlp_needed = mlp_needed
#         self.config = config # Store config for easy access

#         self.loss = torch.tensor(0.0) # Initialize loss as a tensor
#         if not self.mlp_needed:
#             return

#         self.hidden_size = config.hidden_size
#         # MLP input is concatenation of [CLS, Patch_i], so 2 * hidden_size
#         # Output is a single score (predicted attention)
#         mlp_input_dim = 2 * self.hidden_size
#         # Example MLP structure: [2*D, 64, 1] - adjust as needed
#         layer_sizes = [mlp_input_dim, 64, 1]

#         mlp_layers = []
#         for i in range(len(layer_sizes) - 1):
#             if i < len(layer_sizes) - 2:
#                 mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
#             else:
#                 # Final layer outputs a score between 0 and 1
#                 mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])

#         self.mlp_layer = nn.Sequential(*mlp_layers)
#         self.mlp_threshold = mlp_threshold
#         # Use MSELoss for regressing predicted score towards target attention score
#         self.loss_fct = nn.MSELoss()
#         # Store predicted scores and target scores for analysis if needed
#         self.mlp_predicted_scores = None
#         self.target_attention_scores = None


#     # def forward(self,
#     #     hidden_states: torch.Tensor,
#     #     head_mask: Optional[torch.Tensor] = None, # head_mask is for the main attention, not the MLP
#     #     output_attentions: bool = False, # Request attention scores from the main layer if needed
#     #     output_mask: bool = False # Flag to return the computed boolean mask
#     #     ) -> Tuple[torch.Tensor, ...]:


#     def forward(self, hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False, compute_cosine = False, output_mask=False):

#         batch_size, seq_len, dim = hidden_states.shape
#         device = hidden_states.device
#         layer_output = super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
#         final_output = layer_output[0]
#         # If MLP is not needed, behave like a standard layer
#         if not self.mlp_needed:
#             # Ensure output format matches expected tuple structure
#             layer_output = super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
#             # Standard ViTLayer output is (hidden_states,) or (hidden_states, attention_probs)
#             final_output = layer_output[0]
#             if output_mask:
#                  # Create a mask where everything is selected
#                  boolean_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
#                  return (final_output, boolean_mask) + layer_output[1:]
#             else:
#                  return layer_output # Return standard output tuple


#         # --- MLP Prediction Phase ---
#         cls_token = hidden_states[:, 0:1, :]    # Shape: [B, 1, D]
#         patch_tokens = hidden_states[:, 1:, :]  # Shape: [B, N, D] (N = seq_len - 1)
#         num_patch_tokens = patch_tokens.shape[1]

#         # Expand CLS token to match the number of patch tokens
#         expanded_cls_token = cls_token.expand(-1, num_patch_tokens, -1) # Shape: [B, N, D]

#         # Concatenate CLS and each patch token
#         mlp_input = torch.cat((expanded_cls_token, patch_tokens), dim=-1) # Shape: [B, N, 2*D]

#         # Get predicted scores from MLP
#         # mlp_input shape is [B, N, 2*D] -> mlp_output shape is [B, N, 1]
#         mlp_output_scores = self.mlp_layer(mlp_input).squeeze(-1) # Shape: [B, N]
#         self.mlp_predicted_scores = mlp_output_scores # Store for potential analysis
        
#         mlp_output = mlp_output_scores
#         # simia
#         # print(final_output.shape,mlp_output.shape)
#         true_labels =(similarity_val < self.sim_threshold).int().cpu().flatten()
#         pred_labels = (mlp_output.squeeze(-1) > self.mlp_threshold).int().cpu().flatten()
#         # print(true_labels.shape, pred_labels.shape)
#         self.mlp_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])



#         # --- Mask Creation ---
#         # Create mask based on thresholding MLP scores
#         patch_mask = (mlp_output_scores >= self.mlp_threshold) # Shape: [B, N]

#         # Always include the CLS token
#         cls_mask_col = torch.ones((batch_size, 1), dtype=torch.bool, device=device) # Shape: [B, 1]

#         # Combine CLS mask and patch mask
#         boolean_mask = torch.cat((cls_mask_col, patch_mask), dim=1) # Shape: [B, N+1] (or [B, seq_len])


#         # --- Selective Forward Pass ---
#         # Initialize output as a copy of input (tokens not processed retain original values)
#         output = hidden_states.clone()

#         # Process batch by batch using the mask
#         # Note: This loop can be a bottleneck. Vectorized operations (like torch.gather/scatter or
#         # padding/masking techniques if applicable) might be faster but more complex to implement correctly
#         # across batches with varying numbers of selected tokens.
#         for i in range(batch_size):
#             # Get indices of tokens to process for this batch item
#             selected_indices = boolean_mask[i].nonzero(as_tuple=False).squeeze(-1)

#             if len(selected_indices) > 0: # Check if any tokens are selected
#                 # Select the tokens for this batch item
#                 tokens_to_process = hidden_states[i][selected_indices].unsqueeze(0) # Add batch dim back [1, num_selected, D]

#                 # Apply the standard ViTLayer forward pass ONLY to the selected tokens
#                 # We don't need attentions from this inner pass for the loss calculation
#                 processed_tokens_output = super().forward(
#                     tokens_to_process,
#                     head_mask=head_mask, # Apply head_mask if provided (applied internally by super().forward)
#                     output_attentions=False # No need for attentions here
#                 )[0].squeeze(0) # Get hidden states output, remove batch dim [num_selected, D]

#                 # Place the processed tokens back into the correct positions in the output tensor
#                 output[i][selected_indices] = processed_tokens_output


#         # --- Loss Calculation (only during training) ---
#         if self.training:
#             # Perform a FULL forward pass on the ORIGINAL hidden_states to get target attention scores
#             # We need the attention probabilities from this pass
#             with torch.no_grad(): # No need to compute gradients for the target calculation step
#                  # Set output_attentions=True to get the attention probabilities
#                  full_layer_output = super().forward(
#                      hidden_states, # Use original full input
#                      head_mask=head_mask,
#                      output_attentions=True
#                  )
#                  # Output is (final_hidden_state, attention_probs)
#                  attention_probs = full_layer_output[1] # Shape: [B, num_heads, N+1, N+1]

#                  # Extract attention from CLS token (query idx 0) to patch tokens (key idx 1 to N+1)
#                  # Shape: [B, num_heads, N]
#                  cls_to_patch_attention = attention_probs[:, :, 0, 1:]

#                  # Average attention scores across heads to get a single score per patch
#                  # Shape: [B, N]
#                  target_attention_scores = cls_to_patch_attention.mean(dim=1)
#                  self.target_attention_scores = target_attention_scores # Store for potential analysis

#             # Calculate loss: MSE between MLP predicted scores and target attention scores
#             # mlp_output_scores shape: [B, N]
#             # target_attention_scores shape: [B, N]
#             current_loss = self.loss_fct(mlp_output_scores, target_attention_scores)
#             self.loss = current_loss
#         else:
#             # Ensure loss is zero or None during inference
#             self.loss = torch.tensor(0.0, device=device)
#             self.target_attention_scores = None # Clear stored targets


#         # --- Prepare Final Output ---
#         # The main output is the 'output' tensor containing selectively processed tokens
#         final_outputs = (output,)

#         # Append the boolean mask if requested
#         if output_mask:
#             final_outputs += (boolean_mask,)

#         # Append attention probabilities from the *selective* forward pass if requested by the *caller*.
#         # This is tricky because the selective pass didn't compute them by default.
#         # If needed, you'd have to run the selective pass *again* with output_attentions=True,
#         # or modify the loop structure. For simplicity, we currently don't return attentions
#         # from the selective pass. If the caller sets output_attentions=True, they'll get None
#         # unless the mlp_needed=False path is taken.
#         # A compromise: if output_attentions is True during training, return the attentions
#         # from the *full* pass we used for loss calculation?
#         if output_attentions and self.training and self.mlp_needed:
#              # Return attentions calculated for the loss target
#              final_outputs += (attention_probs,)
#         elif output_attentions and not self.mlp_needed:
#              # From the initial super().forward call when mlp_needed=False
#              final_outputs += layer_output[1:]
#         # else: # Not training or not requested
#              # Append None or handle appropriately based on ViTLayer's expected output tuple size
#              # Standard ViTLayer returns (hidden_states,) or (hidden_states, attns)
#              # To match, we might need to append None if attentions were requested but not computed/returned
#              # pass # The tuple already has the correct number of elements based on previous logic


#             # true_labels = (similarity_val < self.sim_threshold).int().cpu().flatten()
#             # pred_labels = (mlp_output.squeeze(-1) > self.mlp_threshold).int().cpu().flatten()
#             # self.mlp_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])


#         return final_outputs # (output_hidden_states, [optional boolean_mask], [optional attention_probs])

#     def get_loss(self):
#         """Helper method to retrieve the computed loss."""
#         return self.loss

#     def get_scores(self):
#         """Helper method to retrieve predicted and target scores (useful for debugging/analysis)."""
#         return self.mlp_predicted_scores, self.target_attention_scores



# class ModifiedViTLayer(ViTLayer):
#     def __init__(self, config, sim_threshold=0.9, mlp_threshold=0.5, mlp_needed=True):
#         super().__init__(config)
#         self.mlp_needed = mlp_needed

#         self.loss = 0
#         if not self.mlp_needed:
#             return
#         self.hidden_size = config.hidden_size
#         layer_sizes = [self.hidden_size, 64, 1]

#         mlp_layers = []
#         for i in range(len(layer_sizes) - 1):
#             if i < len(layer_sizes) - 2:
#                 mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
#             else:
#                 mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])                

#         self.mlp_layer = nn.Sequential(*mlp_layers)
#         self.sim_threshold = sim_threshold
#         self.mlp_threshold = mlp_threshold


#     def forward(self, hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False, compute_cosine = False, output_mask=False):

#         boolean_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool).to(hidden_states.device)
#         if not self.mlp_needed:
#             if output_mask:
#                 return (super().forward(hidden_states)[0], boolean_mask)
#             else:
#                 return super().forward(hidden_states)

#         # expanded_cls_token = hidden_states[:, 0: 1].repeat(1, 196, 1)
#         # mlp_input = torch.cat((expanded_cls_token, hidden_states[:, 1:]), dim=-1)
#         mlp_input = hidden_states[:, 1:]
#         mlp_output = self.mlp_layer(mlp_input)
#         boolean_mask = (mlp_output >= self.mlp_threshold).squeeze(-1)
#         cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
#         boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

#         output = hidden_states.clone()
#         batch_size = hidden_states.shape[0]
#         for i in range(batch_size):
#             output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)

#         if self.training or compute_cosine:
#             # real_output = super().forward(hidden_states[:, 1:])[0]
#             # cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
#             # # cos_similarity = torch.abs(F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1))
#             # euclidean_dist = torch.sum((real_output - hidden_states[:, 1:]) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
#             real_output = super().forward(hidden_states[:, :])[0]
#             cos_similarity = (F.cosine_similarity(real_output[:, 1:], hidden_states[:, 1:], dim=-1) + 1) / 2
#             # cos_similarity = torch.abs(F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1))
#             euclidean_dist = torch.sum((real_output[:, 1:] - hidden_states[:, 1:]) ** 2, dim=-1) / torch.sum(real_output[:, 1:]**2, dim=-1)
#             dist_similarity = 1 / (1 + euclidean_dist)
#             alpha = 0.5
#             similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity
            
#             gamma = 2  # Controls how much we focus on hard examples
#             pt = mlp_output.squeeze(-1)  # Model prediction probability
            
#             focal_weight = (1 - pt) ** gamma  # More weight for misclassified examples
#             target = (similarity_val < self.sim_threshold).float()

#             # else:
#             mlploss = (focal_weight * nn.BCELoss(reduction='none')(pt, target)).mean() 
#             self.loss=mlploss
#             # mlp_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
#             # # mlp_criterion = nn.BCEWithLogitsLoss()
#             # self.loss = mlp_criterion(mlp_output.squeeze(-1), (similarity_val < self.sim_threshold).float())
#             self.mlp_accuracy_arr = ((self.sim_threshold - similarity_val) * (mlp_output.squeeze(-1) - self.mlp_threshold) > 0)

#             true_labels = (similarity_val < self.sim_threshold).int().cpu().flatten()
#             pred_labels = (mlp_output.squeeze(-1) > self.mlp_threshold).int().cpu().flatten()
#             self.mlp_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])


#         else:
#             self.loss = 0
            
#         if output_mask:
#             return (output, boolean_mask)
#         else:
#             return (output, )

class ModifiedViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig, sim_threshold=0.9, mlp_threshold=0.5):
        super().__init__(config)
        mlp_needed_arr = torch.ones(config.num_hidden_layers, dtype=torch.bool)
        # mlp_needed_arr[8] = False
        # indices_to_change = [6, 7, 8]
        # mlp_needed_arr[indices_to_change] = True

        self.layer = nn.ModuleList([ModifiedViTLayer(config, sim_threshold, mlp_threshold, mlp_needed_arr[_]) for _ in range(config.num_hidden_layers)])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        compute_cosine: bool = False,
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
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, compute_cosine=compute_cosine, output_mask=output_mask)

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
    def __init__(self, config: ViTConfig, sim_threshold=0.9, mlp_threshold=0.5):
        super().__init__(config)
        self.encoder = ModifiedViTEncoder(config, sim_threshold, mlp_threshold)
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
        compute_cosine = False,
        output_mask: Optional[bool] = None,
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
            compute_cosine=compute_cosine,
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

    def vit_mlp_train(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def vit_train(self):
        for param in self.parameters():
            param.requires_grad = True

        for layer in self.encoder.layer:
            if not hasattr(layer, 'mlp_layer'):
                continue
            for param in layer.mlp_layer.parameters():
                param.requires_grad = False
    
    def mlp_train(self):
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.encoder.layer:
            if not hasattr(layer, 'mlp_layer'):
                continue
            for param in layer.mlp_layer.parameters():
                param.requires_grad = True


    def classifier_train(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def classifier_mlp_train(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        for layer in self.encoder.layer:
            if not hasattr(layer, 'mlp_layer'):
                continue
            for param in layer.mlp_layer.parameters():
                param.requires_grad = True

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

def write_N_print(string):
    print(string)
    log_file.write(string + "\n")
    log_file.flush()


# Training function to train the model over multiple epochs
def train(model, train_loader, test_loader, save_path, num_epochs=10, loss_type='both', lr=1e-4):
    model.train()
    criterion = nn.CrossEntropyLoss()
    cosine_loss_ratio = 1.5

    best_val_accuracy = 0.0
    best_model = None

    
    model_save_path = f"models/{save_path}.pth"
    

    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
    temp_train_loader = train_loader
    for epoch in range(num_epochs):
        model.train()  # Re-enable training mode at the start of each epoch
        running_loss = 0.0
        # total_correct_mlp = torch.zeros(len(model.encoder.layer))
        # total_mlp_count = torch.zeros(len(model.encoder.layer))

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
            if (batch_idx + 1) % tqdm_update_step == 0:
                train_loader.update(tqdm_update_step)
            # train_loader.set_postfix({'batch_loss': total_loss.item()})

        val_accuracy, mlp_val_accuracy = test(model, test_loader, full_testing=True)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            torch.save(best_model, model_save_path)

        write_N_print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
        write_N_print(f"Test accuracy after {epoch + 1} epochs: {val_accuracy}\n")

    # torch.save(best_model, model_save_path)
    test(model, temp_train_loader, full_testing=True)
    write_N_print(f"Best accuracy: {best_val_accuracy}")


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

    # write_N_print(f"Skip ratio: {each_layer_skip / data_size} and {torch.sum(each_layer_skip) / data_size / len(model.encoder.layer)}")

    if full_testing:
        each_layer_skip = (total_correct_mlp.sum(dim=[1], keepdim=True) / total_correct_mlp.sum(dim=[1, 2], keepdim=True))[:, 0, 0]
        write_N_print(f"Skip ratio: {each_layer_skip}\nand {each_layer_skip.mean()}")
        mlp_accuracy = (torch.sum(total_correct_mlp[:, 1, 1]) + torch.sum(total_correct_mlp[:, 0, 0])) / (torch.sum(total_correct_mlp) + 1e-16)
        mlp_confusion_mat = (total_correct_mlp / (total_correct_mlp.sum(dim=[-2, -1], keepdim=True) + 1e-16)).tolist()
        mlp_accuracy_arr = (total_correct_mlp[:, 1, 1] + total_correct_mlp[:, 0, 0]) / total_correct_mlp.sum(dim=[1, 2])
        formatted_lst = [[[round(num, 4) for num in row] for row in col]  for col in mlp_confusion_mat]

        for i in range(len(mlp_accuracy_arr)):
            write_N_print(f"Correct prediction ratio for layer {i}:\n{formatted_lst[i]}\t{mlp_accuracy_arr[i]}")
        write_N_print(f"Overall accuracy of MLP: {mlp_accuracy}")

        return total_correct / data_size, mlp_accuracy
    
    return total_correct / data_size

# Define the training parameters
pretrained_model_path = False#"/home/ai21btech11012/ViT-pruning/himanshu/models/MLP_trained_2025-02-21_09-45-24_lr-0.0001_st-0.9_mt-0.5_bs-64_trs-10000_tes-1000_nw-16_type-cosine.pth"
loss_type = ["cosine", "classification"]
num_epochs = [10, 5]
lr = [5e-4, 1e-4]
data_path = '../himanshu/data'
train_batch_size = 64
test_batch_size = 128
train_size = 10000
test_size = 1000
sim_threshold = 0.90
mlp_threshold = 0.5
num_workers = 16
pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"

combined_lr = "^".join(map(str, lr))
save_path = f"Vit_tuning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr-{combined_lr}_st-{sim_threshold}_mt-{mlp_threshold}_bs-{train_batch_size}_trs-{train_size}_tes-{test_size}_nw-{num_workers}"

cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)   # Initialize the image processor for CIFAR-100 data pre-processing
pretrained_cifar_model = AutoModelForImageClassification.from_pretrained(pretrained_cifar_model_name)
config = ViTConfig.from_pretrained(pretrained_cifar_model_name)
config.num_labels = len(config.id2label)
modified_vit_model = ModifiedViTModel(config, sim_threshold, mlp_threshold)

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


if loss_type == "cosine":
    modified_vit_model.mlp_train()
elif loss_type == "classification":
    modified_vit_model.vit_train()
elif loss_type == "both":
    modified_vit_model.classifier_mlp_train()


train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    log_file = open(f"logs/{save_path}.txt", "w")

    # print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True)}')

    train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs[0], loss_type=loss_type[0], lr=lr[0], save_path=save_path)
    
    # modified_vit_model.vit_train()
    modified_vit_model.vit_mlp_train()

    train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs[1], loss_type=loss_type[1], lr=lr[1], save_path=save_path)

    # # calculating flops
    input_size = (3, 224, 224)
    flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
    flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
    print(flops, flops_ori)

    log_file.close()

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
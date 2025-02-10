# -*- coding: utf-8 -*-

# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Set
from transformers.models.vit.modeling_vit import ViTOutput,ViTIntermediate,ViTModel, ViTLayer, ViTEncoder, ViTConfig,ViTAttention
class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # print(outputs)
        return outputs


import math
class ModifiedViTSelfAttention(nn.Module):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False,patch_indices_to_keep=None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # print(f" Q:{query_layer.shape}")
        # print(f" K:{key_layer.shape}")
        # print(f" V:{value_layer.shape}")
        # pruned_queries = self.prune_queries(query_layer, patch_indices_to_keep)
        query_layer=self.prune_queries(query_layer,patch_indices_to_keep)
        # print(f" Q':{query_layer.shape}")
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs



        # self_outputs = self.attention(hidden_states, head_mask, output_attentions,patch_indices_to_keep)
        # print('----')
        # print(hidden_states.shape)

        # # print(patch_indices_to_keep.shape)
        # print('----')
        # Create Query, Key, Value vectors
        
        # queries = self.query(hidden_states)  # (batch_size, num_patches, hidden_size)
        # keys = self.key(hidden_states)       # (batch_size, num_patches, hidden_size)
        # values = self.value(hidden_states)   # (batch_size, num_patches, hidden_size)

        # # Reshape to (batch_size, num_heads, num_patches, head_size)
        # queries = queries.view(hidden_states.size(0), -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # keys = keys.view(hidden_states.size(0), -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # values = values.view(hidden_states.size(0), -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # # Prune the Query vectors based on provided indices
        # # # print(patch_indices_to_keep)
        # # if queries.size(2) != patch_indices_to_keep.size(0):
        # #     # Adjust the patch_indices_to_keep mask to match the number of patches in `queries`
        # #     patch_indices_to_keep = patch_indices_to_keep[:queries.size(2)]
                
        # print(f" Q:{queries.shape}")
        # print(f" K:{keys.shape}")
        # print(f" V:{values.shape}")
        # pruned_queries = self.prune_queries(queries, patch_indices_to_keep)
        # print(f" Q':{pruned_queries.shape}")

        # # Calculate the attention scores (Q @ K^T)

        # queries=pruned_queries
        # if(len(queries.shape)!=4):
        #     queries=queries.squeeze()
            
        # # Calculate attention scores and apply softmax
        # attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        # attention_probs = F.softmax(attention_scores, dim=-1)
        # attention_probs = self.dropout(attention_probs)

        # # Weighted sum of values
        # context = torch.matmul(attention_probs, values)

        # # Concatenate heads and reshape to [batch, seq_len, hidden_size]
        # context = context.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, num_heads, head_size]
        # new_context_shape = context.size()[:-2] + (self.all_head_size,)  # [batch, seq_len, hidden_size]
        # context = context.view(new_context_shape)
        # # print(context.shape)
        # return context


    def prune_queries(self,queries,patch_indices_to_keep):
        
        # Select patches where patch_indices_to_keep is True
        # # print('apple')
        # # print(queries.shape)
        # # print(patch_indices_to_keep.shape)
        # # print("##")
        # # print(queries.shape)
        # # print('banana')
        
#  Q:torch.Size([1, 12, 197, 64])
#  Q':torch.Size([1, 12, 1, 150, 64])
        # print(patch_indices_to_keep.shape)
        # print(queries.shape)
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

class ViTSelfOutput(nn.Module):
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

# from pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

class ModifiedViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = ModifiedViTSelfAttention(config)
        self.attention2 = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        patch_indices_to_keep=None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions,patch_indices_to_keep)
        # self_outputs = self.attention2(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:] 
        return outputs



# class ModifiedViTAttention(nn.Module):
#     def __init__(self, config: ViTConfig) -> None:
#         super().__init__()
#         self.attention = ModifiedViTSelfAttention(config)
#         # self.attention2 = ViTSelfAttention(config)
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # Mimic output layer
#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#         self.pruned_heads = set()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         patch_indices_to_keep=None
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#         # Get attention outputs
#         # print(patch_indices_to_keep)
#         self_outputs = self.attention(hidden_states, patch_indices_to_keep)
#         # self_outputs2 = self.attention2(hidden_states, patch_indices_to_keep)
        
#         # # # print(self_outputs.shape)
#         # # # print((self_outputs2[0].shape))
#         attention_output=self_outputs

#         # Mimic behavior of self.output using inline operations
#         # attention_output = self_outputs[0]  # Get the main output from self.attention
#         attention_output = self.dense(attention_output)  # Linear transformation
#         attention_output = self.dropout(attention_output)  # Apply dropout
        
#         # Return the output in the expected format
#         # # print(attention_output.shape)
#         # # print(self_outputs.shape)
#         outputs = attention_output + self_outputs 
        
#         return (outputs,)

class ModifiedViTLayer(nn.Module):
    """Modified version of ViTLayer that switches between different attention mechanisms 
    based on the TrainingMLP flag, while keeping the structure similar to the original ViTLayer.
    """
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        
        # Store configuration
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        
        # Attention layers (Modified and original)
        self.attention = ModifiedViTAttention(config)
        self.attention2 = ViTAttention(config)  # Regular ViT attention
        
        # Intermediate and output layers
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        
        # LayerNorm before and after attention (as in ViT)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        patch_indices_to_keep: Optional[torch.Tensor] = None,
        TrainingMLP: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        Forward pass for the modified Vision Transformer layer. It uses different attention mechanisms 
        based on the 'TrainingMLP' flag.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, num_patches, embed_dim].
            head_mask (Optional[torch.Tensor]): Optional tensor for attention heads masking.
            output_attentions (bool): Whether to return attention weights.
            patch_indices_to_keep (Optional[torch.Tensor]): Boolean mask tensor to prune patches.
            TrainingMLP (bool): Flag to toggle between using the modified or original ViT attention.

        Returns:
            torch.Tensor: Output tensor after applying the selected attention mechanism.
        """
        normed_hidden_states = self.layernorm_before(hidden_states)
        if TrainingMLP:
            self_attention_outputs = self.attention2(normed_hidden_states, head_mask, output_attentions)
            temp_hidden_states=hidden_states
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:] 
        else:
            self_attention_outputs = self.attention(normed_hidden_states, head_mask, output_attentions, patch_indices_to_keep)
                
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:] 
            # print('##########')
            # print(attention_output.shape)
            # print(hidden_states.shape)
            # print(patch_indices_to_keep.shape)
            temp_hidden_states=hidden_states[:,patch_indices_to_keep,:]
            # print(temp_hidden_states)
            # print('##########')
        

        # print('##########')
        # print(attention_output.shape)
        # print(outputs)
        # print('##########')
        # print(patch_indices_to_keep.shape)
        # print('@@@@@@@@@@')

        # extrapolate_attention_output=patch_indices_to_keep  attention_output

        # hidden_states = attention_output + hidden_states
        hidden_states = attention_output + temp_hidden_states

        layer_output = self.layernorm_after(hidden_states)
        
        # Apply the intermediate feedforward layer
        layer_output = self.intermediate(layer_output)

        # Apply the output layer (final layer in the ViT block)
        layer_output = self.output(layer_output, hidden_states)

        # Return the output, including any additional outputs like attention weights
        outputs = (layer_output,) + outputs

        return outputs



# class ModifiedViTLayer(nn.Module):
#     def __init__(self, config: ViTConfig) -> None:  
#         super().__init__()
#         self.attention = ModifiedViTAttention(config)
#         self.attention2 = ViTAttention(config)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         patch_indices_to_keep=None,TrainingMLP=False
#     ) -> torch.Tensor:
#         # # print(patch_indices_to_keep)

#         if(TrainingMLP):
#             attention_output = self.attention(hidden_states, head_mask, output_attentions, patch_indices_to_keep=patch_indices_to_keep)
#             return attention_output



#         attention_output = self.attention(hidden_states, head_mask, output_attentions, patch_indices_to_keep=patch_indices_to_keep)
#         return attention_output

# class DHSLayer(ModifiedViTLayer):
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
#         # print(f" grape: {mlp_output.shape}")
#         # Applies the ViTLayer superclass forward pass only on tokens selected by boolean_mask
#         output = hidden_states.clone()
#         for i in range(batch_size):
#             # # print("map is")
#             # # print(boolean_mask[i])
#             # # print("map end")
#             # print(f"apple: {boolean_mask[i].shape}")
#             # print('!!!!!!!!!!!')
#             # print(hidden_states.shape)
#             # print('!!!!!!!!!!!!!!!')
#             # output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0),patch_indices_to_keep=boolean_mask[i])[0].squeeze(0)
        
#             output[i][boolean_mask[i]] = super().forward(hidden_states[i].unsqueeze(0),patch_indices_to_keep=boolean_mask[i])[0].squeeze(0)
        

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

class DHSLayer(ModifiedViTLayer):
    def __init__(self, config, threshold=0.05):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        layer_sizes = [self.hidden_size, 256, 64, 1]

        mlp_layers = []
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
            else:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])                

        self.mlp_layer = nn.Sequential(*mlp_layers)
        self.threshold = threshold
        self.loss = 0
        self.skip_ratio = 0


    def forward(self, hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False, compute_cosine = True):

        mlp_output = self.mlp_layer(hidden_states[:, 1:])
        boolean_mask = (mlp_output >= self.threshold).squeeze(-1)
        cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
        boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)
        # print('----------')
        # print(hidden_states.shape)
        # print('----------')
        output = hidden_states.clone()
        batch_size = hidden_states.shape[0]
        for i in range(batch_size):
            output[i][boolean_mask[i]] = super().forward(hidden_states[i].unsqueeze(0),patch_indices_to_keep=boolean_mask[i])[0].squeeze(0)

        if self.training or compute_cosine:
            real_output = super().forward(hidden_states[:, 1:],patch_indices_to_keep=boolean_mask[i],TrainingMLP=True)[0]
            cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
            euclidean_dist = torch.sum((real_output - hidden_states[:, 1:]) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
            dist_similarity = 1 / (1 + euclidean_dist)
            alpha = 0.5
            similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity

            # mean_real_output = real_output.mean(dim=0)
            # cov_matrix = torch.cov(real_output.T)  # Calculate covariance matrix
            # inv_cov_matrix = torch.linalg.inv(cov_matrix)  # Inverse of the covariance matrix
            # delta = real_output - hidden_states[:, 1:]
            # mahalanobis_dist = torch.sqrt(torch.sum(delta @ inv_cov_matrix * delta, dim=1))
            # similarity_val = 1 / (1 + mahalanobis_dist) 
            # Compute Manhattan (L1) distance
            # manhattan_dist = torch.sum(torch.abs(real_output - hidden_states[:, 1:]), dim=-1)

            # Combine with Cosine Similarity (similar to what you're doing with alpha)
            # similarity_val = 0.5 * cos_similarity + (0.5) * (1 / (1 + manhattan_dist))



            # print(f'similarity: {similarity_val}') 
            # self.loss = F.kl_div(mlp_output.squeeze(-1).log(), (similarity_val <1-threshold).float(), reduction='batchmean')
            
            self.loss = nn.BCELoss()(mlp_output.squeeze(-1), (similarity_val < 1 - threshold).float())
            self.mlp_accuracy_arr = ((1 - similarity_val - self.threshold) * (mlp_output.squeeze(-1) - self.threshold) > 0)
        else:
            self.loss = 0
            
        self.skip_ratio = torch.sum(~boolean_mask)
        return (output, )

# class ModifiedViTEncoder(ViTEncoder):
#     def __init__(self, config: ViTConfig,, threshold=0.05):
#         super().__init__(config) 
#         self.layer = nn.ModuleList([DHSLayer(config) for _ in range(config.num_hidden_layers)])

class ModifiedViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig, threshold=0.05):
        super().__init__(config)
        self.layer = nn.ModuleList([DHSLayer(config, threshold) for _ in range(config.num_hidden_layers)])


# class ModifiedViTEncoder(ViTEncoder):
#     def __init__(self, config: ViTConfig, threshold=0.05):
#         super().__init__(config)
#         self.layer = nn.ModuleList([ModifiedViTLayer(config, threshold) for _ in range(config.num_hidden_layers)])



import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class ModifiedViTModel(ViTModel):
    def __init__(self, config: ViTConfig, threshold=0.05):
        super().__init__(config)
        self.encoder = ModifiedViTEncoder(config,threshold)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)

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
pretrained_model_path = os.path.join('models', 'prad_model.pth')
loss_type = 'cosine'
num_epochs = 10
lr = 1e-3
data_path = 'data'
batch_size = 64
train_size = 10000
test_size = 1000

config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
threshold = 0.05
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
        for param in layer.mlp_layer.parameters():
            param.requires_grad = True

if loss_type in ['classification', 'both']:
    for param in modified_vit_model.classifier.parameters():
        param.requires_grad = True



train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)


print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True, progress=True)}')

train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs, loss_type=loss_type, lr=lr, progress=True)
torch.save(modified_vit_model.state_dict(), os.path.join('models', 'prad_model2.pth'))

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
from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
from torch import nn
import torch
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPooling
)

class ModifiedViTLayer(ViTLayer):
    def __init__(self, config, mlp_layer, sim_threshold=0.9, mlp_threshold=0.5):
        super().__init__(config)

        self.loss = 0

        self.sim_threshold = sim_threshold
        self.mlp_threshold = mlp_threshold

        self.mlp_layer = mlp_layer


    def forward(self, hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False, compute_cosine = False, output_mask=False):

        boolean_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool).to(hidden_states.device)


        # expanded_cls_token = hidden_states[:, 0: 1].repeat(1, 196, 1)
        # mlp_input = torch.cat((expanded_cls_token, hidden_states[:, 1:]), dim=-1)
        mlp_input = hidden_states[:, 1:]
        mlp_output = self.mlp_layer(mlp_input)
        boolean_mask = (mlp_output >= self.mlp_threshold).squeeze(-1)
        cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
        boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

        output = hidden_states.clone()
        batch_size = hidden_states.shape[0]
        for i in range(batch_size):
            output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)

        if self.training or compute_cosine:
            real_output = super().forward(hidden_states)[0][:, 1:]
            cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1) + 1) / 2
            # cos_similarity = torch.abs(F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1))
            euclidean_dist = torch.sum((real_output - hidden_states[:, 1:]) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
            dist_similarity = 1 / (1 + euclidean_dist)
            alpha = 0.5
            similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity

            # mlp_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(boolean_mask.device))
            mlp_criterion = nn.BCEWithLogitsLoss()
            self.loss = mlp_criterion(mlp_output.squeeze(-1), (similarity_val < self.sim_threshold).float())
            self.mlp_accuracy_arr = ((self.sim_threshold - similarity_val) * (mlp_output.squeeze(-1) - self.mlp_threshold) > 0)

            true_labels = (similarity_val < self.sim_threshold).int().cpu().flatten()
            pred_labels = (mlp_output.squeeze(-1) > self.mlp_threshold).int().cpu().flatten()
            self.mlp_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

        else:
            self.loss = 0
            
        if output_mask:
            return (output, boolean_mask)
        else:
            return (output, )

class ModifiedViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig, sim_threshold=0.9, mlp_threshold=0.5):
        super().__init__(config)
        
        self.hidden_size = config.hidden_size
        layer_sizes = [self.hidden_size, 64, 1]

        mlp_layers = []
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])
            else:
                mlp_layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Sigmoid()])                

        self.mlp_layer = nn.Sequential(*mlp_layers)

        self.layer = nn.ModuleList([ModifiedViTLayer(config, self.mlp_layer, sim_threshold, mlp_threshold) for _ in range(config.num_hidden_layers)])
    
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
        
        for param in self.encoder.mlp_layer.parameters():
            param.requires_grad = False
    
    def mlp_train(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.encoder.mlp_layer.parameters():
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
        for param in self.encoder.mlp_layer.parameters():
            param.requires_grad = True
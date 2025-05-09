import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoConfig
from transformers.models.vit.modeling_vit import ViTModel, ViTPooler, ViTLayer, ViTEncoder, ViTConfig, ViTForImageClassification
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm
from torch.utils.flop_counter import FlopCounterMode
from ptflops import FLOPS_BACKEND,get_model_complexity_info


import sys
from loguru import logger
from collections import defaultdict
from typing import Optional,Union, Tuple
import os
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_file_name = os.path.basename(__file__).split('.')[0]


# ------------------------------------------------------
with open(f'{current_file_name}.html', "w") as f:
    f.write("""<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Log Output</title>
            </head>
            <body>
                <!-- Log messages will be appended here -->
            """)
        
def html_sink(message):
    log_entry = f"<p>{message}</p>\n"
    with open(f'{current_file_name}.html', "a") as f:
        f.write(log_entry)
        f.flush()
        os.fsync(f.fileno())

logger.remove()
logger.add(html_sink, format="<b>{time}</b> {level}: {message}")

# ----------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: np.array(img))  # Converts PIL Image to a NumPy array
])


train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform = transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform = transform)

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

train_subset = Subset(train_dataset, range(10000))
test_subset = Subset(test_dataset, range(2000))

batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# ------------------------------------------------------
alpha = 0.3 # similarity
sim_threshold = 0.91
mlp_threshold_train = 0.5
mlp_threshold_test = 0.5

gamma = 2.0

def print_stats(similarity):
   stats = { "mean": torch.mean(similarity).item(), 
                "variance": torch.var(similarity, unbiased=False).item(),
                "std_dev": torch.std(similarity, unbiased=False).item(), 
                "min": torch.min(similarity).item(), 
                "max": torch.max(similarity).item(), 
                "sum": torch.sum(similarity).item(), 
                "median": torch.median(similarity).item()}
   print(stats)

track_mlp_loss = {}
visual_track = defaultdict(list)
visual_check_list = [0]
batch_id = 0

def plot_visual(loader):
    global visual_track
    loader_batches = list(loader)
    mlp_images = defaultdict(list)
    for mlp_id, records in visual_track.items():
        for batch_id, target_mask in records:
            images, _ = loader_batches[batch_id]
            images_np = images.cpu().numpy().copy()  # shape (batch_size, 224, 224, 3) 
            mask_reshaped = target_mask.reshape(-1, 14, 14)
            patch_size = 16
            for idx in range(len(images_np)):
                for i in range(14):
                    for j in range(14):
                        if mask_reshaped[idx, i, j] == 0:
                            images_np[idx, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = 0
            mlp_images[mlp_id].append(images_np)
            
    mlp_ids = list(mlp_images.keys())[:12]
    for batch_num in range(len(mlp_images[0])): # noof batches for each layer (is same)
        n = len(mlp_images[0][batch_num])
        fig, axes = plt.subplots(n, len(mlp_ids), figsize=(20, 3 * n))
        for i in range(n):
            for j, mlp_id in enumerate(mlp_ids):
                img = mlp_images[mlp_id][batch_num][i,:,:,:]  # shape: (batch_size, 224, 224, 3)
                axes[i, j].imshow(img.astype(np.uint8))
                axes[i, j].set_title(f"MLP {mlp_id} - Img {i}")
                axes[i, j].axis("off")
        plt.tight_layout()
        logger.info("saving visuals")
        plt.savefig(f"./outputs/visual_{batch_num}", dpi=512)
        plt.close()


class NeuralNet(nn.Module):
    id_counter = 0

    def __init__(self, arr, use_batch_norm=True):
        super().__init__()
        layers = []
        for i in range(len(arr) - 2):
            layers.append(nn.Linear(arr[i], arr[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(arr[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(arr[-2], arr[-1]))
        layers.append(nn.Sigmoid())

        self.size = len(arr)
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.identity = NeuralNet.id_counter
        NeuralNet.id_counter += 1

    def forward(self, x):
        result = self.model(x)
        if batch_id in visual_check_list:
            visual_track[self.identity].append([batch_id, result.clone().detach().cpu().numpy().reshape(-1, 196)])
        return result
    
    def focal_loss_binary(self, outputs, targets, alpha= 0.25, gamma=2.0):
        pt = torch.where(targets == 1, outputs, 1 - outputs)
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets.float() , reduction='none')  # shape: [batch_size, 1]
        focal_weight = alpha * ((1 - pt) ** gamma) 
        loss = focal_weight * bce_loss
        return loss.mean()

    def train_model(self, inputs, targets):
        global track_mlp_loss
        global visual_track
        global batch_id
        global gamma

        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        # loss = nn.BCELoss()(outputs,targets.float())
        loss = self.focal_loss_binary(outputs,targets)
        loss.backward()
        self.optimizer.step()


        predictions = (outputs > mlp_threshold_train).float()
        correct = (predictions == targets).float().sum()
        accuracy = correct / len(targets)

        # Class-specific accuracy
        class_0_mask = targets == 0
        class_1_mask = targets == 1
        class_0_correct = ((predictions == targets) * class_0_mask).sum()
        class_1_correct = ((predictions == targets) * class_1_mask).sum()
        class_0_total = class_0_mask.sum()
        class_1_total = class_1_mask.sum()

        class_0_acc = class_0_correct / class_0_total if class_0_total > 0 else torch.tensor(0.0)
        class_1_acc = class_1_correct / class_1_total if class_1_total > 0 else torch.tensor(0.0)

        curr_pos = class_1_total.item()

        if self.identity not in track_mlp_loss:
            track_mlp_loss[self.identity] = [
                len(inputs), accuracy.item(), curr_pos, class_0_acc.item(), class_1_acc.item()
            ]
        else:
            total_samples = track_mlp_loss[self.identity][0] + len(inputs)
            avg_accuracy = (
                track_mlp_loss[self.identity][0] * track_mlp_loss[self.identity][1] + correct
            ) / total_samples
            tot_pos = track_mlp_loss[self.identity][2] + curr_pos

            # Weighted average class accuracies
            prev_class_0_total = (
                track_mlp_loss[self.identity][0] - track_mlp_loss[self.identity][2]
            )
            prev_class_1_total = track_mlp_loss[self.identity][2]

            total_class_0 = prev_class_0_total + class_0_total
            total_class_1 = prev_class_1_total + class_1_total

            class_0_acc_avg = (
                prev_class_0_total * track_mlp_loss[self.identity][3] + class_0_correct
            ) / total_class_0 if total_class_0 > 0 else torch.tensor(0.0)
            class_1_acc_avg = (
                prev_class_1_total * track_mlp_loss[self.identity][4] + class_1_correct
            ) / total_class_1 if total_class_1 > 0 else torch.tensor(0.0)

            track_mlp_loss[self.identity] = [
                total_samples,
                avg_accuracy.item(),
                tot_pos,
                class_0_acc_avg.item(),
                class_1_acc_avg.item()
            ]

        return loss.item()


class ModifyLayer(ViTLayer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding_size = config.hidden_size
        self.mlp = NeuralNet([2 * self.embedding_size, 64 , 1])

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        original: Optional[torch.Tensor] = None,  # custom parameter
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        outputs = ()
        batch_size = hidden_states.size(0)

        if vanilla:
            temp_output = super().forward(hidden_states)
            hidden_states = temp_output[0]
            outputs = (hidden_states,) + outputs
            return outputs, original

        cls_tokens = hidden_states[:, 0, :].unsqueeze(1) #[batch, 1, 768]          
        all_tokens = hidden_states[:, 1:, :]        # [batch, 196, 768]
        cls_expanded = cls_tokens.expand(-1, all_tokens.size(1), -1)  # Now shape: [batch, 196, 768]
        mlp_inputs = torch.cat((cls_expanded, all_tokens), dim=-1) # shape: [batch, 196, 1536]
        mlp_inputs = mlp_inputs.reshape(-1,2*self.embedding_size) # shape: [batch * 196, 1536]
        attention_mask = None

        # ---------------------------------------------------------------
        if train_mlp_only:
            # 1) get the original outputs for next layer
            with torch.no_grad():
                original_output = super().forward(original)
                original = original_output[0]

            # 2) getting similarity score
            num_tokens = len(original[0])
            temp_original = original[:,1:,:].reshape(batch_size*(num_tokens - 1),-1)
            temp_hidden = hidden_states[:,1:,:].reshape(batch_size*(num_tokens-1),-1)
            cos_sim = F.cosine_similarity(temp_original,temp_hidden,-1)
            # euclid_dist = torch.sqrt(torch.sum((temp_original - temp_hidden)**2, dim=1))
            # dist_sim = 1 / (1 + euclid_dist)
            # similarity = alpha * cos_sim + (1 - alpha) * dist_sim
            similarity = cos_sim
            # print_stats(similarity)
            attention_mask = similarity < sim_threshold 
            #1->pass to transformers do not skip , 0 -> skip transformer shape: [batch_size*196,1]

            # 3) mlp donot pass cls
            mlp_loss = self.mlp.train_model(mlp_inputs.clone().detach(),attention_mask.unsqueeze(1).float().clone().detach()) 
            attention_mask = attention_mask.reshape(batch_size,-1)

            hidden_states = original
            outputs = (hidden_states,) + outputs
            return outputs, original
        
        # --------------------------------------------------------------------
        mlp_outputs = self.mlp(mlp_inputs) # shape [batch * 196, 1]
        mlp_outputs = mlp_outputs.reshape(batch_size,-1)  # shape [batch,196]
        attention_mask = mlp_outputs < mlp_threshold_test # 1 need to be used in transformer
        cls_att_mask = torch.zeros((attention_mask.size(0), attention_mask.size(1) + 1), dtype=bool, device=attention_mask.device)
        cls_att_mask[:, 0] = 1  
        cls_att_mask[:, 1:] = attention_mask 
        for i in range(attention_mask.size(0)):
            trimmed_input = hidden_states[i][cls_att_mask[i] == 1]
            layer_output = super().forward(trimmed_input.unsqueeze(0))
            hidden_states[i][cls_att_mask[i] == 1] = layer_output[0]
        outputs = (hidden_states,) + outputs
        return outputs, original


class ModifyEncoder(ViTEncoder):
    def __init__(self,config):
        super().__init__(config)
        self.layer = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.layer.add_module(f'{i}',ModifyLayer(config))

    def forward(self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        output_mask: bool = False,
        ) -> Union[tuple, BaseModelOutput]:

        # optional to necessary
        all_hidden_states = () if output_hidden_states else None
        all_boolean_mask = () if output_mask else None
        all_self_attentions = () if output_attentions else None

        original = hidden_states.clone().detach().to(device)  # make deep copy
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions)
            else:
                layer_outputs,original= layer_module(hidden_states, layer_head_mask, output_attentions,original)
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



# https://huggingface.co/docs/transformers/en/model_doc/vit#transformers.ViTModel (for more info on parameters)

class Modifymodel(ViTModel): # return Union[tuple, BaseModelOutputwithpooling]
    def __init__(self,config,add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        self.encoder = ModifyEncoder(config)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

    def forward( self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None, # bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*): Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_mask: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:

        # converting optinal type to normal type
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Prepare head mask if needed 1.0 in head_mask indicate we keep the head and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        # 1) add embeddings after processing
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding)
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

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ModifyForImageClassification(ViTForImageClassification):
    def __init__(self,config: ViTConfig,processor) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = Modifymodel(config,add_pooling_layer = False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.processor = processor

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None
        ) -> Union[Tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        # move labels to correct device to enable model parallelism
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
            
    def load_weights(self, pretrained_model_name_or_path: str):
        pretrained = ViTForImageClassification.from_pretrained(
            pretrained_model_name_or_path
        )
        missing = set(self.state_dict().keys()) - set(pretrained.state_dict().keys())
        self.load_state_dict(pretrained.state_dict(), strict=False)
        if missing:
            print("Parameters not loaded:")
            for p in missing:
                print(f" - {p}")
        else:
            print("All parameters loaded.")

    def params_mlp(self, freeze: bool = True):
        for layer in self.vit.encoder.layer:
            for param in layer.mlp.parameters():
                param.requires_grad = not freeze

    def train_model(self, train_loader):
        global train_mlp_only
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        loss_hist, acc_hist = [], []

        # Phase 1: MLP-only warmup
        train_mlp_only = True
        self.params_mlp(freeze=False)
        self._train_mlp_phase(train_loader,epochs = 3)

        # Phase 2: Full model training
        for i in range(3):
            train_mlp_only = True
            self.params_mlp(freeze=False)
            self._train_mlp_phase(train_loader,epochs = 1)

            train_mlp_only = False
            self.params_mlp(freeze=True)
            self._train_full_phase(train_loader, criterion, optimizer, loss_hist, acc_hist,epochs = 2)
        self._save_and_plot_metrics(loss_hist, acc_hist)

    def _train_mlp_phase(self, train_loader, epochs: int = 3):
        self.params_mlp(freeze=False)
        for epoch in range(epochs):
            for inputs, targets in tqdm(train_loader, desc=f"MLP Warmup {epoch+1}/{epochs}"):
                processed = self.processor(inputs, return_tensors="pt")
                pixel_values = processed.pixel_values.to(self.device)
                targets = targets.to(self.device)
                _ = self.forward(pixel_values=pixel_values)
            self._log_mlp_metrics()
            self._reset_mlp_tracking()

    def _train_full_phase(
        self,
        train_loader,
        criterion,
        optimizer,
        loss_hist,
        acc_hist,
        epochs
    ):
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            for inputs, targets in tqdm(train_loader, desc=f"Training {epoch+1}/{epochs}"):
                processed = self.processor(inputs, return_tensors="pt")
                pixel_values = processed.pixel_values.to(self.device)
                targets = targets.to(self.device)

                outputs = self.forward(pixel_values=pixel_values)
                loss = criterion(outputs.logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            avg_loss, accuracy = total_loss/len(train_loader), correct/total
            loss_hist.append(avg_loss)
            acc_hist.append(accuracy)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
            self._log_mlp_metrics()
            self._reset_mlp_tracking()

    def _log_mlp_metrics(self):
        if not track_mlp_loss:
            return
        print("MLP Tracking:")
        for mlp_id, stats in track_mlp_loss.items():
            print(f"MLP {mlp_id}: {stats}")

    def _reset_mlp_tracking(self):
        track_mlp_loss.clear()
        visual_track.clear()

    def _save_and_plot_metrics(self, loss_hist, acc_hist, folder: str = 'outputs'):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder, 'model_weights.pth'))

        # Plot metrics
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(loss_hist, label='Loss')
        plt.plot(acc_hist, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(folder, 'training_metrics.png'))
        plt.close()
        print(f"Metrics and model saved in '{folder}'")


    def load_saved_weights(self, path="outputs/model_weights.pth"):
        try:
            self.load_state_dict(torch.load(path, map_location=device))
            logger.info(f"Model weights loaded from: {path}")
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")

# --------------------------------------------------------------------------------------------------

def evaluate_model(model, test_loader, processor, device):
    correct = 0
    total = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for inp, tar in tqdm(test_loader, desc="Evaluating"):
            batch = processor(inp, return_tensors="pt")
            images = batch['pixel_values'].to(device)
            labels = tar.to(device)

            # Start timer
            start_time = time.time()

            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)

            # End timer
            elapsed = time.time() - start_time
            total_time += elapsed

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")

    return accuracy,total_time

def get_complexity(model, test_loader, processor, device):
    total_flops = 0
    total_time = 0.0
    model.eval()

    with torch.no_grad():
        for inp, tar in tqdm(test_loader, desc="Evaluating"):
            batch = processor(inp, return_tensors="pt")
            images = batch['pixel_values'].to(device)
            labels = tar.to(device)

            start_time = time.time()

            # Loop through each image individually because of ptflops structure
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
                    verbose=False,
                    backend=FLOPS_BACKEND.ATEN
                )
                total_flops += flops

            elapsed = time.time() - start_time
            total_time += elapsed

    gflops = total_flops / 1e9
    gflops_per_sec = gflops / total_time if total_time > 0 else 0

    print(f"\nTotal Inference Time: {total_time:.2f} seconds")
    print(f"Total FLOPs: {gflops:.2f} GFLOPs")
    print(f"Throughput: {gflops_per_sec:.2f} GFLOPS")

 


model_name = "Ahmed9275/Vit-Cifar100"
model = AutoModelForImageClassification.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)


newmodel = ModifyForImageClassification(config,processor)
newmodel.load_weights(model_name)
newmodel.to(device)

#baseline model
# train_mlp_only = False
# vanilla = True
# evaluate_model(newmodel, test_loader, processor, device)
# get_complexity(newmodel, test_loader,processor,device)

# #training
vanilla = False
newmodel.train_model(train_loader=train_loader)
newmodel.load_saved_weights()

train_mlp_only = False
evaluate_model(newmodel, test_loader, processor, device)
# get_complexity(newmodel, test_loader, processor,device)

# ------------------------------------------------------------
with open(f'{current_file_name}.html', "a") as f:
    f.write("""
            </body>
            </html>
            """)
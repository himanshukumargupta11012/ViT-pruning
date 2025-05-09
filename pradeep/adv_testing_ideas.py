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

# print("CUDA device:", os.getenv("CUDA_VISIBLE_DEVICES"), ' '.join(sys.argv))

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPooling
)


class ModifiedViTLayer(ViTLayer):
    def __init__(self, config, sim_threshold=0.9, mlp_threshold=0.5, mlp_needed=True):
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
        self.sim_threshold = sim_threshold
        self.mlp_threshold = mlp_threshold
        
        self.skiped=0

    def forward(self, hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            compute_cosine=False,
            output_mask=False,
            top_k: int = 150):  # Add top_k parameter
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, dim = hidden_states.shape
        cls_token = hidden_states[:, 0:1, :]  # [batch_size, 1, hidden_dim]
        patch_tokens = hidden_states[:, 1:, :]  # [batch_size, seq_len-1, hidden_dim]

        # Compute cosine similarity with CLS token
        cos_sim = F.cosine_similarity(patch_tokens, cls_token.expand_as(patch_tokens), dim=-1)  # [batch_size, seq_len-1]
        cos_sim = (cos_sim + 1) / 2  # normalize to [0, 1]

        # Get top-k indices per sample
        topk_values, topk_indices = torch.topk(cos_sim, top_k, dim=1)  # [batch_size, top_k]

        # Initialize mask: skip those in top-k, process rest
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        mask[:, 0] = True  # always skip CLS
        for i in range(batch_size):
            mask[i, topk_indices[i] + 1] = True  # +1 to offset CLS position

        # Process rest
        output = hidden_states.clone()
        for i in range(batch_size):
            idx_to_process = ~mask[i]
            if idx_to_process.sum() > 0:
                processed = super().forward(hidden_states[i][idx_to_process].unsqueeze(0))[0].squeeze(0)
                output[i][idx_to_process] = processed

        output_forced = output.clone()
        output_forced[~mask] = hidden_states[~mask]
        # print(torch.sum(mask).item())


        self.skiped = torch.sum(~mask).item()

        if output_mask:
            return output_forced, mask
        else:
            return (output_forced,)



    # def forward(self, hidden_states: torch.Tensor,
    #     head_mask: Optional[torch.Tensor] = None,
    #     output_attentions: bool = False, compute_cosine = False, output_mask=False):

    #     # boolean_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool).to(hidden_states.device)
    #     # if not self.mlp_needed:
    #     #     if output_mask:
    #     #         return (super().forward(hidden_states)[0], boolean_mask)
    #     #     else:
    #     #         return super().forward(hidden_states)
    #     # mlp_output = self.mlp_layer(hidden_states[:, 1:])

    #     # boolean_mask = (mlp_output >= self.mlp_threshold).squeeze(-1)
    #     # cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
    #     # boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

    #     # output_forced = hidden_states.clone()
    #     # batch_size = hidden_states.shape[0]
    #     # for i in range(batch_size):
    #     #     output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)
    #     # batch_size = hidden_states.shape[0]
    #     # for i in range(batch_size):
    #     # if(self.training==True):
    #     real_output = super().forward(hidden_states[:, 0:])[0]
    #     cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, :], dim=-1) + 1) / 2
    #         # cos_similarity = torch.abs(F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1))
    #     euclidean_dist = torch.sum((real_output - hidden_states[:, 0:]) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
    #     dist_similarity = 1 / (1 + euclidean_dist)
    #     alpha = 0.5
    #     similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 
    #     self.mlp_accuracy_arr = 1  
    #     mask = similarity_val > self.sim_threshold
    #     output = hidden_states.clone()

    #     for i in range(hidden_states.shape[0]):
    #         output[i][~mask[i]] = super().forward(hidden_states[i][~mask[i]].unsqueeze(0))[0].squeeze(0)
            
    #     output_forced = output.clone()
    #     output_forced[mask] = hidden_states[mask, 0:]
    #     self.skiped = torch.sum(mask).item()
    #     if output_mask:
    #         return (output_forced, mask)
    #     else:
    #         return (output_forced,)
        
        # else:
        #     real_output = super().forward(hidden_states[:, 0:])[0]
        #     cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, :], dim=-1) + 1) / 2
        #         # cos_similarity = torch.abs(F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1))
        #     euclidean_dist = torch.sum((real_output - hidden_states[:, 0:]) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
        #     dist_similarity = 1 / (1 + euclidean_dist)
        #     alpha = 1
        #     similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 

            # self.mlp_accuracy_arr = ((self.sim_threshold - similarity_val) * (mlp_output.squeeze(-1) - self.mlp_threshold) > 0)
            # self.mlp_accuracy_arr = 1
            # # print(similarity_val.shape)
            # # true_labels = (similarity_val < self.sim_threshold).int().cpu().flatten()

            # # pred_labels = (mlp_output.squeeze(-1) > self.mlp_threshold).int().cpu().flatten()
            # # self.mlp_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

            # # forcing the model to learn the similarity
            # output_forced = real_output.clone()  # Default to real_output
            
            # # Create a mask where similarity_val > sim_threshold
            # mask = similarity_val > self.sim_threshold
            # # print(mask.shape)

            # # Apply the mask to update the selected indices from hidden_states
            # output_forced[mask] = hidden_states[mask, 0:]

            # self.skiped = torch.sum(mask).item()




            # else:
                # self.loss = 0
                
            # self.skip_ratio = torch.sum(~boolean_mask).item()
            # if output_mask:
            #     return (output, boolean_mask)
            # else:
                # return (output, )
            # return (output_forced, )
    
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
            param.requires_grad =  True
        
        # for param in self.classifier.parameters():
        #     param.requires_grad = True

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


# def write_N_print(string):
    # print(string)
    # log_file.write(string + "\n")
    # log_file.flush()


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
            # torch.save(best_model, model_save_path)
            torch.save(best_model.state_dict(), model_save_path, _use_new_zipfile_serialization=False)


        print(f"")
        # write_N_print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
        # write_N_print(f"Test accuracy after {epoch + 1} epochs: {val_accuracy}")

    # torch.save(best_model, model_save_path)
    # write_N_print(f"Best accuracy: {best_val_accuracy}")


# Function to evaluate the model on the test data
def test(model, dataloader, full_testing=False):
    model.eval()
    total_correct = 0
    each_layer_skip = torch.zeros(len(model.encoder.layer))
    if full_testing:
        total_correct_mlp = torch.zeros(len(model.encoder.layer), 2, 2)

    data_size = len(dataloader.dataset)
    with torch.no_grad():             # Disable gradient computation for efficiency
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
                each_layer_skip[i] += vit_layer.skiped

                # if full_testing and hasattr(vit_layer, 'mlp_accuracy_arr'):
                #     total_correct_mlp[i] += vit_layer.mlp_confusion_matrix
                    
            total_correct += torch.sum(predicted_class_indices == labels).item()

    print(f"Skip ratio: {each_layer_skip / data_size} and {torch.sum(each_layer_skip) / data_size / len(model.encoder.layer)}")

    if full_testing:
        mlp_accuracy = (torch.sum(total_correct_mlp[:, 1, 1]) + torch.sum(total_correct_mlp[:, 0, 0])) / (torch.sum(total_correct_mlp) + 1e-16)
        
        mlp_accuracy_arr = torch.round(total_correct_mlp / (total_correct_mlp.sum(dim=[-2, -1], keepdim=True) + 1e-16), decimals=4)
        print(f"Correct prediction ratio:\n{mlp_accuracy_arr.tolist()}\nand {mlp_accuracy}")

        return total_correct / data_size, mlp_accuracy
    
    return total_correct / data_size

# Define the training parameters
pretrained_model_path = False#os.path.join('models', '2025-01-20_01-28-30_lr-0.0001_st-0.9_mt-0.5_bs-64_trs-50000_tes-10000_nw-16.pth')
loss_type = 'classification'
num_epochs = 3
lr = 1e-4
data_path = '../himanshu/data'
train_batch_size = 64
test_batch_size = 128
train_size = 10_000
test_size = 10_000
sim_threshold = 0.99
mlp_threshold = 0.5
num_workers = 16

pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"

save_path = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr-{lr}_st-{sim_threshold}_mt-{mlp_threshold}_bs-{train_batch_size}_trs-{train_size}_tes-{test_size}_nw-{num_workers}_type-{loss_type}"

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
    # print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True)}')
    # os.makedirs('logs', exist_ok=True)
    # os.makedirs('models', exist_ok=True)
    # log_file = open(f"logs/{save_path}.txt", "w")

    modified_vit_model.vit_train()
    # train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs, loss_type=loss_type, lr=lr, save_path=save_path)

    modified_vit_model.eval()
    # train(modified_vit_model, train_loader, test_loader, num_epochs=5, loss_type="classification", lr=lr, save_path=save_path)
    val_accuracy, mlp_val_accuracy = test(modified_vit_model, test_loader, full_testing=True)
    # # calculating flops
    # input_size = (3, 224, 224)
    # flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
    # flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
    # print(flops, flops_ori)
    print(val_accuracy, mlp_val_accuracy)
    # log_file.close()

"""# pretrained model accuracy 89.85"""


# # model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)

# import os, sys
# from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
# from torch import nn
# import torch
# from typing import Optional, Tuple, Union
# import torch.nn.functional as F
# from datetime import datetime
# from sklearn.metrics import confusion_matrix


# seed_val = 42
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# # print("CUDA device:", os.getenv("CUDA_VISIBLE_DEVICES"), ' '.join(sys.argv))

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# print(device)
# from transformers.modeling_outputs import (
#     BaseModelOutput, BaseModelOutputWithPooling
# )


# class ModifiedViTLayer(ViTLayer):
#     def __init__(self, config, sim_threshold=0.9, mlp_threshold=0.5, mlp_needed=True):
#         super().__init__(config)
#         self.mlp_needed = mlp_needed

#         self.loss = 0
#         self.skip_ratio = 0
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
#         self.skiped=0

#         def forward(self, hidden_states: torch.Tensor,
#             head_mask: Optional[torch.Tensor] = None,
#             output_attentions: bool = False, compute_cosine=False, output_mask=False):

#             real_output = super().forward(hidden_states[:, 0:])[0]
#             cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, :], dim=-1) + 1) / 2
#             euclidean_dist = torch.sum((real_output - hidden_states[:, 0:]) ** 2, dim=-1) / torch.sum(real_output ** 2, dim=-1)
#             dist_similarity = 1 / (1 + euclidean_dist)
#             alpha = 0.5
#             similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 

#             self.mlp_accuracy_arr = 1 

#             # Create a mask where similarity_val > sim_threshold
#             mask = similarity_val > self.sim_threshold

#             # Reshape real_output into a 2D spatial grid (assuming a perfect square)
#             batch_size, num_patches, dim = real_output.shape
#             grid_size = int(num_patches ** 0.5)  # Assuming num_patches is a perfect square

#             output_forced = real_output.clone()  # Default to real_output
#             real_output_reshaped = real_output[:, 1:].view(batch_size, grid_size, grid_size, dim).permute(0, 3, 1, 2)  
#             mask_reshaped = mask[:, 1:].view(batch_size, grid_size, grid_size).float().unsqueeze(1)  # Add channel dimension

#             # Define a 3x3 convolutional kernel for averaging neighbors
#             kernel = torch.tensor([[1, 1, 1], 
#                                 [1, 0, 1], 
#                                 [1, 1, 1]], dtype=torch.float32, device=real_output.device).unsqueeze(0).unsqueeze(0)

#             # Convolve mask with the kernel to count valid neighbors
#             neighbor_count = F.conv2d(mask_reshaped, kernel, padding=1)

#             # Convolve real_output to sum neighbor values
#             real_output_summed = F.conv2d(real_output_reshaped, kernel.expand(dim, 1, 3, 3), padding=1, groups=dim)

#             # Compute neighbor averages (avoid division by zero)
#             neighbor_avg = real_output_summed / (neighbor_count + 1e-6)

#             # Reshape back to (batch, num_patches - 1, dim)
#             neighbor_avg = neighbor_avg.permute(0, 2, 3, 1).view(batch_size, num_patches - 1, dim)

#             # Apply updates only at masked locations
#             output_forced[:, 1:][mask[:, 1:]] = neighbor_avg[mask[:, 1:]]  # Only update masked values
#             output_forced[:, 0] = real_output[:, 0]  # Keep CLS token unchanged

#             self.skiped = torch.sum(mask).item()
#             return (output_forced, )

        
    
#         # forcing the model to learn the similarity
#         # output_forced = real_output.clone()  # Default to real_output
        
#         # # Create a mask where similarity_val > sim_threshold
#         # mask = similarity_val > self.sim_threshold
#         # # Apply the mask to update the selected indices from hidden_states
#         # output_forced[mask] = hidden_states[mask, 0:]

#         # self.skiped = torch.sum(mask).item()
#         # return (output_forced, )
    
# class ModifiedViTEncoder(ViTEncoder):
#     def __init__(self, config: ViTConfig, sim_threshold=0.9, mlp_threshold=0.5):
#         super().__init__(config)
#         mlp_needed_arr = torch.ones(config.num_hidden_layers, dtype=torch.bool)
#         # mlp_needed_arr[8] = False
#         # indices_to_change = [6, 7, 8]
#         # mlp_needed_arr[indices_to_change] = True

#         self.layer = nn.ModuleList([ModifiedViTLayer(config, sim_threshold, mlp_threshold, mlp_needed_arr[_]) for _ in range(config.num_hidden_layers)])
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#         compute_cosine: bool = False,
#         output_mask: bool = False
#     ) -> Union[tuple, BaseModelOutput]:
#         all_hidden_states = () if output_hidden_states else None
#         all_boolean_mask = () if output_mask else None
#         all_self_attentions = () if output_attentions else None

#         for i, layer_module in enumerate(self.layer):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_head_mask = head_mask[i] if head_mask is not None else None

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     layer_module.__call__,
#                     hidden_states,
#                     layer_head_mask,
#                     output_attentions,
#                 )
#             else:
#                 layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, compute_cosine=compute_cosine, output_mask=output_mask)

#             hidden_states = layer_outputs[0]


#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)
#             if output_mask:
#                 all_boolean_mask = all_boolean_mask + (layer_outputs[1],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)
#         if not return_dict:
#             return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         ), all_boolean_mask
    
# class ModifiedViTModel(ViTModel):
#     def __init__(self, config: ViTConfig, sim_threshold=0.9, mlp_threshold=0.5):
#         super().__init__(config)
#         self.encoder = ModifiedViTEncoder(config, sim_threshold, mlp_threshold)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         bool_masked_pos: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         compute_cosine = False,
#         output_mask: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPooling]:

#         r"""
#         bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
#             Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if pixel_values is None:
#             raise ValueError("You have to specify pixel_values")

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
#         expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
#         if pixel_values.dtype != expected_dtype:
#             pixel_values = pixel_values.to(expected_dtype)

#         embedding_output = self.embeddings(
#             pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
#         )

#         encoder_outputs, boolean_masks = self.encoder(
#             embedding_output,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             compute_cosine=compute_cosine,
#             output_mask=output_mask
#         )
#         sequence_output = encoder_outputs[0]
#         sequence_output = self.layernorm(sequence_output)
#         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         if not return_dict:
#             head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
#             return head_outputs + encoder_outputs[1:]

#         outputs = BaseModelOutputWithPooling(
#             last_hidden_state=sequence_output,
#             pooler_output=pooled_output,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         )
#         logits = self.classifier(outputs['last_hidden_state'][:, 0])
#         output = lambda: None
#         setattr(output, 'logits', logits)
#         setattr(output, 'boolean_masks', boolean_masks)
        
#         return output

#     def vit_mlp_train(self):
#         for param in self.parameters():
#             param.requires_grad = True
    
#     def vit_train(self):
#         for param in self.parameters():
#             param.requires_grad = True

#         for layer in self.encoder.layer:
#             if not hasattr(layer, 'mlp_layer'):
#                 continue
#             for param in layer.mlp_layer.parameters():
#                 param.requires_grad = False
    
#     def mlp_train(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         for layer in self.encoder.layer:
#             if not hasattr(layer, 'mlp_layer'):
#                 continue
#             for param in layer.mlp_layer.parameters():
#                 param.requires_grad = True


#     def classifier_train(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         for param in self.classifier.parameters():
#             param.requires_grad = True
    
#     def classifier_mlp_train(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         for param in self.classifier.parameters():
#             param.requires_grad = True
#         for layer in self.encoder.layer:
#             if not hasattr(layer, 'mlp_layer'):
#                 continue
#             for param in layer.mlp_layer.parameters():
#                 param.requires_grad = True

# from torch.utils.data import Dataset, DataLoader, Subset
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# import torch
# import torchvision
# from tqdm import tqdm
# import os
# import torch.optim as optim
# from ptflops import get_model_complexity_info


# # Custom dataset class for CIFAR-100 that integrates with AutoImageProcessor for processing
# class CIFAR100Dataset(Dataset):
#     def __init__(self, root, train=True, size=None, processor=None):
#         self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True)           # Load CIFAR-100 dataset from torchvision
#         if size is not None:
#             self.dataset = Subset(self.dataset, torch.arange(size))                                  # Optionally, load only count number of samples
        
#         self.processor = processor                                                                    # Store the processor (used to preprocess images before passing to the model)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         image, label = self.dataset[idx] 
        
#         # Preprocess the image and convert it to tensor format suitable for the model
#         inputs = self.processor(images=image, return_tensors="pt")
#         # Return the processed pixel values and label
#         return inputs['pixel_values'].squeeze(0), label        


# # def write_N_print(string):
#     # print(string)
#     # log_file.write(string + "\n")
#     # log_file.flush()


# # Training function to train the model over multiple epochs
# def train(model, train_loader, test_loader, save_path, num_epochs=10, loss_type='both', lr=1e-4):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     cosine_loss_ratio = 1.5

#     best_val_accuracy = 0.0
#     best_model = None

    
#     model_save_path = f"models/{save_path}.pth"
    

#     optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
#     temp_train_loader = train_loader
#     for epoch in range(num_epochs):
#         model.train()  # Re-enable training mode at the start of each epoch
#         running_loss = 0.0
#         # total_correct_mlp = torch.zeros(len(model.encoder.layer))
#         # total_mlp_count = torch.zeros(len(model.encoder.layer))

#         tqdm_update_step = len(train_loader) / 100
#         train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

#         for batch_idx, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)                

#             logits = model(inputs).logits

#             if loss_type in ['classification', 'both']:
#                 classification_loss = criterion(logits, labels)

#             # Accumulate cosine similarity losses from each DHSLayer
#             if loss_type in ['cosine', 'both']:
#                 cosine_loss = 0.0
#                 for i, vit_layer in enumerate(model.encoder.layer):
#                     # total_correct_mlp[i] += (vit_layer.mlp_accuracy_arr).sum().item()
#                     # total_mlp_count[i] += vit_layer.mlp_accuracy_arr.numel()
#                     cosine_loss += vit_layer.loss
            
#             # Calculate the total loss as a weighted sum of classification and cosine similarity losses
#             if loss_type == 'classification':
#                 total_loss = classification_loss
#             elif loss_type == 'cosine':
#                 total_loss = cosine_loss
#             elif loss_type == 'both':
#                 total_loss = classification_loss + cosine_loss_ratio * cosine_loss
            
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()

#             # Accumulate running loss for this batch
#             running_loss += total_loss.item()

#             # Update the progress bar with current batch loss
#             if (batch_idx + 1) % tqdm_update_step == 0:
#                 train_loader.update(tqdm_update_step)
#             # train_loader.set_postfix({'batch_loss': total_loss.item()})

#         val_accuracy, mlp_val_accuracy = test(model, test_loader, full_testing=True)
        
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             best_model = model.state_dict()
#             torch.save(best_model, model_save_path)

#         print(f"")
#         # write_N_print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
#         # write_N_print(f"Test accuracy after {epoch + 1} epochs: {val_accuracy}")

#     # torch.save(best_model, model_save_path)
#     # write_N_print(f"Best accuracy: {best_val_accuracy}")


# # Function to evaluate the model on the test data
# def test(model, dataloader, full_testing=False):
#     model.eval()
#     total_correct = 0
#     each_layer_skip = torch.zeros(len(model.encoder.layer))
#     if full_testing:
#         total_correct_mlp = torch.zeros(len(model.encoder.layer), 2, 2)

#     data_size = len(dataloader.dataset)
#     with torch.no_grad():             # Disable gradient computation for efficiency
#         dataloader = tqdm(dataloader, desc=f"Testing", leave=False)
#         for (inputs, labels) in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             if full_testing:
#                 outputs = model(inputs, compute_cosine=True)
#             else:
#                 outputs = model(inputs)
#             logits = outputs.logits

#             predicted_class_indices = logits.argmax(dim=-1)    # Get the class index with the highest logit score

#             # Track the skip ratios from each layer
#             for i, vit_layer in enumerate(model.encoder.layer):
#                 each_layer_skip[i] += vit_layer.skiped

#                 # if full_testing and hasattr(vit_layer, 'mlp_accuracy_arr'):
#                 #     total_correct_mlp[i] += vit_layer.mlp_confusion_matrix
                    
#             total_correct += torch.sum(predicted_class_indices == labels).item()

#     print(f"Skip ratio: {each_layer_skip / data_size} and {torch.sum(each_layer_skip) / data_size / len(model.encoder.layer)}")

#     if full_testing:
#         mlp_accuracy = (torch.sum(total_correct_mlp[:, 1, 1]) + torch.sum(total_correct_mlp[:, 0, 0])) / (torch.sum(total_correct_mlp) + 1e-16)
        
#         mlp_accuracy_arr = torch.round(total_correct_mlp / (total_correct_mlp.sum(dim=[-2, -1], keepdim=True) + 1e-16), decimals=4)
#         print(f"Correct prediction ratio:\n{mlp_accuracy_arr.tolist()}\nand {mlp_accuracy}")

#         return total_correct / data_size, mlp_accuracy
    
#     return total_correct / data_size

# # Define the training parameters
# pretrained_model_path = False#os.path.join('models', '2025-01-20_01-28-30_lr-0.0001_st-0.9_mt-0.5_bs-64_trs-50000_tes-10000_nw-16.pth')
# loss_type = 'cosine'
# num_epochs = 2
# lr = 1e-4
# data_path = '../himanshu/data'
# train_batch_size = 64
# test_batch_size = 128
# train_size = 10_000
# test_size = 10_000
# sim_threshold = 0.95
# mlp_threshold = 0.5
# num_workers = 16

# pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"

# save_path = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr-{lr}_st-{sim_threshold}_mt-{mlp_threshold}_bs-{train_batch_size}_trs-{train_size}_tes-{test_size}_nw-{num_workers}_type-{loss_type}"

# cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)   # Initialize the image processor for CIFAR-100 data pre-processing
# pretrained_cifar_model = AutoModelForImageClassification.from_pretrained(pretrained_cifar_model_name)
# config = ViTConfig.from_pretrained(pretrained_cifar_model_name)
# config.num_labels = len(config.id2label)
# modified_vit_model = ModifiedViTModel(config, sim_threshold, mlp_threshold)

# # -----------   Copying Weights    ------------  #
# if not pretrained_model_path:
#     new_state_dict = {}
#     for key in pretrained_cifar_model.state_dict().keys():
#         new_key = key.replace('vit.', '')
#         new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

#     modified_vit_model.load_state_dict(new_state_dict, strict=False)
# else:
#     modified_vit_model.load_state_dict(torch.load(pretrained_model_path))

# # modified_vit_model = nn.DataParallel(modified_vit_model)
# modified_vit_model.to(device)
# # modified_vit_model = modified_vit_model.module


# if loss_type == "cosine":
#     modified_vit_model.mlp_train()
# elif loss_type == "classification":
#     modified_vit_model.vit_train()
# elif loss_type == "both":
#     modified_vit_model.classifier_mlp_train()


# # train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
# test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
# # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True)


# if __name__ == '__main__':
#     # print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True)}')
#     # os.makedirs('logs', exist_ok=True)
#     # os.makedirs('models', exist_ok=True)
#     # log_file = open(f"logs/{save_path}.txt", "w")

#     # train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs, loss_type=loss_type, lr=lr, save_path=save_path)
    
#     # modified_vit_model.vit_train()

#     modified_vit_model.eval()
#     # train(modified_vit_model, train_loader, test_loader, num_epochs=5, loss_type="classification", lr=lr, save_path=save_path)
#     val_accuracy, mlp_val_accuracy = test(modified_vit_model, test_loader, full_testing=True)
#     # # calculating flops
#     # input_size = (3, 224, 224)
#     # flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
#     # flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
#     # print(flops, flops_ori)
#     print(val_accuracy, mlp_val_accuracy)
#     # log_file.close()

# """# pretrained model accuracy 89.85"""






# # model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)

# import os, sys
# from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEncoder, ViTConfig
# from torch import nn
# import torch
# from typing import Optional, Tuple, Union
# import torch.nn.functional as F
# from datetime import datetime
# from sklearn.metrics import confusion_matrix


# seed_val = 42
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# # print("CUDA device:", os.getenv("CUDA_VISIBLE_DEVICES"), ' '.join(sys.argv))

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
# from transformers.modeling_outputs import (
#     BaseModelOutput, BaseModelOutputWithPooling
# )


# class ModifiedViTLayer(ViTLayer):
#     def __init__(self, config, sim_threshold=0.9, mlp_threshold=0.5, mlp_needed=True):
#         super().__init__(config)
#         self.mlp_needed = mlp_needed

#         self.loss = 0
#         self.skip_ratio = 0
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
        
#         self.skiped=0


#     def forward(self, hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False, compute_cosine = False, output_mask=False):

#         # boolean_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool).to(hidden_states.device)
#         # if not self.mlp_needed:
#         #     if output_mask:
#         #         return (super().forward(hidden_states)[0], boolean_mask)
#         #     else:
#         #         return super().forward(hidden_states)
#         # mlp_output = self.mlp_layer(hidden_states[:, 1:])

#         # boolean_mask = (mlp_output >= self.mlp_threshold).squeeze(-1)
#         # cls_col = torch.ones((hidden_states.shape[0], 1), dtype=torch.bool).to(boolean_mask.device)
#         # boolean_mask = torch.cat((cls_col, boolean_mask), dim=1)

#         # output_forced = hidden_states.clone()
#         # batch_size = hidden_states.shape[0]
#         # for i in range(batch_size):
#         #     output[i][boolean_mask[i]] = super().forward(hidden_states[i][boolean_mask[i]].unsqueeze(0))[0].squeeze(0)
#         # batch_size = hidden_states.shape[0]
#         # for i in range(batch_size):
            
#         real_output = super().forward(hidden_states[:, 0:])[0]
#         cos_similarity = (F.cosine_similarity(real_output, hidden_states[:, :], dim=-1) + 1) / 2
#             # cos_similarity = torch.abs(F.cosine_similarity(real_output, hidden_states[:, 1:], dim=-1))
#         euclidean_dist = torch.sum((real_output - hidden_states[:, 0:]) ** 2, dim=-1) / torch.sum(real_output**2, dim=-1)
#         dist_similarity = 1 / (1 + euclidean_dist)
#         alpha = 1
#         similarity_val = alpha * cos_similarity + (1 - alpha) * dist_similarity 

#         # self.mlp_accuracy_arr = ((self.sim_threshold - similarity_val) * (mlp_output.squeeze(-1) - self.mlp_threshold) > 0)
#         self.mlp_accuracy_arr = 1
#         # print(similarity_val.shape)
#         # true_labels = (similarity_val < self.sim_threshold).int().cpu().flatten()

#         # pred_labels = (mlp_output.squeeze(-1) > self.mlp_threshold).int().cpu().flatten()
#         # self.mlp_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

#         # forcing the model to learn the similarity
#         output_forced = real_output.clone()  # Default to real_output
        
#         # Create a mask where similarity_val > sim_threshold
#         mask = similarity_val > self.sim_threshold
#         # print(mask.shape)

#         # Apply the mask to update the selected indices from hidden_states
#         output_forced[mask] = hidden_states[mask, 0:]

#         self.skiped = torch.sum(mask).item()
#         # else:
#         #     self.loss = 0
            
#         # self.skip_ratio = torch.sum(~boolean_mask).item()
#         # if output_mask:
#         #     return (output, boolean_mask)
#         # else:
#             # return (output, )
#         return (output_forced, )
    
# class ModifiedViTEncoder(ViTEncoder):
#     def __init__(self, config: ViTConfig, sim_threshold=0.9, mlp_threshold=0.5):
#         super().__init__(config)
#         mlp_needed_arr = torch.ones(config.num_hidden_layers, dtype=torch.bool)
#         # mlp_needed_arr[8] = False
#         # indices_to_change = [6, 7, 8]
#         # mlp_needed_arr[indices_to_change] = True

#         self.layer = nn.ModuleList([ModifiedViTLayer(config, sim_threshold, mlp_threshold, mlp_needed_arr[_]) for _ in range(config.num_hidden_layers)])
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#         compute_cosine: bool = False,
#         output_mask: bool = False
#     ) -> Union[tuple, BaseModelOutput]:
#         all_hidden_states = () if output_hidden_states else None
#         all_boolean_mask = () if output_mask else None
#         all_self_attentions = () if output_attentions else None

#         for i, layer_module in enumerate(self.layer):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_head_mask = head_mask[i] if head_mask is not None else None

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     layer_module.__call__,
#                     hidden_states,
#                     layer_head_mask,
#                     output_attentions,
#                 )
#             else:
#                 layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, compute_cosine=compute_cosine, output_mask=output_mask)

#             hidden_states = layer_outputs[0]


#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)
#             if output_mask:
#                 all_boolean_mask = all_boolean_mask + (layer_outputs[1],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)
#         if not return_dict:
#             return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         ), all_boolean_mask
    
# class ModifiedViTModel(ViTModel):
#     def __init__(self, config: ViTConfig, sim_threshold=0.9, mlp_threshold=0.5):
#         super().__init__(config)
#         self.encoder = ModifiedViTEncoder(config, sim_threshold, mlp_threshold)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         bool_masked_pos: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         compute_cosine = False,
#         output_mask: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPooling]:

#         r"""
#         bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
#             Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if pixel_values is None:
#             raise ValueError("You have to specify pixel_values")

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
#         expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
#         if pixel_values.dtype != expected_dtype:
#             pixel_values = pixel_values.to(expected_dtype)

#         embedding_output = self.embeddings(
#             pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
#         )

#         encoder_outputs, boolean_masks = self.encoder(
#             embedding_output,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             compute_cosine=compute_cosine,
#             output_mask=output_mask
#         )
#         sequence_output = encoder_outputs[0]
#         sequence_output = self.layernorm(sequence_output)
#         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         if not return_dict:
#             head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
#             return head_outputs + encoder_outputs[1:]

#         outputs = BaseModelOutputWithPooling(
#             last_hidden_state=sequence_output,
#             pooler_output=pooled_output,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         )
#         logits = self.classifier(outputs['last_hidden_state'][:, 0])
#         output = lambda: None
#         setattr(output, 'logits', logits)
#         setattr(output, 'boolean_masks', boolean_masks)
        
#         return output

#     def vit_mlp_train(self):
#         for param in self.parameters():
#             param.requires_grad = True
    
#     def vit_train(self):
#         for param in self.parameters():
#             param.requires_grad = True

#         for layer in self.encoder.layer:
#             if not hasattr(layer, 'mlp_layer'):
#                 continue
#             for param in layer.mlp_layer.parameters():
#                 param.requires_grad = False
    
#     def mlp_train(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         for layer in self.encoder.layer:
#             if not hasattr(layer, 'mlp_layer'):
#                 continue
#             for param in layer.mlp_layer.parameters():
#                 param.requires_grad = True


#     def classifier_train(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         for param in self.classifier.parameters():
#             param.requires_grad = True
    
#     def classifier_mlp_train(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         for param in self.classifier.parameters():
#             param.requires_grad = True
#         for layer in self.encoder.layer:
#             if not hasattr(layer, 'mlp_layer'):
#                 continue
#             for param in layer.mlp_layer.parameters():
#                 param.requires_grad = True

# from torch.utils.data import Dataset, DataLoader, Subset
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# import torch
# import torchvision
# from tqdm import tqdm
# import os
# import torch.optim as optim
# from ptflops import get_model_complexity_info


# # Custom dataset class for CIFAR-100 that integrates with AutoImageProcessor for processing
# class CIFAR100Dataset(Dataset):
#     def __init__(self, root, train=True, size=None, processor=None):
#         self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True)           # Load CIFAR-100 dataset from torchvision
#         if size is not None:
#             self.dataset = Subset(self.dataset, torch.arange(size))                                  # Optionally, load only count number of samples
        
#         self.processor = processor                                                                    # Store the processor (used to preprocess images before passing to the model)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         image, label = self.dataset[idx] 
        
#         # Preprocess the image and convert it to tensor format suitable for the model
#         inputs = self.processor(images=image, return_tensors="pt")
#         # Return the processed pixel values and label
#         return inputs['pixel_values'].squeeze(0), label        


# # def write_N_print(string):
#     # print(string)
#     # log_file.write(string + "\n")
#     # log_file.flush()


# # Training function to train the model over multiple epochs
# def train(model, train_loader, test_loader, save_path, num_epochs=10, loss_type='both', lr=1e-4):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     cosine_loss_ratio = 1.5

#     best_val_accuracy = 0.0
#     best_model = None

    
#     model_save_path = f"models/{save_path}.pth"
    

#     optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
#     temp_train_loader = train_loader
#     for epoch in range(num_epochs):
#         model.train()  # Re-enable training mode at the start of each epoch
#         running_loss = 0.0
#         # total_correct_mlp = torch.zeros(len(model.encoder.layer))
#         # total_mlp_count = torch.zeros(len(model.encoder.layer))

#         tqdm_update_step = len(train_loader) / 100
#         train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

#         for batch_idx, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)                

#             logits = model(inputs).logits

#             if loss_type in ['classification', 'both']:
#                 classification_loss = criterion(logits, labels)

#             # Accumulate cosine similarity losses from each DHSLayer
#             if loss_type in ['cosine', 'both']:
#                 cosine_loss = 0.0
#                 for i, vit_layer in enumerate(model.encoder.layer):
#                     # total_correct_mlp[i] += (vit_layer.mlp_accuracy_arr).sum().item()
#                     # total_mlp_count[i] += vit_layer.mlp_accuracy_arr.numel()
#                     cosine_loss += vit_layer.loss
            
#             # Calculate the total loss as a weighted sum of classification and cosine similarity losses
#             if loss_type == 'classification':
#                 total_loss = classification_loss
#             elif loss_type == 'cosine':
#                 total_loss = cosine_loss
#             elif loss_type == 'both':
#                 total_loss = classification_loss + cosine_loss_ratio * cosine_loss
            
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()

#             # Accumulate running loss for this batch
#             running_loss += total_loss.item()

#             # Update the progress bar with current batch loss
#             if (batch_idx + 1) % tqdm_update_step == 0:
#                 train_loader.update(tqdm_update_step)
#             # train_loader.set_postfix({'batch_loss': total_loss.item()})

#         val_accuracy, mlp_val_accuracy = test(model, test_loader, full_testing=True)
        
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             best_model = model.state_dict()
#             torch.save(best_model, model_save_path)

#         print(f"")
#         # write_N_print(f"Average loss for epoch {epoch + 1}: {running_loss / len(train_loader)}")
#         # write_N_print(f"Test accuracy after {epoch + 1} epochs: {val_accuracy}")

#     # torch.save(best_model, model_save_path)
#     # write_N_print(f"Best accuracy: {best_val_accuracy}")


# # Function to evaluate the model on the test data
# def test(model, dataloader, full_testing=False):
#     model.eval()
#     total_correct = 0
#     each_layer_skip = torch.zeros(len(model.encoder.layer))
#     if full_testing:
#         total_correct_mlp = torch.zeros(len(model.encoder.layer), 2, 2)

#     data_size = len(dataloader.dataset)
#     with torch.no_grad():             # Disable gradient computation for efficiency
#         dataloader = tqdm(dataloader, desc=f"Testing", leave=False)
#         for (inputs, labels) in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             if full_testing:
#                 outputs = model(inputs, compute_cosine=True)
#             else:
#                 outputs = model(inputs)
#             logits = outputs.logits

#             predicted_class_indices = logits.argmax(dim=-1)    # Get the class index with the highest logit score

#             # Track the skip ratios from each layer
#             for i, vit_layer in enumerate(model.encoder.layer):
#                 each_layer_skip[i] += vit_layer.skiped

#                 # if full_testing and hasattr(vit_layer, 'mlp_accuracy_arr'):
#                 #     total_correct_mlp[i] += vit_layer.mlp_confusion_matrix
                    
#             total_correct += torch.sum(predicted_class_indices == labels).item()

#     print(f"Skip ratio: {each_layer_skip / data_size} and {torch.sum(each_layer_skip) / data_size / len(model.encoder.layer)}")

#     if full_testing:
#         mlp_accuracy = (torch.sum(total_correct_mlp[:, 1, 1]) + torch.sum(total_correct_mlp[:, 0, 0])) / (torch.sum(total_correct_mlp) + 1e-16)
        
#         mlp_accuracy_arr = torch.round(total_correct_mlp / (total_correct_mlp.sum(dim=[-2, -1], keepdim=True) + 1e-16), decimals=4)
#         print(f"Correct prediction ratio:\n{mlp_accuracy_arr.tolist()}\nand {mlp_accuracy}")

#         return total_correct / data_size, mlp_accuracy
    
#     return total_correct / data_size

# # Define the training parameters
# pretrained_model_path = False#os.path.join('models', '2025-01-20_01-28-30_lr-0.0001_st-0.9_mt-0.5_bs-64_trs-50000_tes-10000_nw-16.pth')
# loss_type = 'cosine'
# num_epochs = 2
# lr = 1e-4
# data_path = '../himanshu/data'
# train_batch_size = 64
# test_batch_size = 128
# train_size = 10_000
# test_size = 10_000
# sim_threshold = 0.94
# mlp_threshold = 0.5
# num_workers = 16

# pretrained_cifar_model_name = "Ahmed9275/Vit-Cifar100"

# save_path = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr-{lr}_st-{sim_threshold}_mt-{mlp_threshold}_bs-{train_batch_size}_trs-{train_size}_tes-{test_size}_nw-{num_workers}_type-{loss_type}"

# cifar_processor = AutoImageProcessor.from_pretrained(pretrained_cifar_model_name)   # Initialize the image processor for CIFAR-100 data pre-processing
# pretrained_cifar_model = AutoModelForImageClassification.from_pretrained(pretrained_cifar_model_name)
# config = ViTConfig.from_pretrained(pretrained_cifar_model_name)
# config.num_labels = len(config.id2label)
# modified_vit_model = ModifiedViTModel(config, sim_threshold, mlp_threshold)

# # -----------   Copying Weights    ------------  #
# if not pretrained_model_path:
#     new_state_dict = {}
#     for key in pretrained_cifar_model.state_dict().keys():
#         new_key = key.replace('vit.', '')
#         new_state_dict[new_key] = pretrained_cifar_model.state_dict()[key]

#     modified_vit_model.load_state_dict(new_state_dict, strict=False)
# else:
#     modified_vit_model.load_state_dict(torch.load(pretrained_model_path))

# # modified_vit_model = nn.DataParallel(modified_vit_model)
# modified_vit_model.to(device)
# # modified_vit_model = modified_vit_model.module


# if loss_type == "cosine":
#     modified_vit_model.mlp_train()
# elif loss_type == "classification":
#     modified_vit_model.vit_train()
# elif loss_type == "both":
#     modified_vit_model.classifier_mlp_train()


# # train_dataset = CIFAR100Dataset(root=data_path, train=True, size=train_size, processor=cifar_processor)
# test_dataset = CIFAR100Dataset(root=data_path, train=False, size=test_size, processor=cifar_processor)
# # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True)


# if __name__ == '__main__':
#     # print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, full_testing=True)}')
#     # os.makedirs('logs', exist_ok=True)
#     # os.makedirs('models', exist_ok=True)
#     # log_file = open(f"logs/{save_path}.txt", "w")

#     # train(modified_vit_model, train_loader, test_loader, num_epochs=num_epochs, loss_type=loss_type, lr=lr, save_path=save_path)
    
#     # modified_vit_model.vit_train()

#     modified_vit_model.eval()
#     # train(modified_vit_model, train_loader, test_loader, num_epochs=5, loss_type="classification", lr=lr, save_path=save_path)
#     val_accuracy, mlp_val_accuracy = test(modified_vit_model, test_loader, full_testing=True)
#     # # calculating flops
#     # input_size = (3, 224, 224)
#     # flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
#     # flops_ori, params = get_model_complexity_info(pretrained_cifar_model, input_size, as_strings=True, print_per_layer_stat=False)
#     # print(flops, flops_ori)
#     print(val_accuracy, mlp_val_accuracy)
#     # log_file.close()

# """# pretrained model accuracy 89.85"""

import os
import torch
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

# Ensure images/ folder exists
os.makedirs('images', exist_ok=True)

# Load the pre-trained ViT model and feature extractor
model_name = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Define the CIFAR-100 transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-100 dataset
cifar_data = datasets.CIFAR100(root='../data', train=False, download=False, transform=transform)
data_loader = torch.utils.data.DataLoader(cifar_data, batch_size=1, shuffle=True)
r=100
# Process a batch of images
for images, labels in data_loader:
    # Convert images to PIL format for ViTImageProcessor
    images_pil = [transforms.ToPILImage()(img) for img in images]
    inputs = processor(images=images_pil, return_tensors="pt")

    # Get hidden states from ViT model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # List of hidden states for each layer

    for layer_idx, layer_hidden_states in enumerate(hidden_states):
        # For batch size 1
        cls_token = layer_hidden_states[:, 0, :]  # Shape: (1, 768)
        patch_tokens = layer_hidden_states[:, 1:, :]  # Shape: (1, 196, 768)

        # Normalize the vectors
        cls_token_norm = F.normalize(cls_token, p=2, dim=-1)
        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)

        # Compute cosine similarity between CLS token and each patch token
        cosine_sims = cosine_similarity(cls_token_norm.unsqueeze(1), patch_tokens_norm, dim=-1)  # Shape: (1, 196)

        # Get indices of the top 50 most similar patches
        topr_indices = torch.topk(cosine_sims[0], k=r).indices

        # Print the cosine similarity for each of the top 50 patches
        print(f"Layer {layer_idx + 1}: Top {r} most similar patches to CLS token")
        avg_cos_sim=0
        for i in topr_indices:
            sim = cosine_sims[0, i].item()
            avg_cos_sim+=sim
            # print(f"Patch {i.item()}: Cosine Similarity = {sim:.4f}")
        avg_cos_sim/=r
        print(f'cosim: {avg_cos_sim}')
    break  # Only process the first batch for this example

# import os
# import torch
# from torchvision import datasets, transforms
# from transformers import ViTForImageClassification, ViTImageProcessor
# import matplotlib.pyplot as plt
# import numpy as np

# # Ensure images/ folder exists
# os.makedirs('images', exist_ok=True)

# # Load the pre-trained ViT model and feature extractor
# model_name = 'google/vit-base-patch16-224-in21k'
# model = ViTForImageClassification.from_pretrained(model_name)
# processor = ViTImageProcessor.from_pretrained(model_name)

# # Define the CIFAR-100 transform pipeline
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Load CIFAR-100 dataset (Make sure it's correctly pointed to your directory)
# cifar_data = datasets.CIFAR100(root='../data', train=False, download=False, transform=transform)
# data_loader = torch.utils.data.DataLoader(cifar_data, batch_size=1, shuffle=True)

# # Process a batch of images
# for images, labels in data_loader:
#     # Convert images to PIL format for ViTImageProcessor
#     images_pil = [transforms.ToPILImage()(img) for img in images]
#     inputs = processor(images=images_pil, return_tensors="pt")

#     # Get hidden states from ViT model (all transformer layers)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#     hidden_states = outputs.hidden_states  # List of hidden states for each layer
#     import torch.nn.functional as F
#     from torch.nn.functional import cosine_similarity
#     # Loop through each layer's hidden states
#     for layer_idx, layer_hidden_states in enumerate(hidden_states):
#         cls_token = layer_hidden_states[:, 0, :]  # (batch_size, 768)
#         avg_token = layer_hidden_states[:, 1:, :].mean(dim=1)  # (batch_size, 768)

#         for i in range(layer_hidden_states.shape[0]):
        
#             # for each image
#             # print('similarity', avg_patc)
#             cls_token_norm = F.normalize(cls_token, p=2, dim=-1)
#             avg_token_norm = F.normalize(avg_token, p=2, dim=-1)
# #             print(cls_token_norm[0].shape, avg_token[0].shape)
            
# #             # cosine_sim = cosine_similarity(avg_token, cls_token_norm.transpose(1, 2), dim=-1)
# #             import torch
# #             from torch.nn.functional import cosine_similarity

# #             # # Example tensors of shape [768]
# #             tensor1 = avg_token[0]
# #             tensor2 = cls_token_norm
# #             tensor2=tensor2[0]
# #             cosine_sim = cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=-1)
# #             print("Cosine Similarity:", cosine_sim.item())
# # #     cosine_sim = cosine_similarity(avg_token[0], cls_token_norm.transpose(1, 2)[0])
# # # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)

# #             print(cosine_sim)
            
            
#             # avg_cosine_sim = cosine_sim.mean(dim=0)
#             # avg_l2_dist = l2_dist.mean(dim=0)
#             # avg_l2_dist_squared = l2_dist_squared.mean(dim=0)

#             # print("similarity", avg_cosine_sim, avg_l2_dist)

#         # Averaging over the batch
#         # avg_cls_token = cls_token.mean(dim=0)  # (768,)
#         # avg_patch_token = avg_patch.mean(dim=0)  # (768,)

#         # Convert tokens to numpy for visualization
#         # avg_cls_token_np = avg_cls_token.cpu().numpy()
#         # avg_patch_token_np = avg_patch_token.cpu().numpy()




#         # Plot the results and save the images
#         # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
#         # axes[0].imshow(avg_cls_token_np.reshape(1, -1), cmap='viridis', aspect='auto')
#         # axes[0].set_title(f"Layer {layer_idx+1} - Averaged CLS Token")
#         # axes[0].axis('off')

#         # axes[1].imshow(avg_patch_token_np.reshape(1, -1), cmap='viridis', aspect='auto')
#         # axes[1].set_title(f"Layer {layer_idx+1} - Averaged Patch Token")
#         # axes[1].axis('off')

#         # # Save the plot
#         # plt.tight_layout()
#         # plt.savefig(f'images/layer_{layer_idx+1}.png')
#         # plt.close()

#     break  # Only process the first batch for this example

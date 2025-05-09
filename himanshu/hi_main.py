import os, sys
import time
from transformers.models.vit.modeling_vit import ViTConfig
from torch import nn
import torch
from datetime import datetime
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ptflops import get_model_complexity_info
import importlib
from main_model_utils import CIFAR100Dataset, train, test, write_N_print, TinyImageNetDataset
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

module_name = "model_utils"
module = importlib.import_module(module_name)
ModifiedViTModel = module.ModifiedViTModel


seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


print("CUDA device:", os.getenv("CUDA_VISIBLE_DEVICES"), ' '.join(sys.argv))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the model')
    # parser.add_argument('--num_epochs', type=int, nargs='+', help='Number of epochs to train the model for', required=True)
    # parser.add_argument('--loss_type', type=str, nargs='+', help='Type of loss to use for training', required=True)
    # parser.add_argument('--lr', type=float, nargs='+', help='Learning rate for training', required=True)
    # parser.add_argument('train_size', type=int, help='Size of training dataset', required=True)
    # parser.add_argument('test_size', type=int, help='Size of testing dataset', required=True)
    parser.add_argument('--model_desc', '-d', type=str, help='Desc of model', required=True)

    args = parser.parse_args()

    if not os.path.exists("imagenet-val"):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("titericz/imagenet1k-val", path=os.path.join(".", "imagenet-val"), unzip=True)

    data_path = os.path.join(os.path.join(".", "imagenet-val"), "imagenet-val")
    
    # # Split the dataset into training and validation sets
    # train_data_path = os.path.join(data_path, "train")
    # val_data_path = os.path.join(data_path, "val")

    # os.makedirs(train_data_path, exist_ok=True)
    # os.makedirs(val_data_path, exist_ok=True)
    
    # for class_dir in os.listdir(data_path):

    #     class_path = os.path.join(data_path, class_dir)
    #     if os.path.isdir(class_path) and class_dir != "train" and class_dir != "val":
    #         images = os.listdir(class_path)
    #         os.makedirs(os.path.join(train_data_path, class_dir), exist_ok=True)
    #         os.makedirs(os.path.join(val_data_path, class_dir), exist_ok=True)

    #         # Split images into training and validation sets
    #         train_images = images[:25]
    #         val_images = images[25:50]

    #         for img in train_images:
    #             src = os.path.join(class_path, img)
    #             dst = os.path.join(train_data_path, class_dir, img)
    #             shutil.copy(src, dst)

    #         for img in val_images:
    #             src = os.path.join(class_path, img)
    #             dst = os.path.join(val_data_path, class_dir, img)
    #             shutil.copy(src, dst)

    # path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
    # path = kagglehub.dataset_download("titericz/imagenet1k-val")
    # data_path = os.path.join(path, "imagenet-val")  # ensure full dataset path
    
    # Define the training parameters
    pretrained_model_path = "/home/ai21btech11012/ViT-pruning/himanshu/models/2025-02-27_19-58-51_fixing_loss_weight-1.5_mlp_model_utils_loss-cosine^alternate_lr-0.001^0.0001_st-0.9_mt-0.5_bs-64_trs-10000_tes-1000_nw-16.pth"
    pretrained_model_path = None
    train_type = "both"  # mlp or vit or both or none
    loss_type = ["cosine", "classification"]
    num_epochs = [10, 10]
    lr = [1e-3, 1e-5]
    train_batch_size = 32
    test_batch_size = 128
    train_size = None
    test_size = None
    sim_threshold = .9
    mlp_threshold = 0.5
    avg_threshold = 0
    num_workers = 16
    # data_path = './data'
    DatasetClass = TinyImageNetDataset
    # pretrained_model_name = "Ahmed9275/Vit-Cifar100"
    # pretrained_model_name = "google/vit-base-patch16-224-in21k"
    pretrained_model_name = "google/vit-base-patch16-224"
    csv_path = f"stats.csv"

    combined_lr = "^".join(map(str, lr))
    combined_loss = "^".join(loss_type)

    save_path = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.model_desc}_{train_type}_{module_name}_loss-{combined_loss}_lr-{combined_lr}_st-{sim_threshold}_mt-{mlp_threshold}_at-{avg_threshold}_bs-{train_batch_size}_trs-{train_size}_tes-{test_size}_nw-{num_workers}"

    # if (train_type == "vit") == bool(pretrained_model_path):
    #     if pretrained_model_path:
    #         save_path = f"{save_path}_prevmodel-{pretrained_model_path.split('/')[-1].split('.')[0]}"
    # else:
    #     print("Not correct configuration. Exiting...")
    #     exit()

    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    pretrained_model = AutoModelForImageClassification.from_pretrained(pretrained_model_name)
    config = ViTConfig.from_pretrained(pretrained_model_name)
    config.num_labels = len(config.id2label)
    # config.num_labels = 1000
    modified_vit_model = ModifiedViTModel(config, sim_threshold, mlp_threshold, avg_threshold)

    # -----------   Copying Weights    ------------  #
    if not pretrained_model_path:
        new_state_dict = {}
        for key in pretrained_model.state_dict().keys():
            new_key = key.replace('vit.', '')
            new_state_dict[new_key] = pretrained_model.state_dict()[key]
        # new_state_dict.pop("classifier.weight", None)
        # new_state_dict.pop("classifier.bias", None)
        modified_vit_model.load_state_dict(new_state_dict, strict=False)
    else:
        modified_vit_model.load_state_dict(torch.load(pretrained_model_path))

    # modified_vit_model = nn.DataParallel(modified_vit_model)
    modified_vit_model.to(device)
    # modified_vit_model = modified_vit_model.module

    print(data_path)
    train_dataset = DatasetClass(root=data_path, train=True, size=train_size, processor=processor)
    test_dataset = DatasetClass(root=data_path, train=False, size=test_size, processor=processor)

    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True)

    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    log_file = open(f"logs/{save_path}.txt", "w")

    if train_type not in ['mlp', 'vit', 'both', 'none']:
        print("Train_type variable not correct. Exiting...")
        exit()
    
    # ----   Training and testing started   ----
    write_N_print(f'Test accuracy at starting: {test(modified_vit_model, test_loader, device, log_file, full_testing=True)}', log_file)

    if train_type in ['mlp', 'both']:
        train(modified_vit_model, train_loader, test_loader, device, log_file, num_epochs=num_epochs[0], loss_type=loss_type[0], lr=lr[0], save_path=save_path)
    
    if train_type in ['vit', 'both']:
        train(modified_vit_model, train_loader, test_loader, device, log_file, num_epochs=num_epochs[1], loss_type=loss_type[1], lr=lr[1], save_path=None)

    # ---------------------------------------------------

    accuracy, mlp_accuracy = test(modified_vit_model, test_loader, device, log_file, full_testing=True)
    if os.path.exists(csv_path):
        stats_file = open(csv_path, "a")
    else:
        stats_file = open(csv_path, "w")
        stats_file.write("Model, Accuracy, MLP Accuracy\n")
    stats_file.write(f"{save_path}, {accuracy}, {mlp_accuracy}\n")
    stats_file.close()

    # # calculating flops
    input_size = (3, 224, 224)
    flops, params = get_model_complexity_info(modified_vit_model, input_size, as_strings=True, print_per_layer_stat=False)
    flops_ori, params = get_model_complexity_info(pretrained_model, input_size, as_strings=True, print_per_layer_stat=False)
    print(flops, flops_ori)

    log_file.close()


"""# pretrained model accuracy 89.85"""

















################### ------------------------------------------------------------------------------------------------------------------############
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager", torch_dtype=torch.float32)

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
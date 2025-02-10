import os
from torchvision import datasets
from torchvision.transforms import ToPILImage

# Directory to save images
save_dir = './cifar100_test_images'
os.makedirs(save_dir, exist_ok=True)

# Download CIFAR-100 dataset
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)


# Save images
for idx, (image, label) in enumerate(test_dataset):

    image.save(os.path.join(save_dir, f"test_image_{idx:04d}.png"))

print(f"Test images saved in '{save_dir}'")

# Save labels
labels_file = os.path.join(save_dir, 'labels.txt')
with open(labels_file, 'w') as f:
    for idx, (_, label) in enumerate(test_dataset):
        label_name = test_dataset.classes[label]
        f.write(f"{idx:04d}: {label} - {label_name}\n")

print(f"Labels saved in '{labels_file}'")

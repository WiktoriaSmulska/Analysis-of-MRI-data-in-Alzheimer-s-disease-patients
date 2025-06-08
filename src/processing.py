import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

def splitting_folders():
    data_dir = 'data/raw'
    output_base_dir = "data/split"

    #kategorie
    categories = os.listdir(data_dir)

    for split in ['train', 'test']:
        for category in categories:
            os.makedirs(os.path.join(output_base_dir, split, category), exist_ok=True)


    test_size = 0.2
    for category in categories:
        image_paths = glob.glob(os.path.join(data_dir, category, "*"))

        train_path, test_path = train_test_split(image_paths, test_size=test_size, random_state=42)

        for path in train_path:
            shutil.copy(path, os.path.join(output_base_dir, "train", category))

        for path in test_path:
            shutil.copy(path, os.path.join(output_base_dir, "test", category))

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('data/split/train', transform=transform)
test_dataset = datasets.ImageFolder('data/split/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

images, label = next(iter(test_loader))
print(images.shape)








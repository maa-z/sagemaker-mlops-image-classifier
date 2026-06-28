import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


# def load_config(config_path):
#     import json
#     with open(config_path, 'r') as f:
#         config = json.load(f)
#     return config

def load_dataset(config):

# config = load_config(config_path)

    image_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]
    train_ratio = config["training"]["train_ratio"]
    random_seed = config["training"]["random_seed"]

    # dataset_dir = config["data"]["DATASET_DIR"]
    dataset_dir = "/opt/ml/input/data/training"


    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(
                root=dataset_dir,
                transform=transform
            )
    
    class_names = dataset.classes

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
                                dataset, 
                                [train_size, val_size], 
                                generator=torch.Generator().manual_seed(random_seed))
    
    train_loader = DataLoader(
        train_dataset,  
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,    
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, class_names


# epochs
# batch_size
# learning_rate
# image_size
# train_ratio
# random_seed
# model_name
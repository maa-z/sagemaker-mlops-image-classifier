import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def create_model(config, num_classes):

    model_name = config["training"]["model_name"]

    if model_name == "resnet18":
        model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features, 
            num_classes)

    else:

        raise ValueError(f"Unsupported model name: {model_name}")

    return model
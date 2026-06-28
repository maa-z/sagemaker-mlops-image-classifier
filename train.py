import json
import torch

from src.dataset import load_dataset
from src.model import create_model
from src.trainer import train_model


def load_config(path):

    with open(path) as f:

        return json.load(f)

def main():

    MODEL_DIR = "/opt/ml/model"

    config = load_config("config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = load_dataset(config)

    model = create_model(config,len(class_names))


    best_model = train_model(model,train_loader,val_loader,config,device)

    torch.save(best_model.state_dict(),f"{MODEL_DIR}/best_model.pth")

if __name__ == "__main__":
    main()
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from src.evaluator import evaluate_model



def train_model(model,train_loader,val_loader, config,device):

    training_config = config["training"]
    epochs = training_config["epochs"]
    learning_rate = training_config["learning_rate"]

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
                    model.parameters(),
                    lr=learning_rate
                )

    model.to(device)

  

    
    best_accuracy = float("-inf")
    best_model = None

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        accuracy = evaluate_model(
            model,
            val_loader,
            device
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)


        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")  

    return best_model
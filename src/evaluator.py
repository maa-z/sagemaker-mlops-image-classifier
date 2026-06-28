import torch

def evaluate_model(model,data_loader,device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = torch.argmax(outputs,dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy





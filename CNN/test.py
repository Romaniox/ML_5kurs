import os
import numpy as np
import cv2
import torch
import enum
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn


class ModelType(enum.Enum):
    JIT = 1
    PT = 2


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_name = 'exp13_best'
    model_type = ModelType.JIT

    match model_type:
        case ModelType.JIT:
            model = torch.jit.load(os.path.join(os.getcwd(), exp_name, 'model.pt'))
        case ModelType.PT:
            from train import CNN
            model = torch.load(os.path.join(os.getcwd(), exp_name, 'model.pth'))
        case _:
            raise ValueError('Invalid model type')

    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
    )

    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        total_loss /= len(test_loader)
        print(
            f'Loss: {total_loss:.4f}, '
            f'Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    main()

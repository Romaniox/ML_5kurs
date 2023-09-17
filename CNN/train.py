import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import pandas as pd
from config import cfg

from utils import get_logger, get_dataloaders, create_plot


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=8,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(32 * 3 * 3, cfg.DATA.NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def train(model: nn.Module,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int,
          device: torch.device,
          logger: logging.Logger):
    metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            total_loss /= len(train_loader)
            logger.info(f'Epoch [{epoch + 1}/{epochs}], '
                        f'Loss: {total_loss:.4f}, '
                        f'Accuracy: {accuracy:.2f}%')

            # concatenate metrics
            metrics = pd.concat([metrics, pd.DataFrame([[epoch + 1, total_loss, accuracy]],
                                                       columns=['epoch', 'loss', 'accuracy'])])

    metrics.to_csv(cfg.DATA.ROOT / cfg.TRAIN.EXPERIMENT_NAME / 'metrics.csv', index=False)
    # torch.save(model.state_dict(), cfg.DATA.ROOT / cfg.TRAIN.EXPERIMENT_NAME / 'model.pth')
    torch.save(model, cfg.DATA.ROOT / cfg.TRAIN.EXPERIMENT_NAME / 'model.pth')
    torch.jit.save(torch.jit.script(model), cfg.DATA.ROOT / cfg.TRAIN.EXPERIMENT_NAME / 'model.pt')

    create_plot(metrics, cfg.DATA.ROOT / cfg.TRAIN.EXPERIMENT_NAME / 'metrics.png')
    print('Finished Training')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # print number of parameters for each layer
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    save_path = cfg.DATA.ROOT / cfg.TRAIN.EXPERIMENT_NAME
    os.makedirs(save_path, exist_ok=True)

    train_loader, test_loader = get_dataloaders(cfg.DATA.TRAIN_BATCH_SIZE, cfg.DATA.TEST_BATCH_SIZE)

    logger = get_logger('train')

    epochs = cfg.TRAIN.EPOCHS

    criterion = nn.CrossEntropyLoss()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=0.001)
    else:
        raise ValueError(f'Unknown optimizer: {cfg.TRAIN.OPTIMIZER}')

    train(model, criterion, optimizer, train_loader, test_loader, epochs, device, logger)


if __name__ == '__main__':
    main()

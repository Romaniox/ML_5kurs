from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import logging
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


def get_logger(name, log_file=None, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def get_dataloaders(train_batch_size, test_batch_size):
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def create_plot(metrics: pd.DataFrame, save_path: str):
    sns.set_style("darkgrid")
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.lineplot(x='epoch', y='loss', data=metrics, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('График Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.lineplot(x='epoch', y='accuracy', data=metrics, label='Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('График Accuracy')
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)

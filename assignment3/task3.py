import importlib
import torch
import torch.nn as nn
from dataloaders import load_cifar10
import numpy as np

from trainer import Trainer, compute_loss_and_accuracy
from task2 import create_plots, ExampleModel


def run():
    torch.multiprocessing.freeze_support()


if __name__ == '__main__':
    run()
    epochs = 1
    batch_size = 64
    learning_rate = 1e-2  # Should be 5e-5 for LeNet
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()

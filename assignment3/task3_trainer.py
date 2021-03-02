import importlib
import torch
import torch.nn as nn
from dataloaders import load_cifar10
import numpy as np

from trainer import Trainer, compute_loss_and_accuracy
from task2 import create_plots, test_model, print_results
from task3 import ModelX, ModelY


def run():
    torch.multiprocessing.freeze_support()


if __name__ == '__main__':
    run()
    epochs = 0
    batch_size = 64
    learning_rate = 1e-2  # Should be 5e-5 for LeNet
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size, trans_opt='y')
    """
    model_x = ModelX(image_channels=3, num_classes=10)
    trainer_x = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_x,
        dataloaders
    )
    trainer_x.train()
    """
    model_y = ModelY(image_channels=3, num_classes=10)
    trainer_y = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_y,
        dataloaders
    )
    trainer_y.train()

    # test_model(trainer_x)
    test_model(trainer_y)

    #create_plots(trainer_x, "Trainer X")
    create_plots(trainer_y, "Trainer Y")

import importlib
import torch
import torch.nn as nn
from dataloaders_task4 import load_cifar10
import numpy as np

import trainer_task4
import task2
import task4
from trainer_task4 import Trainer, compute_loss_and_accuracy
from task2 import create_plots, test_model, print_results
from task4 import Model


def run():
    torch.multiprocessing.freeze_support()


if __name__ == '__main__':
    run()

    epochs = 1
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)

    model = Model()
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )

    trainer.train()

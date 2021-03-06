import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)

        self.hidden_layer_units = 64
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # First layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # Second Layer
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2*num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # Third Layer
            nn.Conv2d(
                in_channels=2*num_filters,
                out_channels=4*num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4 * num_filters * 4 * 4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, self.hidden_layer_units),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_units, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]

        features = self.feature_extractor(x)
        # Flatten before classification layer
        features = torch.reshape(
            features, (x.shape[0], self.num_output_features))
        out = self.classifier(features)

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.xlabel("Training steps")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(
        trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.xlabel("Training steps")
    utils.plot_loss(
        trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def plot_comparison(trainer1: Trainer, trainer2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.title("Loss")
    plt.xlabel("Training steps")
    utils.plot_loss(
        trainer1.train_history["loss"], label="Training loss, Model X", npoints_to_average=10)
    utils.plot_loss(
        trainer1.validation_history["loss"], label="Validation loss, Model X")
    utils.plot_loss(
        trainer2.train_history["loss"], label="Training loss, Base Model", npoints_to_average=10)
    utils.plot_loss(
        trainer2.validation_history["loss"], label="Validation loss, Base Model")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def print_final_results(trainer: Trainer):
    # Computes final training-, validation- and test-results
    # over their entire respective datasets and prints them

    # Turn of training-spesicif parts of network training
    trainer.model.eval()

    # Turn of gradient computation and calculate results
    with torch.no_grad():
        # Train data
        train_loss, train_acc = compute_loss_and_accuracy(
            trainer.dataloader_train, trainer.model, trainer.loss_criterion
        )

        # Validation data
        val_loss, val_acc = compute_loss_and_accuracy(
            trainer.dataloader_val, trainer.model, trainer.loss_criterion
        )

        # Test data
        test_loss, test_acc = compute_loss_and_accuracy(
            trainer.dataloader_test, trainer.model, trainer.loss_criterion
        )

    # Turn training mode back on
    trainer.model.train()

    # Print final results
    print(f"Training Loss: {train_loss:.2f}",
          f"Training Accuracy: {train_acc:.3f}",
          f"Validation Loss: {val_loss:.2f}",
          f"Validation Accuracy: {val_acc:.3f}",
          f"Test Loss: {test_loss:.2f}",
          f"Test Accuracy: {test_acc:.3f}",
          sep=", ")

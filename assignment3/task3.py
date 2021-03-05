import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ModelX(nn.Module):

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

        # Model Parameters
        self.conv_stride = 1
        self.pool_stride = 2
        self.conv_kernel = 5
        self.hidden_layer_units = 64
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # First layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
            # Second Layer
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2*num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
            # Third Layer
            nn.Conv2d(
                in_channels=2*num_filters,
                out_channels=4*num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4 * num_filters * 4 * 4

        # Spatial batch normalization
        self.spatial_normalizer = nn.BatchNorm2d(4 * num_filters)

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

        batch_size = x.shape[0]

        # Extract features
        features = self.feature_extractor(x)

        # Flatten before classification layer
        features = torch.reshape(
            features, (x.shape[0], self.num_output_features))

        # Classify
        out = self.classifier(features)

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class ModelY(nn.Module):

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

        # Model Parameters
        self.conv_stride = 1
        self.pool_stride = 2
        self.conv_kernel = 5
        self.hidden_layer_units = 64
        num_filters = 32  # Number of filters in first conv layer
        self.num_classes = num_classes

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # First layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
            # Second Layer
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2*num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
            # Third Layer
            nn.Conv2d(
                in_channels=2*num_filters,
                out_channels=4*num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
        )

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4 * num_filters * 4 * 4

        # Spatial batch normalization
        self.spatial_normalizer = nn.BatchNorm2d(4 * num_filters)

        # Dropout
        self.dropout = nn.Dropout(0.25)

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

        batch_size = x.shape[0]

        # Extract features
        features = self.feature_extractor(x)

        # Normalize features
        features_normalized = self.spatial_normalizer(features)

        # Flatten before classification layer
        features_normalized = torch.reshape(
            features_normalized, (x.shape[0], self.num_output_features))

        # Apply dropout
        features_normalized = self.dropout(features_normalized)

        # Pass through classification layer
        out = self.classifier(features_normalized)

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class ModelZ(nn.Module):

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

        # Model Parameters
        self.conv_stride = 1
        self.pool_stride = 2
        self.conv_kernel = 5
        self.hidden_layer_units = 64
        num_filters = 32  # Number of filters in first conv layer
        self.num_classes = num_classes

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # First layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
            # Second Layer
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2*num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
            # Third Layer
            nn.Conv2d(
                in_channels=2*num_filters,
                out_channels=4*num_filters,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=self.pool_stride
            ),
        )

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4 * num_filters * 4 * 4

        # Spatial batch normalization
        self.spatial_normalizer = nn.BatchNorm2d(4 * num_filters)

        # Dropout
        self.dropout = nn.Dropout(0.7)

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

        batch_size = x.shape[0]

        # Extract features
        features = self.feature_extractor(x)

        # Normalize features
        features_normalized = self.spatial_normalizer(features)

        # Flatten before classification layer
        features_normalized = torch.reshape(
            features_normalized, (x.shape[0], self.num_output_features))

        # Apply dropout
        features_normalized = self.dropout(features_normalized)

        # Pass through classification layer
        out = self.classifier(features_normalized)

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

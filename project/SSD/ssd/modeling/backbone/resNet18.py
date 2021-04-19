import torchvision
import torch
from torch import nn
import numpy as np


class ResNet18(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        

        #self.model.fc = nn.Linear(512, 10)  # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters():  # Freeze all parameters
            param.requires_grad = False
        """
        for param in self.model.fc.parameters():  # Unfreeze the last fully-connected
            param.requires_grad = True  # layer
        """
        for param in self.model.layer4.parameters():  # Unfreeze the last 5 convolutional
            param.requires_grad = True  # layers


    def forward(self, x):
        # TODO: Add documentation

        out_features = []

        x = self.model.conv1(x)
        idx_counter = 0

        layers = nn.Sequential(*(list(self.model.children())[1:]))

        feature_layers = [5,6,7,8]
        for layer in layers:
            idx_counter += 1
            x = layer(x)
            if idx_counter in feature_layers:
                out_features.append(x)
            
        # Ensure correct shapes
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
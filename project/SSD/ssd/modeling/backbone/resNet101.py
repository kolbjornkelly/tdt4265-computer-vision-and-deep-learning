import torchvision
import torch
from torch import nn
import numpy as np


class ResNet101(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        # The following code can freeze layers
        """
        # Freeze all parameters
        for param in self.model.parameters():  
            param.requires_grad = False
        # Unfreeze some parameters
        for param in self.model.fc.parameters(): 
            param.requires_grad = True 
        for param in self.model.layer4.parameters():
            param.requires_grad = True 
        """

    def forward(self, x):

        # TODO: Add documentation
        out_features = []

        # Old implementation, using features from all layers
        """
        # Pass through first layer
        out_features.append(self.model.conv1(x))
        # Define remaining layers
        layers = nn.Sequential(*(list(self.model.children())[1:9]))

        # Pass through remaining layers
        idx_counter = 0
        layers_to_use = []
        for layer in layers:
            if idx_counter == 
            out_features.append(layer(out_features[-1]))
        """
        # New implementation, using features from some layers
        x = self.model.conv1(x)
        idx_counter = 0
        # Removed: 4
        layers_to_use = [5,6,7,8]

        layers = nn.Sequential(*(list(self.model.children())[1:9]))

        for layer in layers:
            idx_counter += 1
            x = layer(x)
            if idx_counter in layers_to_use:
                out_features.append(x)
            
        # Ensure correct shapes
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

            
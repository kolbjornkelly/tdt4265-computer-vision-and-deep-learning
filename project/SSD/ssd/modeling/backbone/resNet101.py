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

        # Freeze all parameters
        for param in self.model.parameters():  
            param.requires_grad = False

    def forward(self, x):

        # TODO: Add documentation
        out_features = []
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
        x = self.model.conv1(x)
        idx_counter = 0
        layers_to_use = [2,4,5,6,7,8]

        layers = nn.Sequential(*(list(self.model.children())[1:9]))

        for layer in layers:
            x = layer(x)
            if idx_counter in layers_to_use:
                out_features.append(x)
            idx_counter += 1


        
        # Ensure correct shapes
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

            
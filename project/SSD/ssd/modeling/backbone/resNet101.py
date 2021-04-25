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


    def forward(self, x):

        out_features = []

        x = self.model.conv1(x)
        idx_counter = 0
        
        # Select layers for feature map extraction
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

            
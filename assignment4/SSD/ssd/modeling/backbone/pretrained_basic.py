import torch
from torch import nn
from torch import torchvision


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.num_classes = 10

        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)

        self.model.fc = nn.Linear(512, 10)  # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters():  # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters():  # Unfreeze the last fully-connected
            param.requires_grad = True  # layer
        for param in self.model.layer4.parameters():  # Unfreeze the last 5 convolutional
            param.requires_grad = True  # layers

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        out_features = []
        # Feed through network
        #backbone0 = self.backbone_0(x)
        out_features.append(self.backbone_1(x))
        out_features.append(self.backbone_2(out_features[-1]))
        out_features.append(self.backbone_3(out_features[-1]))
        out_features.append(self.backbone_4(out_features[-1]))
        out_features.append(self.backbone_5(out_features[-1]))

        out_features.append(self.backbone_6(out_features[-1]))

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

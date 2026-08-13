import torch.nn as nn
from torchvision import models


class RESNETFRModel(nn.Module):
    """

    A ResNet-based model for face recognition tasks. This model allows for optional pretraining, freezing of the backbone, and customization of the number of output classes.
    Args:
        pretrained (bool): If True, initializes the model with pretrained weights. Default is False.
        freeze_backbone (bool): If True, freezes the backbone layers of the model, preventing them from being updated during training. Default is False.
        num_classes (int): The number of output classes for the final fully connected layer. If set to 0, the final layer will be an identity layer. Default is 0.
    Attributes:
        model (nn.Module): The underlying ResNet model.
        in_features (int): The number of input features to the final fully connected layer.
    Methods:
        forward(x): Defines the forward pass of the model, passing input through the ResNet architecture

    """

    def __init__(
        self,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
        num_classes=0,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes

        if backbone_name == "resnet18":
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        elif backbone_name == "resnet50":
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. Only 'resnet18' and 'resnet50' are supported."
            )
        self.in_features = self.model.fc.in_features
        self.embedding_size = self.in_features

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # If not freezing the backbone, ensure all parameters are trainable
            for param in self.model.parameters():
                param.requires_grad = True

        # Replace the final fully connected layer with a new one for the desired number of classes
        self.model.fc = (
            nn.Linear(self.in_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, x):
        return self.model(x)

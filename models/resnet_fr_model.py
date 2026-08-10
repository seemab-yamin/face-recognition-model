import torch.nn as nn
from torchvision import models


class RESNETFRModel(nn.Module):
    def __init__(
        self, pretrained=False, freeze_backbone=False, num_classes=0
    ) -> nn.Module:
        super().__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.in_features = self.model.fc.in_features

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer with a new one for the desired number of classes
        self.model.fc = (
            nn.Linear(self.in_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, x):
        return self.model(x)

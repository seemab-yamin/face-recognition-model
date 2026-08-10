import torch
import torch.nn as nn

from models.resnet_fr_model import RESNETFRModel


def make_model(
    arch: str, num_classes: int, pretrained: bool, freeze_backbone: bool
) -> nn.Module:
    """

    Create a model based on the specified architecture.
    Args:
        arch (str): The architecture of the model (e.g., 'resnet').
        num_classes (int): The number of output classes for the model.
        pretrained (bool): Whether to use a pretrained model.
        freeze_backbone (bool): Whether to freeze the backbone of the model.
    Returns:
        nn.Module: The created model.

    """

    if arch == "resnet":
        model = RESNETFRModel(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


if __name__ == "__main__":
    arch = "resnet"
    num_classes = 10
    pretrained = False
    freeze_backbone = False
    model = make_model(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    print(f"Selected model:\n{arch} with pretrained={pretrained}")
    model_summary = str(model)
    print(f"Created model:\n{model_summary}")

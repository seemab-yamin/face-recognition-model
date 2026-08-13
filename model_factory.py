import torch.nn as nn

from models import RESNETFRModel


def make_model(
    arch: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    return_embeddings: bool,
    device,
) -> nn.Module:
    """

    Create a model based on the specified architecture.
    Args:
        arch (str): The architecture of the model (e.g., 'resnet18', 'resnet50').
        num_classes (int): The number of output classes for the model.
        pretrained (bool): Whether to use a pretrained model.
        freeze_backbone (bool): Whether to freeze the backbone of the model.
    Returns:
        nn.Module: The created model.

    """

    if "resnet" in arch:
        model = RESNETFRModel(
            backbone_name=arch,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            num_classes=0 if return_embeddings else num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model.to(device)


if __name__ == "__main__":
    arch = "resnet18"
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
    print(f"Created model:\n{model}")

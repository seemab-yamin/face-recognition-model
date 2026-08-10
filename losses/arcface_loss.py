import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        # Block 2: Normalize and compute cos_theta
        x = F.normalize(input, dim=1)
        w = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(x, w)

        # Block 3: Compute cos(θ + m)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta = torch.sqrt(1.0 - cos_theta**2)
        cos_theta_m = cos_theta * torch.cos(self.m) - sin_theta * torch.sin(self.m)

        # Block 4: Apply margin to ground-truth class
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0])
        logits = torch.where(one_hot == 1, cos_theta_m, cos_theta)

        # Block 5: Scale and return
        logits = self.s * logits
        return logits

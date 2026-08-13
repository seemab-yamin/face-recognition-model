import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.register_buffer("s", torch.tensor(s))
        self.register_buffer("m", torch.tensor(m))
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        # Normalize
        x = F.normalize(input, dim=1)
        w = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(x, w)

        # Stability: clamp cos_theta to [-1, 1] to prevent sqrt of negative
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Stability: safe sqrt with epsilon to prevent 0 denominator
        eps = 1e-7
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=eps, max=1.0))

        # cos(θ + m) = cosθ*cosm - sinθ*sinm
        cosm = torch.cos(self.m)
        sinm = torch.sin(self.m)
        cos_theta_m = cos_theta * cosm - sin_theta * sinm

        # Clamp final logits
        cos_theta_m = torch.clamp(cos_theta_m, -1.0, 1.0)

        # Apply margin to ground-truth class
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0])
        logits = torch.where(one_hot == 1, cos_theta_m, cos_theta)

        # Scale and clamp to prevent overflow
        logits = self.s * logits
        logits = torch.clamp(logits, -100.0, 100.0)

        # Single line summary if you want to monitor stability
        # print(f"ArcFace: cos=[{cos_theta.min():.3f}, {cos_theta.max():.3f}], logits=[{logits.min():.1f}, {logits.max():.1f}]")

        return logits

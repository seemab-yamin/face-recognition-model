import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import check_tensor


class ArcFace(nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        # Use buffers for device compatibility
        self.register_buffer("s", torch.tensor(s))
        self.register_buffer("m", torch.tensor(m))
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        # Normalize
        x = F.normalize(input, dim=1)
        w = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(x, w)
        check_tensor(cos_theta, "cos_theta")

        # 🔒 STABILITY FIX 1: Clamp cos_theta to [-1, 1] to prevent sqrt of negative
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        check_tensor(cos_theta, "cos_theta")

        # 🔒 STABILITY FIX 2: Use safe sqrt with epsilon to prevent 0 denominator
        eps = 1e-7
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=eps, max=1.0))
        check_tensor(sin_theta, "sin_theta")

        # 🔒 STABILITY FIX 3: Avoid grad explosion near sin=0 by using safe formula
        # For cos near ±1, the formula is unstable; use alternative
        # cos(θ + m) = cosθ*cosm - sinθ*sinm
        cosm = torch.cos(self.m)
        sinm = torch.sin(self.m)
        cos_theta_m = cos_theta * cosm - sin_theta * sinm
        check_tensor(cos_theta_m, "cos_theta_m")

        # 🔒 STABILITY FIX 4: Clamp final logits
        cos_theta_m = torch.clamp(cos_theta_m, -1.0, 1.0)
        check_tensor(cos_theta_m, "cos_theta_m")

        # Apply margin
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0])
        logits = torch.where(one_hot == 1, cos_theta_m, cos_theta)
        check_tensor(logits, "logits")
        # 🔒 STABILITY FIX 5: Scale but keep within reasonable range
        logits = self.s * logits
        check_tensor(logits, "logits")

        # 🔒 STABILITY FIX 6: Final clamp to prevent overflow in cross-entropy
        logits = torch.clamp(logits, -100.0, 100.0)
        check_tensor(logits, "logits")
        print(f"cos_theta range: [{cos_theta.min():.4f}, {cos_theta.max():.4f}]")
        print(f"sin_theta min: {sin_theta.min():.4f}")  # If near 0, gradient explodes
        print(f"logits max: {logits.max():.4f}")  # If > 100, scale too high
        return logits

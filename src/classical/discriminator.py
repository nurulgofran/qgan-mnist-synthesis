import torch
import torch.nn as nn
from typing import List, Optional, Literal


class Discriminator4x4(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: List[int] = [32, 16],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


class Discriminator16x16(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))

            if use_batchnorm and i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


class ConvDiscriminator16x16(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 2 * 2, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 16, 16)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv(x)
        return self.fc(x)


def get_discriminator(
    image_size: int, architecture: Literal["mlp", "conv"] = "mlp", **kwargs
) -> nn.Module:
    if image_size == 4:
        return Discriminator4x4(input_dim=16, **kwargs)
    elif image_size == 16:
        if architecture == "conv":
            return ConvDiscriminator16x16(**kwargs)
        else:
            return Discriminator16x16(input_dim=256, **kwargs)
    else:
        raise ValueError(f"Unsupported image_size: {image_size}. Use 4 or 16.")

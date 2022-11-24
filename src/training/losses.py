import torch
import torch.nn as nn
from typing import Tuple, Literal


class BCEGANLoss:
    def __init__(self, label_smoothing: float = 0.0):
        self.criterion = nn.BCELoss()
        self.label_smoothing = label_smoothing

    def discriminator_loss(
        self, real_pred: torch.Tensor, fake_pred: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = real_pred.size(0)

        real_labels = torch.full(
            (batch_size, 1), 1.0 - self.label_smoothing, device=device
        )
        fake_labels = torch.zeros(batch_size, 1, device=device)

        loss_real = self.criterion(real_pred, real_labels)
        loss_fake = self.criterion(fake_pred, fake_labels)

        return loss_real + loss_fake, loss_real, loss_fake

    def generator_loss(
        self, fake_pred: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        batch_size = fake_pred.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        return self.criterion(fake_pred, real_labels)


class WassersteinLoss:
    def __init__(self, gradient_penalty_weight: float = 10.0):
        self.gp_weight = gradient_penalty_weight

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        discriminator: nn.Module = None,
        real_data: torch.Tensor = None,
        fake_data: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wasserstein_dist = fake_pred.mean() - real_pred.mean()

        gp = torch.tensor(0.0)
        if (
            discriminator is not None
            and real_data is not None
            and fake_data is not None
        ):
            gp = self._gradient_penalty(discriminator, real_data, fake_data)

        total_loss = wasserstein_dist + self.gp_weight * gp

        return total_loss, wasserstein_dist, gp

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        return -fake_pred.mean()

    def _gradient_penalty(
        self, discriminator: nn.Module, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        device = real_data.device
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1, device=device)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        d_interpolated = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty


class FeatureMatchingLoss:
    def __init__(self, feature_layer: int = -2):
        self.feature_layer = feature_layer
        self.mse = nn.MSELoss()

    def __call__(
        self, real_features: torch.Tensor, fake_features: torch.Tensor
    ) -> torch.Tensor:
        return self.mse(fake_features.mean(dim=0), real_features.mean(dim=0).detach())


def get_loss_function(loss_type: Literal["bce", "wasserstein"] = "bce", **kwargs):
    if loss_type == "bce":
        return BCEGANLoss(**kwargs)
    elif loss_type == "wasserstein":
        return WassersteinLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

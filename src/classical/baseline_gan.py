"""
Classical Baseline GAN for comparison with Quantum GAN.

This module implements a classical generator of similar complexity to the quantum
generator, enabling fair performance comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class ClassicalGenerator4x4(nn.Module):
    """Classical generator for 4x4 images (comparable to 4-qubit QGAN)."""

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dims: List[int] = [32, 32],
        output_dim: int = 16,
    ):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class ClassicalGenerator16x16(nn.Module):
    """Classical generator for 16x16 images (comparable to 8-qubit QGAN)."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: List[int] = [64, 128, 256],
        output_dim: int = 256,
    ):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


def get_classical_generator(
    image_size: int,
    latent_dim: int,
    **kwargs
) -> nn.Module:
    """Factory function to get classical generator matching image size."""
    if image_size == 4:
        return ClassicalGenerator4x4(latent_dim=latent_dim, **kwargs)
    elif image_size == 16:
        return ClassicalGenerator16x16(latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported image_size: {image_size}. Use 4 or 16.")


class ClassicalGANTrainer:
    """Training loop for classical GAN (similar interface to QGANTrainer)."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        latent_dim: int,
        device: str = "cpu",
        learning_rate_g: float = 0.0002,
        learning_rate_d: float = 0.0002,
    ):
        self.device = torch.device(device)
        self.latent_dim = latent_dim

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.criterion = nn.BCELoss()

        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            betas=(0.5, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            betas=(0.5, 0.999)
        )

        self.history = {
            "loss_d": [],
            "loss_g": [],
            "loss_d_real": [],
            "loss_d_fake": [],
        }

    def sample_latent(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def train_discriminator_step(
        self, real_data: torch.Tensor
    ) -> Tuple[float, float, float]:
        batch_size = real_data.size(0)
        self.optimizer_d.zero_grad()

        # Real data
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_pred = self.discriminator(real_data)
        loss_real = self.criterion(real_pred, real_labels)

        # Fake data
        z = self.sample_latent(batch_size)
        fake_data = self.generator(z).detach()
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_pred = self.discriminator(fake_data)
        loss_fake = self.criterion(fake_pred, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        self.optimizer_d.step()

        return loss_d.item(), loss_real.item(), loss_fake.item()

    def train_generator_step(self, batch_size: int) -> float:
        self.optimizer_g.zero_grad()

        z = self.sample_latent(batch_size)
        fake_data = self.generator(z)
        
        # Generator wants discriminator to think fakes are real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_pred = self.discriminator(fake_data)
        loss_g = self.criterion(fake_pred, real_labels)

        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.item()

    def train(
        self,
        dataloader,
        n_epochs: int,
        n_critic: int = 1,
        verbose: bool = True,
    ) -> dict:
        from tqdm import tqdm

        epoch_iterator = (
            tqdm(range(n_epochs), desc="Classical GAN Training")
            if verbose else range(n_epochs)
        )

        for epoch in epoch_iterator:
            epoch_loss_d = 0.0
            epoch_loss_g = 0.0
            epoch_loss_d_real = 0.0
            epoch_loss_d_fake = 0.0
            n_batches = 0

            for batch_idx, (real_images, _) in enumerate(dataloader):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Train discriminator
                for _ in range(n_critic):
                    loss_d, loss_real, loss_fake = self.train_discriminator_step(
                        real_images
                    )

                # Train generator
                loss_g = self.train_generator_step(batch_size)

                epoch_loss_d += loss_d
                epoch_loss_g += loss_g
                epoch_loss_d_real += loss_real
                epoch_loss_d_fake += loss_fake
                n_batches += 1

            # Record epoch averages
            self.history["loss_d"].append(epoch_loss_d / n_batches)
            self.history["loss_g"].append(epoch_loss_g / n_batches)
            self.history["loss_d_real"].append(epoch_loss_d_real / n_batches)
            self.history["loss_d_fake"].append(epoch_loss_d_fake / n_batches)

            if verbose:
                epoch_iterator.set_postfix({
                    "D": f"{epoch_loss_d / n_batches:.4f}",
                    "G": f"{epoch_loss_g / n_batches:.4f}"
                })

        return self.history

    def generate(self, n_samples: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = self.sample_latent(n_samples)
            samples = self.generator(z).cpu().numpy()
        self.generator.train()
        return samples

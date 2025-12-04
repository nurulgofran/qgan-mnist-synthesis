import torch
import torch.nn as nn
import numpy as np
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional, Callable
import logging
from tqdm import tqdm

from .losses import BCEGANLoss, get_loss_function

logger = logging.getLogger(__name__)


class QGANTrainer:
    def __init__(
        self,
        generator_qnn: SamplerQNN,
        discriminator: nn.Module,
        latent_dim: int,
        device: str = "cpu",
        learning_rate_g: float = 0.01,
        learning_rate_d: float = 0.001,
        loss_type: str = "bce",
        label_smoothing: float = 0.0,
    ):
        self.device = torch.device(device)
        self.latent_dim = latent_dim

        self.generator = TorchConnector(generator_qnn)
        self.generator.to(self.device)

        self.discriminator = discriminator.to(self.device)

        self.loss_fn = get_loss_function(loss_type, label_smoothing=label_smoothing)

        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate_g
        )

        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            betas=(0.5, 0.999),
        )

        self.history = {
            "loss_d": [],
            "loss_g": [],
            "loss_d_real": [],
            "loss_d_fake": [],
        }

        self.best_loss_g = float("inf")
        self.best_weights_g = None

        logger.info(f"QGANTrainer initialized on {device}")

    def sample_latent(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def train_discriminator_step(
        self, real_data: torch.Tensor
    ) -> Tuple[float, float, float]:
        batch_size = real_data.size(0)

        self.optimizer_d.zero_grad()

        real_pred = self.discriminator(real_data)

        z = self.sample_latent(batch_size)
        with torch.no_grad():
            fake_data = self.generator(z)
        fake_pred = self.discriminator(fake_data)

        loss_d, loss_real, loss_fake = self.loss_fn.discriminator_loss(
            real_pred, fake_pred, self.device
        )

        loss_d.backward()
        self.optimizer_d.step()

        return loss_d.item(), loss_real.item(), loss_fake.item()

    def train_generator_step(self, batch_size: int) -> float:
        self.optimizer_g.zero_grad()

        z = self.sample_latent(batch_size)
        fake_data = self.generator(z)

        fake_pred = self.discriminator(fake_data)

        loss_g = self.loss_fn.generator_loss(fake_pred, self.device)

        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.item()

    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        n_critic: int = 1,
        save_interval: int = 10,
        sample_callback: Optional[Callable] = None,
        verbose: bool = True,
    ) -> Dict:
        logger.info(f"Starting training for {n_epochs} epochs")

        epoch_iterator = (
            tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
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

                for _ in range(n_critic):
                    loss_d, loss_real, loss_fake = self.train_discriminator_step(
                        real_images
                    )

                loss_g = self.train_generator_step(batch_size)

                epoch_loss_d += loss_d
                epoch_loss_g += loss_g
                epoch_loss_d_real += loss_real
                epoch_loss_d_fake += loss_fake
                n_batches += 1

            avg_loss_d = epoch_loss_d / n_batches
            avg_loss_g = epoch_loss_g / n_batches
            avg_loss_d_real = epoch_loss_d_real / n_batches
            avg_loss_d_fake = epoch_loss_d_fake / n_batches

            self.history["loss_d"].append(avg_loss_d)
            self.history["loss_g"].append(avg_loss_g)
            self.history["loss_d_real"].append(avg_loss_d_real)
            self.history["loss_d_fake"].append(avg_loss_d_fake)

            if avg_loss_g < self.best_loss_g:
                self.best_loss_g = avg_loss_g
                self.best_weights_g = {
                    k: v.clone() for k, v in self.generator.state_dict().items()
                }

            if verbose:
                epoch_iterator.set_postfix(
                    {"D": f"{avg_loss_d:.4f}", "G": f"{avg_loss_g:.4f}"}
                )

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} | "
                    f"D Loss: {avg_loss_d:.4f} | G Loss: {avg_loss_g:.4f}"
                )

            if (epoch + 1) % save_interval == 0 and sample_callback:
                with torch.no_grad():
                    z = self.sample_latent(16)
                    samples = self.generator(z).cpu().numpy()
                sample_callback(samples, epoch + 1)

        logger.info("Training complete!")
        return self.history

    def generate(self, n_samples: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = self.sample_latent(n_samples)
            samples = self.generator(z).cpu().numpy()
        self.generator.train()
        return samples

    def load_best_generator(self):
        if self.best_weights_g is not None:
            self.generator.load_state_dict(self.best_weights_g)
            logger.info(f"Loaded best generator (loss: {self.best_loss_g:.4f})")

    def save_checkpoint(self, path: str):
        checkpoint = {
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "optimizer_g_state": self.optimizer_g.state_dict(),
            "optimizer_d_state": self.optimizer_d.state_dict(),
            "history": self.history,
            "best_loss_g": self.best_loss_g,
            "best_weights_g": self.best_weights_g,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state"])
        self.history = checkpoint["history"]
        self.best_loss_g = checkpoint["best_loss_g"]
        self.best_weights_g = checkpoint["best_weights_g"]
        logger.info(f"Checkpoint loaded from {path}")

"""
GPU-Accelerated QGAN Trainer.

Centralized trainer class for GPU quantum generator training,
used by both run_experiment.py and run_comparison.py.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any

from src.quantum.gpu_quantum import GPUQuantumGenerator


class GPUQGANTrainer:
    """
    QGAN Trainer using GPU-accelerated quantum generator.
    
    This trainer provides a complete training loop with:
    - Discriminator and generator alternating updates
    - Learning rate scheduling
    - Gradient clipping
    - Checkpoint saving
    - Training history tracking
    """
    
    def __init__(
        self,
        generator: GPUQuantumGenerator,
        discriminator: nn.Module,
        device: str = 'cuda',
        lr_g: float = 0.01,
        lr_d: float = 0.001,
        label_smoothing: float = 0.1,
        use_scheduler: bool = True,
        grad_clip: float = 1.0,
    ):
        """
        Initialize the GPU QGAN trainer.
        
        Args:
            generator: GPU quantum generator instance
            discriminator: Classical discriminator network
            device: Device to train on ('cuda' or 'cpu')
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            label_smoothing: Label smoothing factor for real labels
            use_scheduler: Whether to use learning rate scheduler
            grad_clip: Gradient clipping threshold
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.grad_clip = grad_clip
        
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g)
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_g, T_max=100, eta_min=1e-5
            )
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d, T_max=100, eta_min=1e-6
            )
        
        self.label_smoothing = label_smoothing
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.history: Dict[str, list] = {
            'loss_d': [],
            'loss_g': [],
            'loss_d_real': [],
            'loss_d_fake': [],
        }
        
        self.best_loss_g = float('inf')
        self.best_weights_g = None
        
        logging.info(f"GPUQGANTrainer initialized on {self.device}")
    
    def train_discriminator_step(self, real_data: torch.Tensor) -> tuple:
        """Single discriminator training step."""
        batch_size = real_data.size(0)
        
        self.optimizer_d.zero_grad()
        
        # Real data
        real_pred = self.discriminator(real_data)
        real_labels = torch.ones(batch_size, 1, device=self.device) * (1 - self.label_smoothing)
        loss_real = self.criterion(real_pred, real_labels)
        
        # Fake data
        z = self.generator.sample_latent(batch_size)
        with torch.no_grad():
            fake_data = self.generator(z)
        fake_pred = self.discriminator(fake_data)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        loss_fake = self.criterion(fake_pred, fake_labels)
        
        loss_d = loss_real + loss_fake
        loss_d.backward()
        
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
        
        self.optimizer_d.step()
        
        return loss_d.item(), loss_real.item(), loss_fake.item()
    
    def train_generator_step(self, batch_size: int) -> float:
        """Single generator training step."""
        self.optimizer_g.zero_grad()
        
        z = self.generator.sample_latent(batch_size)
        fake_data = self.generator(z)
        fake_pred = self.discriminator(fake_data)
        
        # Generator wants discriminator to output 1 (real)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        loss_g = self.criterion(fake_pred, real_labels)
        
        loss_g.backward()
        
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
        
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
    ) -> Dict[str, list]:
        """
        Train the QGAN for specified number of epochs.
        
        Args:
            dataloader: Training data loader
            n_epochs: Number of epochs to train
            n_critic: Number of discriminator updates per generator update
            save_interval: Epoch interval for saving samples
            sample_callback: Callback function for saving samples
            verbose: Whether to show progress bar
            
        Returns:
            Training history dictionary
        """
        logging.info(f"Starting training for {n_epochs} epochs")
        
        epoch_iterator = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
        
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
                    loss_d, loss_real, loss_fake = self.train_discriminator_step(real_images)
                
                # Train generator
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
            
            self.history['loss_d'].append(avg_loss_d)
            self.history['loss_g'].append(avg_loss_g)
            self.history['loss_d_real'].append(avg_loss_d_real)
            self.history['loss_d_fake'].append(avg_loss_d_fake)
            
            # Track best generator
            if avg_loss_g < self.best_loss_g:
                self.best_loss_g = avg_loss_g
                self.best_weights_g = {
                    k: v.clone() for k, v in self.generator.state_dict().items()
                }
            
            if verbose:
                epoch_iterator.set_postfix({
                    'D': f'{avg_loss_d:.4f}',
                    'G': f'{avg_loss_g:.4f}'
                })
            
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{n_epochs} | "
                    f"D Loss: {avg_loss_d:.4f} | G Loss: {avg_loss_g:.4f}"
                )
            
            if (epoch + 1) % save_interval == 0 and sample_callback:
                samples = self.generator.generate(16)
                sample_callback(samples, epoch + 1)
            
            # Update learning rates
            if self.use_scheduler:
                self.scheduler_g.step()
                self.scheduler_d.step()
        
        logging.info("Training complete!")
        return self.history
    
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate samples using the generator."""
        return self.generator.generate(n_samples)
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_state': self.optimizer_d.state_dict(),
            'history': self.history,
            'best_loss_g': self.best_loss_g,
            'best_weights_g': self.best_weights_g,
        }
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
        self.history = checkpoint['history']
        self.best_loss_g = checkpoint['best_loss_g']
        self.best_weights_g = checkpoint['best_weights_g']
        logging.info(f"Checkpoint loaded from {path}")

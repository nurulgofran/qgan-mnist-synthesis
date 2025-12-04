# src/training/__init__.py
"""
Training components for QGAN.
Includes trainer class, loss functions, and optimizer utilities.
"""

from .qgan_trainer import QGANTrainer
from .losses import BCEGANLoss, WassersteinLoss, get_loss_function

__all__ = ["QGANTrainer", "BCEGANLoss", "WassersteinLoss", "get_loss_function"]

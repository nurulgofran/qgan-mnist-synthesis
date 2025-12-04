# src/classical/__init__.py
"""
Classical neural network components for QGAN.
Includes discriminator architectures and baseline models.
"""

from .discriminator import (
    Discriminator4x4,
    Discriminator16x16,
    ConvDiscriminator16x16,
    get_discriminator,
)

__all__ = [
    "Discriminator4x4",
    "Discriminator16x16",
    "ConvDiscriminator16x16",
    "get_discriminator",
]

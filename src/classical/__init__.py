from .discriminator import (
    Discriminator4x4,
    Discriminator16x16,
    ConvDiscriminator16x16,
    get_discriminator,
)

from .baseline_gan import (
    ClassicalGenerator4x4,
    ClassicalGenerator16x16,
    get_classical_generator,
    ClassicalGANTrainer,
)

__all__ = [
    "Discriminator4x4",
    "Discriminator16x16",
    "ConvDiscriminator16x16",
    "get_discriminator",
    "ClassicalGenerator4x4",
    "ClassicalGenerator16x16",
    "get_classical_generator",
    "ClassicalGANTrainer",
]

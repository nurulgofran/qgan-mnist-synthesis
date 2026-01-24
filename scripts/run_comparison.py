#!/usr/bin/env python3
"""
GPU-Accelerated QGAN vs Classical GAN Comparison Script.

Uses the PyTorch-native GPU quantum generator instead of Qiskit,
providing fast comparison without Qiskit dependency issues.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.dataloader import get_dataloader
from src.quantum.gpu_quantum import create_gpu_generator
from src.classical.discriminator import get_discriminator
from src.classical.baseline_gan import get_classical_generator, ClassicalGANTrainer
from src.training.gpu_trainer import GPUQGANTrainer
from src.utils.visualization import plot_sample_grid
from src.utils.metrics import evaluate_generator, count_parameters


def setup_logging(results_dir: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(results_dir / "comparison.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def run_comparison(
    csv_path: str = "data/raw/EMNIST_16x16_Dataset.csv",
    target_digits: list = [0, 1],
    image_size: int = 4,
    n_qubits: int = 4,
    n_layers: int = 2,
    latent_dim: int = 4,
    n_epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
):
    """Run side-by-side comparison of GPU QGAN and Classical GAN."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(project_root) / "experiments" / "results" / f"gpu_comparison_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(results_dir)
    
    logger.info("=" * 60)
    logger.info("GPU QGAN vs Classical GAN Comparison")
    logger.info("=" * 60)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load data
    csv_full_path = Path(project_root) / csv_path
    dataloader = get_dataloader(
        csv_path=str(csv_full_path),
        target_digits=target_digits,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        max_samples_per_digit=500,
        normalize_range=(0, 1),
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Get real samples for evaluation
    real_samples = []
    for images, _ in dataloader:
        real_samples.append(images.numpy())
        if len(real_samples) * batch_size >= 100:
            break
    real_samples = np.concatenate(real_samples)[:100]
    
    # =========================================================================
    # CLASSICAL GAN
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Training Classical GAN")
    logger.info("=" * 40)
    
    classical_gen = get_classical_generator(
        image_size=image_size,
        latent_dim=latent_dim,
    )
    classical_disc = get_discriminator(image_size=image_size, dropout=0.3)
    
    logger.info(f"Classical Generator parameters: {count_parameters(classical_gen)}")
    logger.info(f"Classical Discriminator parameters: {count_parameters(classical_disc)}")
    
    classical_trainer = ClassicalGANTrainer(
        generator=classical_gen,
        discriminator=classical_disc,
        latent_dim=latent_dim,
        device=device,
        learning_rate_g=0.0002,
        learning_rate_d=0.0002,
    )
    
    import time
    start_time = time.time()
    classical_history = classical_trainer.train(
        dataloader=dataloader,
        n_epochs=n_epochs,
        n_critic=1,
        verbose=True,
    )
    classical_time = time.time() - start_time
    
    classical_samples = classical_trainer.generate(64)
    
    # =========================================================================
    # GPU QUANTUM GAN
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Training GPU Quantum GAN")
    logger.info("=" * 40)
    
    quantum_gen = create_gpu_generator(
        n_qubits=n_qubits,
        n_layers=n_layers,
        latent_dim=latent_dim,
        device=device,
    )
    quantum_disc = get_discriminator(image_size=image_size, dropout=0.3)
    
    logger.info(f"Quantum Generator parameters: {count_parameters(quantum_gen)}")
    logger.info(f"Quantum Discriminator parameters: {count_parameters(quantum_disc)}")
    
    quantum_trainer = GPUQGANTrainer(
        generator=quantum_gen,
        discriminator=quantum_disc,
        device=device,
        lr_g=0.01,
        lr_d=0.001,
    )
    
    start_time = time.time()
    quantum_history = quantum_trainer.train(
        dataloader=dataloader,
        n_epochs=n_epochs,
        n_critic=1,
        verbose=True,
    )
    quantum_time = time.time() - start_time
    
    quantum_samples = quantum_trainer.generate(64)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Computing Metrics")
    logger.info("=" * 40)
    
    classical_metrics = evaluate_generator(
        real_samples, classical_samples,
        loss_history_g=classical_history["loss_g"],
    )
    
    quantum_metrics = evaluate_generator(
        real_samples, quantum_samples,
        loss_history_g=quantum_history["loss_g"],
    )
    
    logger.info(f"\nClassical GAN (trained in {classical_time:.2f}s):")
    for k, v in classical_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"\nQuantum GAN (trained in {quantum_time:.2f}s):")
    for k, v in quantum_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Save results
    comparison_results = {
        "classical": {
            "metrics": {k: float(v) for k, v in classical_metrics.items()},
            "training_time_seconds": classical_time,
        },
        "quantum": {
            "metrics": {k: float(v) for k, v in quantum_metrics.items()},
            "training_time_seconds": quantum_time,
        },
        "config": {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "latent_dim": latent_dim,
            "n_epochs": n_epochs,
            "image_size": image_size,
        }
    }
    
    with open(results_dir / "comparison_results.yaml", "w") as f:
        yaml.dump(comparison_results, f)
    
    # Visualizations
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(classical_history["loss_d"], label="Discriminator", alpha=0.8)
    axes[0].plot(classical_history["loss_g"], label="Generator", alpha=0.8)
    axes[0].set_title(f"Classical GAN Training ({classical_time:.1f}s)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(quantum_history["loss_d"], label="Discriminator", alpha=0.8)
    axes[1].plot(quantum_history["loss_g"], label="Generator", alpha=0.8)
    axes[1].set_title(f"GPU Quantum GAN Training ({quantum_time:.1f}s)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {results_dir / 'training_comparison.png'}")
    
    # Sample comparison
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for i in range(8):
        axes[0, i].imshow(classical_samples[i].reshape(image_size, image_size), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Classical", fontsize=12)
        
        axes[1, i].imshow(quantum_samples[i].reshape(image_size, image_size), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Quantum", fontsize=12)
    
    plt.suptitle("Generated Samples Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "samples_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {results_dir / 'samples_comparison.png'}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Classical GAN: FID={classical_metrics.get('fid', 'N/A'):.4f}, Time={classical_time:.2f}s")
    logger.info(f"Quantum GAN:   FID={quantum_metrics.get('fid', 'N/A'):.4f}, Time={quantum_time:.2f}s")
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info("=" * 60)
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Compare GPU QGAN vs Classical GAN")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--layers", type=int, default=2, help="Circuit layers")
    parser.add_argument("--latent-dim", type=int, default=4, help="Latent dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_comparison(
        n_epochs=args.epochs,
        n_qubits=args.qubits,
        n_layers=args.layers,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

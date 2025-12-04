#!/usr/bin/env python3

import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.dataloader import get_dataloader
from src.quantum.ansatz import create_generator_ansatz, initialize_parameters
from src.quantum.encoding import create_generator_qnn
from src.classical.discriminator import get_discriminator
from src.training.qgan_trainer import QGANTrainer
from src.utils.visualization import (
    plot_sample_grid,
    plot_training_curves,
    plot_real_vs_fake_comparison,
    create_sample_callback,
)
from src.utils.metrics import evaluate_generator


def setup_logging(results_dir: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(results_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(config_path: str):
    config = load_config(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config["experiment"]["name"]
    results_dir = (
        Path(project_root) / "experiments" / "results" / f"{exp_name}_{timestamp}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(results_dir)

    logger.info("=" * 60)
    logger.info(f"QGAN Experiment: {exp_name}")
    logger.info("=" * 60)

    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    seed = config["experiment"].get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_config = config["data"]
    csv_path = Path(project_root) / data_config["csv_path"]

    dataloader = get_dataloader(
        csv_path=str(csv_path),
        target_digits=data_config["target_digits"],
        target_size=tuple(data_config["target_size"]),
        batch_size=data_config["batch_size"],
        max_samples_per_digit=data_config.get("samples_per_digit"),
        normalize_range=tuple(data_config["normalize_range"]),
    )

    img_size = data_config["target_size"][0]
    n_pixels = img_size * img_size

    gen_config = config["generator"]
    n_qubits = gen_config["n_qubits"]
    n_layers = gen_config["n_layers"]
    latent_dim = gen_config["latent_dim"]
    entanglement = gen_config.get("entanglement", "linear")

    ansatz, latent_params, trainable_params = create_generator_ansatz(
        n_qubits=n_qubits,
        n_layers=n_layers,
        latent_dim=latent_dim,
        entanglement=entanglement,
    )

    qnn = create_generator_qnn(ansatz, latent_params, trainable_params)

    disc_config = config["discriminator"]
    discriminator = get_discriminator(
        image_size=img_size,
        architecture=disc_config.get("type", "mlp"),
        dropout=disc_config.get("dropout", 0.3),
    )

    train_config = config["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = QGANTrainer(
        generator_qnn=qnn,
        discriminator=discriminator,
        latent_dim=latent_dim,
        device=device,
        learning_rate_g=train_config["learning_rate_g"],
        learning_rate_d=train_config["learning_rate_d"],
    )

    samples_dir = results_dir / "samples"
    sample_callback = create_sample_callback(
        save_dir=str(samples_dir), img_shape=(img_size, img_size)
    )

    logger.info(f"Starting training for {train_config['n_epochs']} epochs")

    history = trainer.train(
        dataloader=dataloader,
        n_epochs=train_config["n_epochs"],
        n_critic=train_config["n_critic"],
        save_interval=train_config["save_interval"],
        sample_callback=sample_callback,
        verbose=True,
    )

    np.savez(results_dir / "history.npz", **history)

    plot_training_curves(
        history, save_path=str(results_dir / "training_curves.png"), show=False
    )

    final_samples = trainer.generate(64)
    np.save(results_dir / "final_samples.npy", final_samples)

    plot_sample_grid(
        final_samples[:16],
        img_shape=(img_size, img_size),
        n_rows=4,
        n_cols=4,
        title=f"Final Generated Samples ({exp_name})",
        save_path=str(results_dir / "final_samples.png"),
        show=False,
    )

    trainer.save_checkpoint(str(results_dir / "checkpoint.pt"))

    real_samples = []
    for images, _ in dataloader:
        real_samples.append(images.numpy())
        if len(real_samples) * dataloader.batch_size >= 100:
            break
    real_samples = np.concatenate(real_samples)[:100]

    plot_real_vs_fake_comparison(
        real_samples[:8],
        final_samples[:8],
        img_shape=(img_size, img_size),
        n_samples=8,
        save_path=str(results_dir / "real_vs_fake.png"),
        show=False,
    )

    eval_config = config.get("evaluation", {})
    if eval_config.get("compute_coverage", True):
        metrics = evaluate_generator(
            real_samples,
            final_samples,
            loss_history_g=history["loss_g"],
            n_clusters=min(10, len(np.unique(dataloader.dataset.labels))),
        )

        with open(results_dir / "metrics.yaml", "w") as f:
            yaml.dump({k: float(v) for k, v in metrics.items()}, f)

        logger.info("Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    logger.info("=" * 60)
    logger.info("Experiment complete!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("=" * 60)

    return results_dir


def main():
    parser = argparse.ArgumentParser(description="Run QGAN experiment")
    parser.add_argument(
        "config",
        nargs="?",
        default="experiments/configs/experiment_001.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config_path = Path(project_root) / args.config

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating default config...")

        config_path.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            "experiment": {"name": "qgan_4qubit_baseline", "seed": 42},
            "data": {
                "csv_path": "data/raw/EMNIST_16x16_Dataset.csv",
                "target_digits": [0, 1],
                "target_size": [4, 4],
                "samples_per_digit": 500,
                "batch_size": 32,
                "normalize_range": [0, 1],
            },
            "generator": {
                "n_qubits": 4,
                "n_layers": 2,
                "latent_dim": 4,
                "entanglement": "linear",
            },
            "discriminator": {"type": "mlp", "dropout": 0.3},
            "training": {
                "n_epochs": 50,
                "learning_rate_g": 0.01,
                "learning_rate_d": 0.001,
                "n_critic": 1,
                "save_interval": 10,
            },
            "evaluation": {"compute_fid": False, "compute_coverage": True},
        }

        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        print(f"Created: {config_path}")

    run_experiment(str(config_path))


if __name__ == "__main__":
    main()

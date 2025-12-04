#!/usr/bin/env python3

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from pathlib import Path

from data.dataloader import EMNISTQuantumDataset, prepare_amplitude_encoding_data


def prepare_data():
    raw_path = Path(project_root) / "data" / "raw" / "EMNIST_16x16_Dataset.csv"
    processed_dir = Path(project_root) / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QGAN Data Preparation")
    print("=" * 60)

    if not raw_path.exists():
        print(f"\nError: Dataset not found at {raw_path}")
        return

    print("\nPreparing 4-qubit dataset (4x4 images, digits 0-1)...")
    dataset_4q = EMNISTQuantumDataset(
        csv_path=str(raw_path),
        target_digits=[0, 1],
        target_size=(4, 4),
        normalize_range=(0, 1),
        max_samples_per_digit=500,
    )
    save_path_4q = processed_dir / "mnist_4x4_binary.npz"
    np.savez(save_path_4q, images=dataset_4q.images, labels=dataset_4q.labels)
    print(f"Saved: {save_path_4q}")

    print("\nPreparing 4-qubit dataset with angle encoding...")
    dataset_4q_angle = EMNISTQuantumDataset(
        csv_path=str(raw_path),
        target_digits=[0, 1],
        target_size=(4, 4),
        normalize_range=(0, np.pi),
        max_samples_per_digit=500,
    )
    save_path_4q_angle = processed_dir / "mnist_4x4_angle.npz"
    np.savez(
        save_path_4q_angle,
        images=dataset_4q_angle.images,
        labels=dataset_4q_angle.labels,
    )
    print(f"Saved: {save_path_4q_angle}")

    print("\nPreparing 8-qubit dataset (16x16 images, digits 0-9)...")
    dataset_8q = EMNISTQuantumDataset(
        csv_path=str(raw_path),
        target_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        target_size=(16, 16),
        normalize_range=(0, 1),
        max_samples_per_digit=200,
    )
    save_path_8q = processed_dir / "mnist_16x16_full.npz"
    np.savez(save_path_8q, images=dataset_8q.images, labels=dataset_8q.labels)
    print(f"Saved: {save_path_8q}")

    print("\nPreparing 8-qubit binary dataset (16x16 images, digits 0-1)...")
    dataset_8q_binary = EMNISTQuantumDataset(
        csv_path=str(raw_path),
        target_digits=[0, 1],
        target_size=(16, 16),
        normalize_range=(0, 1),
        max_samples_per_digit=500,
    )
    save_path_8q_binary = processed_dir / "mnist_16x16_binary.npz"
    np.savez(
        save_path_8q_binary,
        images=dataset_8q_binary.images,
        labels=dataset_8q_binary.labels,
    )
    print(f"Saved: {save_path_8q_binary}")

    print("\nPreparing amplitude encoding data...")
    amp_data = prepare_amplitude_encoding_data(dataset_8q.images)
    save_path_amp = processed_dir / "mnist_16x16_amplitude.npz"
    np.savez(save_path_amp, images=amp_data, labels=dataset_8q.labels)
    print(f"Saved: {save_path_amp}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()

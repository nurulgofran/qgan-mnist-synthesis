import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List
import os


def plot_sample_grid(
    images: np.ndarray,
    img_shape: Tuple[int, int],
    n_rows: int = 4,
    n_cols: int = 4,
    title: str = "Generated Samples",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
):
    if figsize is None:
        figsize = (2.5 * n_cols, 2.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]

            if idx < len(images):
                img = images[idx].reshape(img_shape)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)

            ax.axis("off")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    history: dict, save_path: Optional[str] = None, show: bool = True
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss_d"]) + 1)

    ax1 = axes[0]
    ax1.plot(epochs, history["loss_d"], label="Discriminator", alpha=0.8, color="blue")
    ax1.plot(epochs, history["loss_g"], label="Generator", alpha=0.8, color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    window = min(10, len(epochs) // 5) if len(epochs) > 10 else 1

    if window > 1:
        kernel = np.ones(window) / window
        d_smooth = np.convolve(history["loss_d"], kernel, mode="valid")
        g_smooth = np.convolve(history["loss_g"], kernel, mode="valid")
        smooth_epochs = range(window, len(d_smooth) + window)

        ax2.plot(
            smooth_epochs, d_smooth, label="D (smoothed)", linewidth=2, color="blue"
        )
        ax2.plot(
            smooth_epochs, g_smooth, label="G (smoothed)", linewidth=2, color="orange"
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (Moving Average)")
        ax2.set_title(f"Smoothed Loss ({window}-epoch window)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "Not enough epochs\nfor smoothing",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Smoothed Loss")

    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_real_vs_fake_comparison(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    img_shape: Tuple[int, int],
    n_samples: int = 8,
    save_path: Optional[str] = None,
    show: bool = True,
):
    n_samples = min(n_samples, len(real_images), len(fake_images))

    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))

    for i in range(n_samples):
        ax_real = axes[0, i] if n_samples > 1 else axes[0]
        img_real = real_images[i].reshape(img_shape)
        ax_real.imshow(img_real, cmap="gray", vmin=0, vmax=1)
        ax_real.axis("off")
        if i == 0:
            ax_real.set_title("Real", fontsize=12)

        ax_fake = axes[1, i] if n_samples > 1 else axes[1]
        img_fake = fake_images[i].reshape(img_shape)
        ax_fake.imshow(img_fake, cmap="gray", vmin=0, vmax=1)
        ax_fake.axis("off")
        if i == 0:
            ax_fake.set_title("Generated", fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_components(
    history: dict, save_path: Optional[str] = None, show: bool = True
):
    if "loss_d_real" not in history or "loss_d_fake" not in history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history["loss_d_real"]) + 1)

    ax.plot(epochs, history["loss_d_real"], label="D(real)", alpha=0.8)
    ax.plot(epochs, history["loss_d_fake"], label="D(fake)", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Equilibrium")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Component")
    ax.set_title("Discriminator Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_progress(
    samples_over_time: List[np.ndarray],
    epochs: List[int],
    img_shape: Tuple[int, int],
    n_samples_per_epoch: int = 8,
    save_path: Optional[str] = None,
    show: bool = True,
):
    n_epochs = len(samples_over_time)
    n_samples = min(n_samples_per_epoch, min(len(s) for s in samples_over_time))

    fig, axes = plt.subplots(n_epochs, n_samples, figsize=(2 * n_samples, 2 * n_epochs))

    if n_epochs == 1:
        axes = axes.reshape(1, -1)
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, (samples, epoch) in enumerate(zip(samples_over_time, epochs)):
        for j in range(n_samples):
            ax = axes[i, j]
            img = samples[j].reshape(img_shape)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

            if j == 0:
                ax.set_ylabel(f"Epoch {epoch}", rotation=0, ha="right", va="center")

    plt.suptitle("Training Progress", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def create_sample_callback(
    save_dir: str, img_shape: Tuple[int, int], n_samples: int = 16
):
    os.makedirs(save_dir, exist_ok=True)
    samples_history = []
    epochs_history = []

    def callback(samples, epoch):
        save_path = os.path.join(save_dir, f"samples_epoch_{epoch:04d}.png")
        n_cols = int(np.ceil(np.sqrt(n_samples)))
        n_rows = int(np.ceil(n_samples / n_cols))

        plot_sample_grid(
            samples[:n_samples],
            img_shape,
            n_rows=n_rows,
            n_cols=n_cols,
            title=f"Generated Samples (Epoch {epoch})",
            save_path=save_path,
            show=False,
        )

        samples_history.append(samples[:8])
        epochs_history.append(epoch)

        if epoch > 0 and epoch % 50 == 0:
            progress_path = os.path.join(save_dir, f"progress_epoch_{epoch:04d}.png")
            plot_training_progress(
                samples_history[-5:],
                epochs_history[-5:],
                img_shape,
                n_samples_per_epoch=4,
                save_path=progress_path,
                show=False,
            )

    return callback

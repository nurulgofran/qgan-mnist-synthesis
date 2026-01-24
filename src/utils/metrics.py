import numpy as np
from scipy import linalg
from typing import Tuple, Optional
from sklearn.cluster import KMeans


def calculate_fid(
    real_images: np.ndarray, generated_images: np.ndarray, use_pixel_space: bool = True
) -> float:
    real_flat = real_images.reshape(real_images.shape[0], -1)
    gen_flat = generated_images.reshape(generated_images.shape[0], -1)

    mu_real, sigma_real = _compute_statistics(real_flat)
    mu_gen, sigma_gen = _compute_statistics(gen_flat)

    diff = mu_real - mu_gen

    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return float(fid)


def _compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    if sigma.ndim == 0:
        sigma = np.array([[sigma]])

    return mu, sigma


def calculate_pixel_mse(
    real_images: np.ndarray, generated_images: np.ndarray
) -> Tuple[float, float]:
    real_flat = real_images.reshape(real_images.shape[0], -1)
    gen_flat = generated_images.reshape(generated_images.shape[0], -1)

    real_mean = np.mean(real_flat, axis=0)
    gen_mean = np.mean(gen_flat, axis=0)

    squared_errors = (real_mean - gen_mean) ** 2

    mse = np.mean(squared_errors)
    std = np.std(squared_errors)

    return float(mse), float(std)


def calculate_mode_coverage(
    generated_images: np.ndarray, n_clusters: int = 10, random_state: int = 42
) -> float:
    flat = generated_images.reshape(generated_images.shape[0], -1)

    if len(flat) < n_clusters:
        return 1.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(flat)

    unique_clusters = len(np.unique(labels))
    coverage = unique_clusters / n_clusters

    return float(coverage)


def training_stability_score(loss_history: list, window: int = 20) -> float:
    if len(loss_history) < window:
        return float("inf")

    recent_losses = loss_history[-window:]
    variance = np.var(recent_losses)

    return float(variance)


def calculate_diversity_score(generated_images: np.ndarray) -> float:
    flat = generated_images.reshape(generated_images.shape[0], -1)
    n_samples = len(flat)

    if n_samples < 2:
        return 0.0

    max_pairs = min(1000, n_samples * (n_samples - 1) // 2)

    distances = []
    for _ in range(max_pairs):
        i, j = np.random.choice(n_samples, 2, replace=False)
        dist = np.linalg.norm(flat[i] - flat[j])
        distances.append(dist)

    return float(np.mean(distances))


def evaluate_generator(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    loss_history_g: Optional[list] = None,
    n_clusters: int = 10,
) -> dict:
    results = {}

    results["fid"] = calculate_fid(real_images, generated_images)

    mse, mse_std = calculate_pixel_mse(real_images, generated_images)
    results["pixel_mse"] = mse
    results["pixel_mse_std"] = mse_std

    results["mode_coverage"] = calculate_mode_coverage(
        generated_images, n_clusters=n_clusters
    )

    results["diversity"] = calculate_diversity_score(generated_images)

    if loss_history_g is not None:
        results["training_stability"] = training_stability_score(loss_history_g)

    return results


def count_parameters(model) -> int:
    """
    Count trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch nn.Module
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

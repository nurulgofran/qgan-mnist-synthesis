from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.gradients import ParamShiftSamplerGradient
import numpy as np
from typing import Tuple, Optional


def create_generator_qnn(
    ansatz: QuantumCircuit,
    latent_params: ParameterVector,
    trainable_params: ParameterVector,
    output_mode: str = "probabilities",
) -> SamplerQNN:
    if output_mode != "probabilities":
        raise ValueError(f"Unsupported output_mode: {output_mode}")

    sampler = StatevectorSampler()
    gradient = ParamShiftSamplerGradient(sampler=sampler)

    def interpret(x):
        return x

    output_shape = (2**ansatz.num_qubits,)

    qnn = SamplerQNN(
        circuit=ansatz,
        input_params=list(latent_params),
        weight_params=list(trainable_params),
        interpret=interpret,
        output_shape=output_shape,
        gradient=gradient,
    )

    return qnn


def probabilities_to_image(
    probs: np.ndarray, img_shape: Tuple[int, int], rescale: bool = True
) -> np.ndarray:
    is_batch = probs.ndim > 1

    if is_batch:
        batch_size = probs.shape[0]
        images = []
        for i in range(batch_size):
            img = probs[i].reshape(img_shape)
            if rescale:
                img = _rescale_to_01(img)
            images.append(img)
        return np.array(images)
    else:
        img = probs.reshape(img_shape)
        if rescale:
            img = _rescale_to_01(img)
        return img


def _rescale_to_01(img: np.ndarray) -> np.ndarray:
    img_min = img.min()
    img_max = img.max()

    if img_max - img_min < 1e-8:
        return np.full_like(img, 0.5)

    return (img - img_min) / (img_max - img_min)


def image_to_amplitudes(images: np.ndarray, normalize: bool = True) -> np.ndarray:
    if images.ndim == 3:
        flat = images.reshape(images.shape[0], -1)
    else:
        flat = images

    if normalize:
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        flat = flat / norms

    return flat.astype(np.float32)


def get_qnn_info(qnn: SamplerQNN) -> dict:
    return {
        "num_inputs": qnn.num_inputs,
        "num_weights": qnn.num_weights,
        "output_shape": qnn.output_shape,
        "sparse": qnn.sparse,
    }

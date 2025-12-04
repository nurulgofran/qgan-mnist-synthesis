from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np
from typing import Tuple, Literal


def create_generator_ansatz(
    n_qubits: int,
    n_layers: int,
    latent_dim: int,
    entanglement: Literal["linear", "circular", "full"] = "linear",
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    n_trainable_params = n_layers * 2 * n_qubits
    latent_params = ParameterVector("z", latent_dim)
    trainable_params = ParameterVector("θ", n_trainable_params)

    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for i in range(n_qubits):
        latent_idx = i % latent_dim
        qc.ry(latent_params[latent_idx], i)

    qc.barrier()

    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(trainable_params[param_idx], i)
            param_idx += 1

        for i in range(n_qubits):
            qc.rz(trainable_params[param_idx], i)
            param_idx += 1

        if entanglement == "linear":
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
        elif entanglement == "circular":
            for i in range(n_qubits):
                qc.cx(i, (i + 1) % n_qubits)
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qc.cx(i, j)

        qc.barrier()

    return qc, latent_params, trainable_params


def create_strongly_entangling_ansatz(
    n_qubits: int, n_layers: int
) -> Tuple[QuantumCircuit, ParameterVector]:
    n_params = n_layers * n_qubits * 3
    params = ParameterVector("θ", n_params)

    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.rz(params[param_idx], i)
            qc.ry(params[param_idx + 1], i)
            qc.rz(params[param_idx + 2], i)
            param_idx += 3

        for i in range(n_qubits):
            target = (i + 1 + layer) % n_qubits
            if target != i:
                qc.cx(i, target)

        qc.barrier()

    return qc, params


def create_minimal_ansatz(
    n_qubits: int, latent_dim: int
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    latent_params = ParameterVector("z", latent_dim)
    trainable_params = ParameterVector("θ", n_qubits)

    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        latent_idx = i % latent_dim
        qc.ry(latent_params[latent_idx], i)

    qc.barrier()

    for i in range(n_qubits):
        qc.ry(trainable_params[i], i)

    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    return qc, latent_params, trainable_params


def initialize_parameters(n_params: int, strategy: str = "small_random") -> np.ndarray:
    if strategy == "small_random":
        return np.random.uniform(-0.1, 0.1, size=n_params)
    elif strategy == "zero":
        return np.zeros(n_params)
    elif strategy == "uniform_pi":
        return np.random.uniform(0, 2 * np.pi, size=n_params)
    elif strategy == "normal":
        return np.random.normal(0, 0.1, size=n_params)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def count_circuit_parameters(n_qubits: int, n_layers: int, latent_dim: int) -> dict:
    n_trainable = n_layers * 2 * n_qubits
    n_latent = latent_dim
    output_dim = 2**n_qubits

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "latent_dim": latent_dim,
        "trainable_params": n_trainable,
        "latent_params": n_latent,
        "total_params": n_trainable + n_latent,
        "output_dim": output_dim,
        "output_shape": (int(np.sqrt(output_dim)), int(np.sqrt(output_dim))),
    }

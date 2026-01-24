"""
GPU-Accelerated Quantum Circuit Simulator using PyTorch.

This module implements a differentiable quantum circuit simulator that runs
entirely on GPU using PyTorch tensors. It's significantly faster than
CPU-based Qiskit simulation for small qubit counts (4-8 qubits).

For 4 qubits: statevector has 2^4 = 16 complex amplitudes
For 8 qubits: statevector has 2^8 = 256 complex amplitudes

Both fit easily in GPU memory and benefit from massive parallelization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import math


class QuantumGates:
    """Collection of quantum gate matrices as PyTorch tensors."""
    
    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def identity(device=None):
        if device is None:
            device = QuantumGates.get_device()
        return torch.eye(2, dtype=torch.complex64, device=device)
    
    @staticmethod
    def pauli_x(device=None):
        if device is None:
            device = QuantumGates.get_device()
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    
    @staticmethod
    def pauli_y(device=None):
        if device is None:
            device = QuantumGates.get_device()
        return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    
    @staticmethod
    def pauli_z(device=None):
        if device is None:
            device = QuantumGates.get_device()
        return torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    
    @staticmethod
    def hadamard(device=None):
        if device is None:
            device = QuantumGates.get_device()
        return torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=device) / math.sqrt(2)
    
    @staticmethod
    def rx(theta: torch.Tensor):
        """Rotation around X-axis."""
        device = theta.device
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        # Handle batched theta
        if theta.dim() == 0:
            return torch.stack([
                torch.stack([cos, -1j * sin]),
                torch.stack([-1j * sin, cos])
            ]).to(dtype=torch.complex64, device=device)
        else:
            batch_size = theta.shape[0]
            gates = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
            gates[:, 0, 0] = cos
            gates[:, 0, 1] = -1j * sin
            gates[:, 1, 0] = -1j * sin
            gates[:, 1, 1] = cos
            return gates
    
    @staticmethod
    def ry(theta: torch.Tensor):
        """Rotation around Y-axis."""
        device = theta.device
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        if theta.dim() == 0:
            return torch.stack([
                torch.stack([cos, -sin]),
                torch.stack([sin, cos])
            ]).to(dtype=torch.complex64, device=device)
        else:
            batch_size = theta.shape[0]
            gates = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
            gates[:, 0, 0] = cos
            gates[:, 0, 1] = -sin
            gates[:, 1, 0] = sin
            gates[:, 1, 1] = cos
            return gates
    
    @staticmethod
    def rz(theta: torch.Tensor):
        """Rotation around Z-axis."""
        device = theta.device
        if theta.dim() == 0:
            return torch.diag(torch.stack([
                torch.exp(-1j * theta / 2),
                torch.exp(1j * theta / 2)
            ])).to(dtype=torch.complex64, device=device)
        else:
            batch_size = theta.shape[0]
            gates = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
            gates[:, 0, 0] = torch.exp(-1j * theta / 2)
            gates[:, 1, 1] = torch.exp(1j * theta / 2)
            return gates
    
    @staticmethod
    def cnot(device=None):
        """CNOT (CX) gate."""
        if device is None:
            device = QuantumGates.get_device()
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64, device=device)
    
    @staticmethod
    def cz(device=None):
        """CZ gate."""
        if device is None:
            device = QuantumGates.get_device()
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=torch.complex64, device=device)


class GPUQuantumCircuit(nn.Module):
    """
    A differentiable quantum circuit that runs on GPU.
    
    Implements a variational quantum circuit with:
    - Parameterized rotation gates (RX, RY, RZ)
    - Entangling layers (CNOT)
    - Measurement in computational basis (probabilities)
    
    The entire forward pass is differentiable, allowing gradient-based
    optimization with PyTorch autograd.
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        latent_dim: int,
        entanglement: str = 'linear',
        device: str = 'cuda'
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.entanglement = entanglement
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.dim = 2 ** n_qubits  # Hilbert space dimension
        
        # Number of parameters: 3 rotations per qubit per layer (RY, RZ, RY)
        # Plus latent encoding parameters
        n_trainable_params = n_layers * n_qubits * 3
        
        # Trainable parameters (weights)
        self.weights = nn.Parameter(
            torch.randn(n_trainable_params, device=self.device) * 0.1
        )
        
        self._precompute_entanglement_gates()
    
    def _precompute_entanglement_gates(self):
        """Pre-compute the full entanglement unitary for efficiency."""
        # For linear entanglement: CNOT chain 0->1, 1->2, ..., (n-1)->0
        # We'll apply these as 2-qubit gates
        self.entangle_pairs = []
        if self.entanglement == 'linear':
            for i in range(self.n_qubits - 1):
                self.entangle_pairs.append((i, i + 1))
        elif self.entanglement == 'circular':
            for i in range(self.n_qubits):
                self.entangle_pairs.append((i, (i + 1) % self.n_qubits))
        elif self.entanglement == 'full':
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    self.entangle_pairs.append((i, j))
    
    def _apply_single_qubit_gate(
        self,
        state: torch.Tensor,
        gate: torch.Tensor,
        qubit: int
    ) -> torch.Tensor:
        """
        Apply a single-qubit gate to the statevector.
        
        Args:
            state: Complex tensor of shape (batch, 2^n)
            gate: Single-qubit gate (2, 2) or batched (batch, 2, 2)
            qubit: Target qubit index (0-indexed)
            
        Returns:
            New statevector after applying gate
        """
        batch_size = state.shape[0]
        n = self.n_qubits
        
        # Reshape state to separate target qubit
        # state: (batch, 2^n) -> (batch, 2^(n-q-1), 2, 2^q)
        state = state.reshape(batch_size, 2**(n - qubit - 1), 2, 2**qubit)
        
        # Apply gate
        if gate.dim() == 2:
            # Non-batched gate
            state = torch.einsum('ij,bkjl->bkil', gate, state)
        else:
            # Batched gate (batch, 2, 2)
            state = torch.einsum('bij,bkjl->bkil', gate, state)
        
        # Reshape back
        state = state.reshape(batch_size, self.dim)
        return state
    
    def _apply_cnot(
        self,
        state: torch.Tensor,
        control: int,
        target: int
    ) -> torch.Tensor:
        """
        Apply CNOT gate between control and target qubits.
        
        Uses an efficient implementation that avoids building the full matrix.
        """
        batch_size = state.shape[0]
        n = self.n_qubits
        
        # Create index arrays for the bit flip
        indices = torch.arange(self.dim, device=self.device)
        
        # Identify states where control qubit is |1>
        control_mask = (indices >> (n - control - 1)) & 1
        
        # For those states, flip the target qubit
        flipped_indices = indices.clone()
        flipped_indices[control_mask == 1] = indices[control_mask == 1] ^ (1 << (n - target - 1))
        
        # Apply the permutation
        new_state = state[:, flipped_indices]
        
        return new_state
    
    def _encode_latent(
        self,
        latent: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Encode latent variables into the quantum state."""
        # Use RY gates to encode latent variables
        for i in range(min(self.latent_dim, self.n_qubits)):
            theta = latent[:, i] * np.pi  # Scale to [0, pi] if latent is in [0, 1]
            ry_gate = QuantumGates.ry(theta)
            state = self._apply_single_qubit_gate(state, ry_gate, i)
        return state
    
    def _apply_variational_layer(
        self,
        state: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Apply one layer of the variational ansatz."""
        param_offset = layer_idx * self.n_qubits * 3
        batch_size = state.shape[0]
        
        # Apply RY-RZ-RY rotation to each qubit
        for q in range(self.n_qubits):
            base_idx = param_offset + q * 3
            
            # RY rotation
            theta1 = self.weights[base_idx].expand(batch_size)
            ry1 = QuantumGates.ry(theta1)
            state = self._apply_single_qubit_gate(state, ry1, q)
            
            # RZ rotation
            theta2 = self.weights[base_idx + 1].expand(batch_size)
            rz = QuantumGates.rz(theta2)
            state = self._apply_single_qubit_gate(state, rz, q)
            
            # RY rotation
            theta3 = self.weights[base_idx + 2].expand(batch_size)
            ry2 = QuantumGates.ry(theta3)
            state = self._apply_single_qubit_gate(state, ry2, q)
        
        # Apply entangling layer
        for control, target in self.entangle_pairs:
            state = self._apply_cnot(state, control, target)
        
        return state
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: generate output from latent noise.
        
        Args:
            latent: Tensor of shape (batch_size, latent_dim)
            
        Returns:
            Probabilities of shape (batch_size, 2^n_qubits)
        """
        batch_size = latent.shape[0]
        latent = latent.to(self.device)
        
        # Initialize |0...0> state
        state = torch.zeros(batch_size, self.dim, dtype=torch.complex64, device=self.device)
        state[:, 0] = 1.0
        
        # Encode latent variables
        state = self._encode_latent(latent, state)
        
        # Apply variational layers
        for layer in range(self.n_layers):
            state = self._apply_variational_layer(state, layer)
        
        # Compute probabilities (Born rule)
        probabilities = torch.abs(state) ** 2
        
        return probabilities
    
    def generate_images(
        self,
        latent: torch.Tensor,
        img_shape: Tuple[int, int] = (4, 4)
    ) -> torch.Tensor:
        """
        Generate images from latent vectors.
        
        Args:
            latent: Tensor of shape (batch_size, latent_dim)
            img_shape: Output image shape (must match 2^n_qubits pixels)
            
        Returns:
            Images of shape (batch_size, H, W) normalized to [0, 1]
        """
        probs = self.forward(latent)
        
        # Reshape to images
        batch_size = probs.shape[0]
        images = probs.view(batch_size, *img_shape)
        
        # Normalize each image to [0, 1]
        images_min = images.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        images_max = images.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        
        images = (images - images_min) / (images_max - images_min + 1e-8)
        
        return images


class GPUQuantumGenerator(nn.Module):
    """
    High-level wrapper for the GPU quantum generator.
    
    This class provides a simple interface compatible with the existing
    QGAN trainer, outputting flattened image tensors.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        latent_dim: int = 4,
        img_shape: Tuple[int, int] = (4, 4),
        device: str = 'cuda'
    ):
        super().__init__()
        self.img_shape = img_shape
        self.n_pixels = img_shape[0] * img_shape[1]
        
        assert 2 ** n_qubits == self.n_pixels, \
            f"2^{n_qubits} = {2**n_qubits} must equal {self.n_pixels} pixels"
        
        self.circuit = GPUQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            latent_dim=latent_dim,
            entanglement='linear',
            device=device
        )
        
        self.latent_dim = latent_dim
        self.device = self.circuit.device
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate flattened images from latent vectors.
        
        Args:
            z: Latent noise of shape (batch_size, latent_dim)
            
        Returns:
            Flattened images of shape (batch_size, n_pixels)
        """
        images = self.circuit.generate_images(z, self.img_shape)
        return images.view(-1, self.n_pixels)
    
    def sample_latent(self, batch_size: int) -> torch.Tensor:
        """Sample random latent vectors."""
        return torch.randn(batch_size, self.latent_dim, device=self.device)
    
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate samples as numpy array."""
        self.eval()
        with torch.no_grad():
            z = self.sample_latent(n_samples)
            samples = self.forward(z)
        return samples.cpu().numpy()


def create_gpu_generator(
    n_qubits: int = 4,
    n_layers: int = 2,
    latent_dim: int = 4,
    device: str = 'cuda'
) -> GPUQuantumGenerator:
    """
    Factory function to create a GPU quantum generator.
    
    Args:
        n_qubits: Number of qubits (2^n_qubits = number of pixels)
        n_layers: Number of variational layers
        latent_dim: Dimension of latent noise vector
        device: Device ('cuda' or 'cpu')
        
    Returns:
        GPUQuantumGenerator instance
    """
    img_size = int(np.sqrt(2 ** n_qubits))
    img_shape = (img_size, img_size)
    
    return GPUQuantumGenerator(
        n_qubits=n_qubits,
        n_layers=n_layers,
        latent_dim=latent_dim,
        img_shape=img_shape,
        device=device
    )




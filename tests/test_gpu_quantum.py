"""
Unit tests for GPU quantum components.

Run with: pytest tests/test_gpu_quantum.py -v
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quantum.gpu_quantum import (
    GPUQuantumCircuit,
    GPUQuantumGenerator,
    create_gpu_generator,
    QuantumGates,
)
from src.classical.discriminator import get_discriminator, count_parameters
from src.classical.baseline_gan import get_classical_generator


class TestQuantumGates:
    """Tests for quantum gate implementations."""

    def test_pauli_gates(self):
        """Test Pauli gate matrices."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x = QuantumGates.pauli_x(device)
        y = QuantumGates.pauli_y(device)
        z = QuantumGates.pauli_z(device)
        
        assert x.shape == (2, 2)
        assert y.shape == (2, 2)
        assert z.shape == (2, 2)

    def test_rotation_gates(self):
        """Test parameterized rotation gates."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        theta = torch.tensor(np.pi / 4, device=device)
        
        rx = QuantumGates.rx(theta)
        ry = QuantumGates.ry(theta)
        rz = QuantumGates.rz(theta)
        
        assert rx.shape == (2, 2)
        assert ry.shape == (2, 2)
        assert rz.shape == (2, 2)

    def test_batched_rotation(self):
        """Test batched rotation gates."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        theta = torch.randn(8, device=device)
        
        ry = QuantumGates.ry(theta)
        
        assert ry.shape == (8, 2, 2)


class TestGPUQuantumCircuit:
    """Tests for GPU quantum circuit."""

    def test_initialization(self):
        """Test circuit initialization."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        circuit = GPUQuantumCircuit(
            n_qubits=4,
            n_layers=2,
            latent_dim=4,
            device=device
        )
        
        assert circuit.n_qubits == 4
        assert circuit.n_layers == 2
        assert circuit.dim == 16  # 2^4

    def test_forward_shape(self):
        """Test forward pass output shape."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        circuit = GPUQuantumCircuit(
            n_qubits=4,
            n_layers=2,
            latent_dim=4,
            device=device
        )
        
        latent = torch.randn(8, 4, device=device)
        output = circuit(latent)
        
        assert output.shape == (8, 16)

    def test_output_is_probability(self):
        """Test that output is valid probability distribution."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        circuit = GPUQuantumCircuit(
            n_qubits=4,
            n_layers=2,
            latent_dim=4,
            device=device
        )
        
        latent = torch.randn(8, 4, device=device)
        output = circuit(latent)
        
        # Should sum to 1 (probabilities)
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        
        # Should be non-negative
        assert torch.all(output >= 0)

    def test_gradient_flow(self):
        """Test that gradients flow through circuit."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        circuit = GPUQuantumCircuit(
            n_qubits=4,
            n_layers=2,
            latent_dim=4,
            device=device
        )
        
        latent = torch.randn(8, 4, device=device)
        output = circuit(latent)
        loss = output.sum()
        loss.backward()
        
        assert circuit.weights.grad is not None
        assert not torch.all(circuit.weights.grad == 0)


class TestGPUQuantumGenerator:
    """Tests for GPU quantum generator wrapper."""

    def test_create_generator(self):
        """Test generator factory function."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        gen = create_gpu_generator(
            n_qubits=4,
            n_layers=2,
            latent_dim=4,
            device=device
        )
        
        assert isinstance(gen, GPUQuantumGenerator)
        assert gen.n_pixels == 16

    def test_forward_output(self):
        """Test generator forward pass."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        gen = create_gpu_generator(n_qubits=4, n_layers=2, latent_dim=4, device=device)
        
        z = gen.sample_latent(16)
        output = gen(z)
        
        assert output.shape == (16, 16)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_generate_numpy(self):
        """Test numpy generation method."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        gen = create_gpu_generator(n_qubits=4, n_layers=2, latent_dim=4, device=device)
        
        samples = gen.generate(32)
        
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (32, 16)


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_generator_discriminator_compatibility(self):
        """Test GPU quantum generator with classical discriminator."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        gen = create_gpu_generator(n_qubits=4, n_layers=2, latent_dim=4, device=device)
        disc = get_discriminator(image_size=4, dropout=0.3).to(device)
        
        z = gen.sample_latent(8)
        fake_data = gen(z)
        prediction = disc(fake_data)
        
        assert prediction.shape == (8, 1)

    def test_training_step(self):
        """Test a single training step."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        gen = create_gpu_generator(n_qubits=4, n_layers=2, latent_dim=4, device=device)
        disc = get_discriminator(image_size=4, dropout=0.3).to(device)
        
        optimizer_g = torch.optim.Adam(gen.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Generate fake data
        z = gen.sample_latent(8)
        fake_data = gen(z)
        fake_pred = disc(fake_data)
        
        # Compute loss
        real_labels = torch.ones(8, 1, device=device)
        loss = criterion(fake_pred, real_labels)
        
        # Backward pass
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()
        
        # Check weights updated
        assert gen.circuit.weights.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# QGAN MNIST Synthesis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0+-blueviolet.svg)](https://qiskit.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

A **Quantum Generative Adversarial Network (QGAN)** for synthesizing handwritten digits using a hybrid quantum-classical architecture.

## 🎯 Project Overview

This project implements a hybrid GAN where:
- **Generator**: Parameterized Quantum Circuit (PQC) using Qiskit
- **Discriminator**: Classical neural network using PyTorch
- **Dataset**: EMNIST 16×16 (112,800 samples, 47 classes)

The quantum generator uses the probability distribution from quantum measurements as pixel intensities, enabling image generation from quantum circuits.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QGAN Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Latent Vector z ──► Quantum Generator ──► Fake Image      │
│       (4-8 dim)      (PQC with Ry/Rz/CNOT)  (4×4 or 16×16) │
│                              │                              │
│                              ▼                              │
│  Real Image ────────► Discriminator ◄────── Fake Image     │
│  (from EMNIST)        (MLP/Conv NN)                        │
│                              │                              │
│                              ▼                              │
│                      Real/Fake Score                        │
│                         (0-1)                               │
└─────────────────────────────────────────────────────────────┘
```

### Qubit-to-Pixel Mapping

| Qubits | Output Dimension | Image Size | Use Case |
|--------|------------------|------------|----------|
| 4      | 2⁴ = 16          | 4×4        | Fast experiments |
| 8      | 2⁸ = 256         | 16×16      | Full resolution |

## 📁 Project Structure

```
qgan-mnist-synthesis/
├── data/
│   ├── raw/                    # Original dataset
│   │   └── EMNIST_16x16_Dataset.csv
│   ├── processed/              # Preprocessed .npz files
│   └── dataloader.py           # PyTorch DataLoader
├── src/
│   ├── quantum/
│   │   ├── ansatz.py           # PQC architectures
│   │   └── encoding.py         # SamplerQNN wrapper
│   ├── classical/
│   │   └── discriminator.py    # PyTorch discriminators
│   ├── training/
│   │   ├── qgan_trainer.py     # Hybrid training loop
│   │   └── losses.py           # GAN loss functions
│   └── utils/
│       ├── metrics.py          # FID, mode coverage
│       └── visualization.py    # Plotting utilities
├── scripts/
│   ├── prepare_data.py         # Data preprocessing
│   └── run_experiment.py       # Main training script
├── experiments/
│   ├── configs/                # YAML experiment configs
│   └── results/                # Saved models & metrics
├── tests/
│   └── test_circuits.py        # Unit tests
├── notebooks/                  # Jupyter notebooks
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qgan-mnist-synthesis.git
cd qgan-mnist-synthesis

# Create virtual environment
python -m venv qgan-env
source qgan-env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from qiskit_machine_learning.neural_networks import SamplerQNN; print('OK')"
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py
```

This creates preprocessed datasets in `data/processed/`:
- `mnist_4x4_binary.npz` - 4×4 images, digits 0-1
- `mnist_16x16_full.npz` - 16×16 images, digits 0-9

### 3. Run Training

```bash
# Run default 4-qubit experiment
python scripts/run_experiment.py

# Or specify a config file
python scripts/run_experiment.py experiments/configs/experiment_001.yaml
```

### 4. View Results

Results are saved to `experiments/results/<experiment_name>_<timestamp>/`:
- `training_curves.png` - Loss over epochs
- `final_samples.png` - Generated images
- `real_vs_fake.png` - Comparison with real images
- `checkpoint.pt` - Model checkpoint
- `metrics.yaml` - Evaluation metrics

## ⚙️ Configuration

Experiments are configured via YAML files in `experiments/configs/`:

```yaml
experiment:
  name: "qgan_4qubit_baseline"
  seed: 42

data:
  target_digits: [0, 1]
  target_size: [4, 4]
  batch_size: 32

generator:
  n_qubits: 4
  n_layers: 2
  latent_dim: 4
  entanglement: "linear"

discriminator:
  type: "mlp"
  dropout: 0.3

training:
  n_epochs: 50
  learning_rate_g: 0.01
  learning_rate_d: 0.001
```

## 🧪 Running Tests

```bash
pytest tests/test_circuits.py -v
```

## 📊 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **FID** | Fréchet Inception Distance | Lower is better |
| **Mode Coverage** | Diversity via clustering | > 0.7 |
| **Training Stability** | Loss variance | < 0.1 |

## 🔬 Key Components

### Quantum Generator (ansatz.py)

```python
from src.quantum.ansatz import create_generator_ansatz

qc, latent_params, trainable_params = create_generator_ansatz(
    n_qubits=4,      # Number of qubits
    n_layers=2,      # Circuit depth
    latent_dim=4,    # Noise vector size
    entanglement='linear'  # CNOT pattern
)
```

### SamplerQNN Wrapper (encoding.py)

```python
from src.quantum.encoding import create_generator_qnn

qnn = create_generator_qnn(qc, latent_params, trainable_params)
# Output: probability distribution over 2^n_qubits states
```

### Training Loop (qgan_trainer.py)

```python
from src.training.qgan_trainer import QGANTrainer

trainer = QGANTrainer(
    generator_qnn=qnn,
    discriminator=discriminator,
    latent_dim=4
)
history = trainer.train(dataloader, n_epochs=50)
```

## ⚠️ Known Limitations

1. **Barren Plateaus**: Keep circuit depth ≤ 4 layers
2. **Simulation Overhead**: 8 qubits max for reasonable training time
3. **Low Resolution**: Quantum advantage unclear for small images

## 📚 References

- [Quantum Generative Adversarial Networks](https://arxiv.org/abs/1804.09139)
- [Qiskit Machine Learning Documentation](https://qiskit-community.github.io/qiskit-machine-learning/)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read the project documentation before submitting PRs.

---

*Built with Qiskit 💜 and PyTorch 🔥*

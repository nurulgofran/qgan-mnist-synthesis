# QGAN MNIST Synthesis

A Quantum Generative Adversarial Network for synthesizing handwritten digits using a hybrid quantum-classical architecture.

## Overview

This project implements a hybrid GAN where:
- **Generator**: Parameterized Quantum Circuit (PQC) using Qiskit
- **Discriminator**: Classical neural network using PyTorch
- **Dataset**: MNIST/EMNIST 16x16 images

## Project Structure

```
qgan-mnist-synthesis/
├── src/
│   ├── quantum/
│   │   └── gpu_quantum.py
│   ├── classical/
│   │   ├── discriminator.py
│   │   └── baseline_gan.py
│   ├── training/
│   │   ├── gpu_trainer.py
│   │   └── losses.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_experiment.py
│   └── run_comparison.py
├── data/
├── experiments/
│   └── configs/
├── tests/
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

1. Clone the repository
   ```bash
   git clone https://github.com/nurulgofran/qgan-mnist-synthesis.git
   cd qgan-mnist-synthesis
   ```

2. Create a virtual environment
   ```bash
   python -m venv qgan-env
   source qgan-env/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation
   ```bash
   python -c "from qiskit_machine_learning.neural_networks import SamplerQNN; print('OK')"
   ```

## Running the Code

### Prepare the Dataset

```bash
python scripts/prepare_data.py
```

### Train the QGAN

Default training (4-qubit experiment):
```bash
python scripts/run_experiment.py
```

With a specific configuration:
```bash
python scripts/run_experiment.py experiments/configs/experiment_001.yaml
```

### Compare QGAN vs Classical GAN

```bash
python scripts/run_comparison.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Configuration

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

## Results

After training, results are saved to `experiments/results/<experiment_name>_<timestamp>/`:
- `training_curves.png` - Loss over epochs
- `final_samples.png` - Generated images
- `real_vs_fake.png` - Comparison with real images
- `checkpoint.pt` - Model checkpoint
- `metrics.yaml` - Evaluation metrics

## License

MIT License - see LICENSE for details.

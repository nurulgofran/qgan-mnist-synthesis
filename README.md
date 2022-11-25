# QGAN MNIST Synthesis

A Quantum Generative Adversarial Network for synthesizing handwritten digits using a hybrid quantum-classical architecture.

## Overview

This project implements a hybrid GAN where:
- **Generator**: Parameterized Quantum Circuit (PQC) using Qiskit
- **Discriminator**: Classical neural network using PyTorch
- **Dataset**: MNIST/EMNIST 16x16 images

## Requirements

- Python 3.10 or higher
- pip (Python package manager)
- Git

## Installation

### Step 1: Clone the Repository

Open a terminal (or Command Prompt on Windows) and run:

```bash
git clone https://github.com/nurulgofran/qgan-mnist-synthesis.git
cd qgan-mnist-synthesis
```

### Step 2: Create a Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv qgan-env
source qgan-env/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv qgan-env
qgan-env\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
python -m venv qgan-env
.\qgan-env\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print('PyTorch OK')"
```

## Running the Project

### Step 1: Prepare the Dataset

This script preprocesses the MNIST data for training:

**On macOS/Linux:**
```bash
python3 scripts/prepare_data.py
```

**On Windows:**
```cmd
python scripts/prepare_data.py
```

### Step 2: Train the QGAN

Run the default 4-qubit experiment:

**On macOS/Linux:**
```bash
python3 scripts/run_experiment.py
```

**On Windows:**
```cmd
python scripts/run_experiment.py
```

Or specify a configuration file:

**On macOS/Linux:**
```bash
python3 scripts/run_experiment.py experiments/configs/experiment_001.yaml
```

**On Windows:**
```cmd
python scripts/run_experiment.py experiments/configs/experiment_001.yaml
```

### Step 3: Compare QGAN vs Classical GAN

**On macOS/Linux:**
```bash
python3 scripts/run_comparison.py
```

**On Windows:**
```cmd
python scripts/run_comparison.py
```

### Step 4: Run Tests

**On macOS/Linux:**
```bash
pytest tests/ -v
```

**On Windows:**
```cmd
pytest tests/ -v
```

## Output

After training, results are saved to `experiments/results/<experiment_name>_<timestamp>/`:

| File | Description |
|------|-------------|
| `training_curves.png` | Loss over epochs |
| `final_samples.png` | Generated images |
| `real_vs_fake.png` | Comparison with real images |
| `checkpoint.pt` | Model checkpoint |
| `metrics.yaml` | Evaluation metrics |

## Project Structure

```
qgan-mnist-synthesis/
├── src/                    # Source code
│   ├── quantum/            # Quantum circuit implementations
│   ├── classical/          # Classical neural networks
│   ├── training/           # Training utilities
│   └── utils/              # Helper functions
├── scripts/                # Executable scripts
│   ├── prepare_data.py     # Data preprocessing
│   ├── run_experiment.py   # Main training script
│   └── run_comparison.py   # QGAN vs Classical comparison
├── data/                   # Dataset storage
├── experiments/            # Experiment configs and results
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md
```

## Configuration

Edit the YAML files in `experiments/configs/` to customize training:

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

training:
  n_epochs: 50
  learning_rate_g: 0.01
  learning_rate_d: 0.001
```

## Troubleshooting

**"python not found" error:**
- Make sure Python 3.10+ is installed
- On macOS/Linux, try `python3` instead of `python`

**"pip not found" error:**
- Run `python -m pip install --upgrade pip`

**Virtual environment not activating:**
- On Windows PowerShell, you may need to run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## License

MIT License - see LICENSE for details.

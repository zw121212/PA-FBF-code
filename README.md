# A Physics-Aware Flow-Based Bayesian Filtering Framework for Stochastic Partial Differential Equations

This repository provides the implementation of a physics-aware flow-based Bayesian filtering framework for stochastic partial differential equations (SPDEs). The method combines flow-based probabilistic modeling with recursive Bayesian filtering, while embedding physical constraints into the inference procedure.

The code is used to conduct numerical experiments on several benchmark problems, including a 2D Nonlinear System, a Stochastic Advection–Diffusion Model, and the Stochastic Heat Equation.

---

## Code Structure

```text
.
├── flow_models.py        # Main implementation of the physics-aware flow-based Bayesian filtering framework
├── compute.py            # Numerical utility functions

├── data_2d.py            # Data generation: 2D nonlinear system
├── data_diffusion.py     # Data generation: Stochastic Advection–Diffusion Model
├── data_heat.py          # Data generation: Stochastic Heat Equation

├── train_2d.py           # Training: 2D nonlinear system
├── train_diffusion.py    # Training: Stochastic Advection–Diffusion Model
├── train_heat.py         # Training: Stochastic Heat Equation

├── test_2d.py            # Testing & visualization: 2D nonlinear system
├── test_diffusion.py     # Testing & visualization: Stochastic Advection–Diffusion Model
├──  test_heat.py          # Testing & visualization: Stochastic Heat Equation

├── Data/                 # Datasets used in experiments
└── PI/                   # Pre-trained model parameters


```

---

## Numerical Experiments

The repository includes three benchmark problems:

- 2D nonlinear system
- Stochastic Advection–Diffusion Model
- Heat Equation

Each experiment follows a unified pipeline:

data generation → training → evaluation

The suffix in each script name indicates the corresponding experiment:

- `2d`: 2D nonlinear system  
- `diffusion`: Stochastic Advection–Diffusion Model
- `heat`:  Stochastic Heat Equation  

---

## Usage

### 1. Training

Run the following scripts to train the model:

```bash
python train_2d.py
python train_diffusion.py
python train_heat.py
```

Each training script:
- loads or generates data,
- initializes the model defined in `flow_models.py`,
- performs iterative training,
- saves model parameters.

### 2. Testing / Reproducing Results

To reproduce numerical results using the provided datasets and pre-trained models (stored in Data/ and PI/), run:

```bash
python test_2d.py
python test_diffusion.py
python test_heat.py
```

Each testing script:
- loads trained model parameters,
- performs inference,
- generates the corresponding figures.

---

## Requirements

The implementation is based on:

```text
Python >= 3.8
PyTorch
NumPy
Matplotlib
```

---

## Reproducibility

- All experiments are script-based and reproducible.
- Training and testing are fully separated.
- Datasets are provided in the `Data/` directory.
- Pre-trained model parameters are provided in the `PI/` directory.
- Results can be reproduced by directly running the corresponding `test_*` scripts.
---

## Notes

- The core implementation is located in `flow_models.py`.
- The framework is designed for stochastic PDE inference, uncertainty quantification, and physics-aware probabilistic modeling.

---

## Remarks

- The repository includes both data and pre-trained models to facilitate direct reproduction of all reported results.


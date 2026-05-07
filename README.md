# Kinematic Budget of Quantum Correlations

This repository contains the numerical implementations and figure generation scripts for the paper [**"Kinematic budget of quantum correlations"**](https://arxiv.org/abs/2603.03887) by Maaz Khan and Subhadip Mitra.

## Overview

Quantum correlations such as discord, entanglement, steering, and Bell nonlocality are typically treated as separate phenomena. This work introduces a unified geometric framework that treats **state purity ($P$)** as a finite resource budget. 

By mapping quantum systems onto a compact, two-dimensional $(X, Y)$ manifold (where $X$ represents local polarisation and $Y$ represents nonlocal correlations), the framework reveals the nested hierarchy of quantum resources and their dynamic redistribution under decoherence.

## Repository Structure

The project consists of several Python scripts, each corresponding to specific figures in the paper. The scripts leverage high-performance numerical libraries to simulate quantum states and noise channels.

### Figure Generation Scripts

| Script | Figures | Description |
| :--- | :--- | :--- |
| `Fig_1A.py` | 1a | Analytic boundaries for 2-qubit systems in the $(X, Y)$ plane. |
| `Fig_1B.py` | 1b | Monte Carlo validation of 2-qubit boundaries using random state sampling. |
| `Fig_2AB.py` | 2a, b | Analysis of stabilizer magic and negativity vs. budget angle $\theta$ for 2 qubits. |
| `Fig_2CD.py` | 2c, d | High-dimensional (5 and 7-qubit) magic capacity limits.|
| `Fig_3A_6A.py` | 3a, 6a | Higher-dimensional bottlenecks (Qubit-Qutrit systems). |
| `Fig_3B_6B.py` | 3b, 6b | Continued analysis of dimensional bottlenecks in 2x3 systems. |
| `Fig_3C_6C.py` | 3c, 6c | Multipartite (3-qubit) phase space and kinematic limits. |
| `Fig_3D.py` | 3d | Witnessing NPT entanglement in Qubit-Qutrit systems. |
| `Fig_3E.py` | 3e | Entanglement witnessing for Qutrit-Qutrit systems. |
| `Fig_3FG.py` | 3f, g | Biseparable and Genuine Multipartite Entanglement thresholds for 3rd order. |
| `Fig_4.py` | 4 | 2-qubit decoherence trajectories under local and correlated noise channels. |
| `Fig_5.py` | 5 | Geometric conflict between entropy extraction and symmetry restoration (i.e. $P$ and $Q$ do not increase simultaneously.). |
| `Fig_7.py` | 7 | Comprehensive Qubit-Qutrit (2x3) decoherence and dual-plane mapping. |
| `Fig_8.py` | 8 | Comprehensive Qutrit-Qutrit (3x3) decoherence and dual-plane mapping. |
| `Fig_9.py` | 9 | Multipartite (3-qubit) kinematic budget flows and decoherence trajectories. |
| `Fig_10.py` | 10 | Active purification protocols and virtual distillation simulations. |

## Requirements

To run these scripts, you will need the following Python libraries:

- `numpy`
- `matplotlib`
- `scipy`
- `torch` (for high-dimensional magic calculations)
- `jax` (for vectorized sampling)
- `numba` (for high-speed metric processing)

## Usage

Simply run the desired script using Python:

```bash
python Fig_1A.py
```

Each script will generate a high-resolution PDF of the corresponding figure in the same directory.

## Authors

- **Maaz Khan** (IIIT Hyderabad) - [maaz.khan@research.iiit.ac.in](mailto:maaz.khan@research.iiit.ac.in)
- **Subhadip Mitra** (IIIT Hyderabad) - [subhadip.mitra@iiit.ac.in](mailto:subhadip.mitra@iiit.ac.in)

## Citation

If you use this work in your research, please cite:

```bibtex
@article{Khan:2026okk,
    author = "Khan, Maaz and Mitra, Subhadip",
    title = "{Kinematic budget of quantum correlations}",
    eprint = "2603.03887",
    archivePrefix = "arXiv",
    primaryClass = "quant-ph",
    month = "3",
    year = "2026"
}
```

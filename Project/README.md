# Qutrits for physics at LHC ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)
[![Static Badge](https://img.shields.io/badge/ArXiv-2510.14001-red)](https://arxiv.org/abs/2510.14001)




### Abstract

The identification of anomalous events, not explained by the Standard Model of particle physics, and the possible discovery of exotic physical phenomena pose significant theoretical, experimental and computational challenges. The task will intensify at next-generation colliders, such as the High- Luminosity Large Hadron Collider (HL-LHC). Consequently, considerable challenges are expected concerning data processing, signal reconstruction, and analysis. This work explores the use of qutrit- based Quantum Machine Learning models for anomaly detection in high-energy physics data, with a focus on LHC applications. We propose the development of a qutrit quantum model and benchmark its performance against qubit-based approaches, assessing accuracy, scalability, and computational efficiency. This study aims to establish whether qutrit architectures can offer an advantage in addressing the computational and analytical demands of future collider experiments.

---

This repository provides a complete end-to-end framework for the development, execution, and statistical evaluation of **quantum autoencoders (QAEs)** applied to **high-energy physics jet data** from CMS. It includes implementations based on **qubits and qutrits**, large-scale **HPC executions**, and a dedicated workflow for **statistical analysis and anomaly detection**.

All large-scale experiments were executed on **Finisterrae III**, the supercomputer of **CESGA (Centro de Supercomputación de Galicia)**, due to the high computational cost of repeated quantum model executions.

---


Each directory corresponds to a key stage of the project pipeline, from model construction to high-performance execution and final statistical analysis.

---

## 📋 Prerequisites

- **Python 3.8+**
- **Quantum libraries**:
  - [Pennylane](https://pennylane.ai/)
- **Scientific libraries**: NumPy, SciPy, Matplotlib, Pandas
- **Machine Learning**: TensorFlow or PyTorch
- **Environment**: JupyterLab/Notebook

---

## ⚙️ Installation

```bash
git clone https://github.com/MirandaCarou/Qutrits-for-physics-at-LHC.git
```

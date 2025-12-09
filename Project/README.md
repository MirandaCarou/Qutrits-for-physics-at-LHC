# Qutrits for physics at LHC

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

## 1. Qubit-Based QAE

**Path:** `Qubit-based_Model/`

This module implements a **qubit-based Quantum Autoencoder (QAE)** using **PennyLane and PyTorch**, following the methodology proposed in:

> https://arxiv.org/abs/2502.17301

### Main Features

- Loading and preprocessing of **jet events from CMS datasets**
- Selection of leading jet constituents by **transverse momentum ($p_T$)**
- Dataset splitting:
  - 10,000 training samples  
  - 2,500 validation samples  
  - 10,000 inference samples  
- **1P1Q encoding scheme**
- Definition of:
  - Encoder
  - Variational layers
  - Fidelity-based cost function
- **Variational training using Adam optimizer**
- **Inference on background and multiple signal processes**
- **ROC and AUC evaluation using anomaly scores** (`1 - fidelity`)
- Includes a **reduced QAE circuit** for fast experimentation

This model establishes the baseline for quantum anomaly detection using qubits.

---

## 2. Qutrit-Based QAE

**Path:** `Qutrit-based_Model/`

This module extends the QAE framework to **qutrit-based systems**, including both **standard SU(3) encoding** and a novel **Majorana-based encoding**.

### Submodules

#### a) Before Majorana encoding
- First qutrit QAE inspired by the reference paper
- Qutrit operators (SU(3), SO(3))
- Controlled TSWAP gates
- Fidelity-based optimization
- ROC/AUC analysis

#### b) With majorana encoding
- Extended feature space including:
  - Transverse impact parameter (`d0`)
  - Longitudinal impact parameter (`dz`)
- Majorana encoding for qutrits
- Gram–Schmidt unitary construction
- Variational training and inference
- Robustness validation of qutrit operations

#### c) Additional tools
- `plot_dataset_values.ipynb`:
  - Jet feature correlation studies
  - Redundancy analysis
- `robustness_test.ipynb`:
  - Mathematical and numerical verification of qutrit encodings
- `saved_fidelities/`:
  - Stores `.npz` files with previously computed fidelity results

This module enables the study of **higher-dimensional quantum encodings** and their impact on anomaly detection performance.

---

## 3. Scripts for data processing and conversion (CESGA)

**Path:** `Scripts_CESGA/`

This section contains scripts used to **process and convert massive datasets of real and simulated CMS data** generated from quantum circuits.

### Main Functions

- Load large volumes of raw experimental and simulated data
- Clean and standardize dataset structures
- Convert data to formats directly compatible with:
  - Qubit-based models
  - Qutrit-based models
- Automate large-scale preprocessing workflows

These scripts ensure **full consistency between real and simulated datasets** before training and evaluation.

---

## 4. High-Performance model execution

**Path:** `Test_models_In_HPC/`

This directory contains the **final `.py` implementations of the qubit and qutrit QAEs**, prepared for large-scale execution on **Finisterrae III**.

### Key Characteristics

- Each experiment is executed **100 independent times**
- Separate implementations for:
  - Qubits
  - Qubits with real data
  - Qutrits
  - Qutrits with real data
- Automated batch executions
- Output stored for subsequent statistical analysis

Due to the extremely high computational cost of repeated quantum simulations, all experiments were deployed on the **CESGA supercomputing infrastructure**.

---

## 5. Statistical analysis & anomaly detection

**Path:** `Analysis_and_Stadistics/`

This module performs the **final statistical evaluation and anomaly detection analysis** using the fidelity outputs generated on the HPC.

### Main tasks

- Loads `.npz` files containing **100-run fidelity outputs**
- Computes:
  - Mean and median fidelities
  - **Jensen–Shannon Divergence (JSD)** between datasets
- Converts fidelities into **anomaly scores**: anomaly_score = 1 - fidelity

- Computes **ROC curves and AUC values** for:
- `H → bb`
- `t → bqq`
- `W → qq`
- Aggregates:
- **Average AUC**
- **Average JSD**
- Supports:
- Histogram visualization
- ROC plotting
- Dispersion analysis across iterations

### Model variants analyzed

- Qubits trained on:
- Simulated CMS data
- Real CMS data
- Qutrits trained on:
- Simulated CMS data
- Real CMS data
- Multiple feature encodings:
  - (`d0`, `dz`)
  - (`τ12`, `E`)
  - (`τ23`, `E`)
  - (`τ34`, `E`)
  - Individual (`τ1`, `τ2`, `τ3`, `τ4`)

This framework provides a **robust statistical comparison of quantum encodings and model variants** for anomaly detection.

---

## Global summary

This repository provides a **complete research pipeline for quantum anomaly detection in jet physics**, including:

- ✅ Qubit-based and qutrit-based quantum autoencoders  
- ✅ Standard and Majorana-based qutrit encodings  
- ✅ Real and simulated CMS dataset processing  
- ✅ Large-scale HPC execution on Finisterrae III (CESGA)  
- ✅ 100-run statistical evaluation per experiment  
- ✅ JSD, ROC, AUC, and fidelity distribution analysis  

It enables **systematic benchmarking of quantum representations, feature encodings, and anomaly detection capabilities** in realistic high-energy physics environments.

---

If needed, I can also:
- Add an **installation section**
- Add **environment requirements**
- Or adapt this README for a **journal or thesis repository**.

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
git clone https://github.com/MirandaCarou/Research-Intership-Memory.git
cd Research-Intership-Memory
```

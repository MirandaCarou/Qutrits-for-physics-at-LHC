# Qutrit-Based_Model

This repository contains code and resources for implementing and analyzing **qutrit-based Quantum Autoencoders (QAEs)** applied to high-energy physics jet data, including versions with standard encoding and Majorana-based encoding. The scripts cover data preprocessing, qutrit quantum circuit construction, variational training, inference, evaluation, and visualization. The repository also includes tools for exploring jet parameter distributions and testing the robustness of qutrit encodings.

## Repository Structure

Qutrit-Based_Model/
│
├── Before_discover_Majorana_Encodinf/
│ └── QAE_qutrits.ipynb
│ # First implementation of a qutrit-based Quantum Autoencoder inspired by
│ # the paper https://arxiv.org/abs/2502.17301. Implements:
│ # - Data loading and preprocessing of jet events
│ # - Dataset splitting (training, validation, inference)
│ # - Qutrit quantum operators and generators (SU(3) and SO(3))
│ # - Qutrit QAE circuit: encoder, variational layer, controlled TSwap
│ # - Cost function using negative fidelity
│ # - Variational training and optimization
│ # - Inference and fidelity evaluation for background and signal jets
│ # - Fidelity distribution plots and ROC/AUC analysis
│
├── With_Majorana/
│ └── Pure_states/
│ └── QAE_qutrits_majorana_encoding.ipynb
│ # Qutrit-based QAE with Majorana encoding:
│ # - Extended preprocessing including traversal (d0) and longitudinal (dz) parameters
│ # - Qutrit operators, TSWAP gates, unitary constructions
│ # - Encoder and variational layers using Majorana encoding
│ # - Training with Adam optimizer
│ # - Inference on background and signal jets
│ # - Fidelity distributions and ROC/AUC evaluations
│
├── saved_fidelities/
│ └── *.npz
│ # Stores previously computed fidelity results from QAE executions
│
├── plot_dataset_values.ipynb
│ # Jet parameter correlation and redundancy analysis:
│ # - Loads and preprocesses jet data
│ # - Extracts jet- and constituent-level features
│ # - Performs exploratory data analysis (scatter plots, histograms, correlations)
│ # - Compares distributions across datasets and physics processes (QCD, H→bb, t→bqq, W→qq)
│
└── robustness_test.ipynb
# Robustness test for qutrit-based quantum encoding and variational circuits:
# - Encodes qutrits using Majorana representation and checks state preparation
# - Applies Gram–Schmidt to construct unitaries
# - Validates encoding and rotations mathematically and via Pennylane simulations
# - Implements variational layers with TAdd gates and rotations
# - Confirms consistency of state evolution and variational outputs


## Summary

The repository provides a comprehensive framework for:

- Constructing **qutrit-based Quantum Autoencoders** for particle physics data.
- Preprocessing and analyzing **jet-level and constituent-level features**.
- Implementing **Majorana-based qutrit encoding** for quantum circuits.
- Performing **variational training**, fidelity evaluation, and anomaly detection (ROC/AUC).
- Testing **robustness** and correctness of qutrit encodings and operations.
- Visualizing **fidelity distributions, correlations, and jet parameter redundancies**.

This collection of scripts and data files supports both research and development of quantum machine learning models using qutrits in high-energy physics applications.

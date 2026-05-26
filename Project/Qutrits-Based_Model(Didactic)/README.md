# Qutrit-based model

(Didactic purpose) This repository contains code and resources for implementing and analyzing **qutrit-based Quantum Autoencoders (QAEs)** applied to high-energy physics jet data, including versions with standard encoding and Majorana-based encoding. The scripts cover data preprocessing, qutrit quantum circuit construction, variational training, inference, evaluation, and visualization. The repository also includes tools for exploring jet parameter distributions and testing the robustness of qutrit encodings.

## Repository structure

  - **With_Majorana/**
    - **Pure_states/**
    - `QAE_qutrits_majorana_encoding.ipynb`  
      Qutrit-based QAE with Majorana encoding:
      - Extended preprocessing including traversal (d0) and longitudinal (dz) parameters
      - Qutrit operators, TSWAP gates, unitary constructions
      - Encoder and variational layers using Majorana encoding
      - Training with Adam optimizer
      - Inference on background and signal jets
      - Fidelity distributions and ROC/AUC evaluations


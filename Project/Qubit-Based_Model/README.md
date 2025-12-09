# Qubit-based Quantum Autoencoder (QAE)

This part contains code for a **qubit-based quantum autoencoder (QAE)**, implemented using PennyLane and PyTorch. The implementation is based on the paper [https://arxiv.org/abs/2502.17301](https://arxiv.org/abs/2502.17301), and it allows analysis and comparison of quantum autoencoder performance on jet event datasets.

---

## Data Loading and Processing

The script loads multiple JSON files containing **jet events**, selects the top constituents by **transverse momentum ($p_T$)** for each jet, and computes their kinematic variables (**$p_T$, $\eta$, $\phi$**) for further analysis.

- **Loading training real data**  
- **Loading simulated CMS signal data** (mainly for inference)

---

## Splitting Data into Train, Validation, and Inference Sets

The datasets are converted to NumPy arrays and split as follows:

- 10,000 samples for training  
- 2,500 samples for validation  
- Remaining 10,000 samples for inference  

The splits are randomized to ensure diverse sampling.

---

## Quantum Circuit Setup and Encoder Definition

This section sets up a **qubit-based quantum autoencoder**:

- Defines the quantum device and initializes wires for **latent**, **trash**, **reference**, and **ancilla** qubits  
- Implements the **1P1Q encoding scheme** for jet constituents  
- Defines **variational layers** and the **QAE circuit**  
- Includes a **cost function based on fidelity** for training

---

## Quantum Autoencoder Training Loop

The script initializes trainable parameters and sets up an **Adam optimizer**. During training:

- The loss and fidelity are computed for each jet using the QAE circuit  
- Parameters are updated via backpropagation  
- Average loss and fidelity per epoch are recorded and printed  

Additionally, fidelity distributions can be plotted for monitoring.

---

## Evaluating Fidelity on Different Jet Samples (Inference)

The trained QAE is evaluated on:

- Background (`X_inf`)  
- Signal datasets (`datos_HToBB`, `datos_TTBar`, `datos_WToqq`)  

For each jet, the **fidelity** is computed and stored with labels for later ROC/AUC analysis.

---

## ROC Curve and AUC Evaluation

The code calculates **anomaly scores** as `1 - fidelity` and evaluates **ROC AUC**:

- Compares background with signal types (`H→bb`, `t→bqq`, `W→qq`)  
- Plots ROC curves with reference random classifier  
- Provides a visual representation of anomaly detection performance

---

## Reduced QAE Circuit

A smaller QAE is implemented with:

- 4 particles and 1 latent qubit  

This reduced circuit allows **faster experimentation** while training and evaluating:

- Loss and fidelity are computed per jet  
- Parameters are updated via backpropagation  
- Average fidelities are recorded for monitoring

---

## Repeated QAE Training and Fidelity Col

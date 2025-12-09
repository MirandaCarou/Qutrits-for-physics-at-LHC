# 🍄 Quantum fidelity statistical analysis and anomaly detection 

This part contains code for **statistical analysis of quantum autoencoder outputs** (both qubit- and qutrit-based models) and evaluates their **anomaly detection performance**. The main objectives include analyzing fidelity distributions, computing divergences between datasets, and assessing separability using ROC-AUC metrics.

---

## 🍃 Data loading

The script loads **`.npz` files** containing results from multiple executions (usually 100 runs) of qubit and qutrit quantum models.  

Example datasets include:

- `qubitsReal`  
- `qutritsReal`  
- `qutritsTau12E`  
- `qutritsd0dz`  

These files store all experiment results for analysis.

---

## ✍🏻 Statistical analysis

For each dataset and model, the script computes:

- **Mean and median fidelities** for different processes/events:  
  - `HToBB`  
  - `WToQQ`  
  - `TTBar`  
  - `back` (background)  
- **Jensen-Shannon Divergence distance (JSD)** between distributions to quantify similarity/differences between signal and background or between different signal datasets.

---

## ⚛️ Anomaly detection metrics

- Converts fidelities into **anomaly scores**: `anomaly_score = 1 - fidelity`  
- Calculates **ROC AUC** for each signal type against background to quantify how well the model separates signals from background

---

## 🍂 Aggregated metrics

- Computes **average AUC** and **average JSD** across multiple trials (typically 100 iterations)  
- Stores results for **comparative analysis** across different quantum states and configurations (qubits vs qutrits, real vs simulated CMS data)

---

## 👾 Model variants analyzed

### Qubit-based Models

- Trained on **real CMS data** (`qubitsReal`)  
- Trained on **simulated CMS data** (`qubits`)  

### 🫧 Qutrit-based Models

- Trained on **real CMS data** (`qutritsReal`)  
- Trained on **simulated CMS data** with different features:  
  - Longitudinal and transverse impact parameters (`qutritsd0dz`)  
  - $\tau_{12}$ and jet energy (`qutritsTau12E`)  
  - $\tau_{23}$ and jet energy (`qutritsTau23E`)  
  - $\tau_{34}$ and jet energy (`qutritsTau34E`)  
  - Individual $\tau_1, \tau_2, \tau_3, \tau_4$ (`qutritsTaus`)  

For each iteration and dataset, the script calculates:

- Mean and median fidelities  
- Jensen-Shannon divergence (JSD) between signals and background, and between different signals  
- Anomaly scores (`1 - probability`)  
- ROC AUC for signal vs background separation  

Finally, **average AUC** and **average JSD** across iterations are computed to summarize overall performance.

---

## 🖨️ Printing results

The script includes routines to:

- Organize computed **JSD values** between pairs of datasets (`HToBB`, `WToQQ`, `TTBar`)  
- Print **mean JSD values** for each dataset variant (qubits, qutrits, and feature variations)  
- Inspect **anomaly scores** and distribution statistics

---

## 🥸 Visualization

The code supports plotting:

- **ROC curves** for background vs different signal types  
- **Fidelity distribution histograms**  
- **Dispersion analysis** across multiple iterations to assess variability and consistency of results

---

## ✨ Summary

This workflow provides a **comprehensive analysis framework** for quantum autoencoder outputs:

- Evaluates qubit- and qutrit-based models  
- Quantifies differences between distributions via JSD  
- Measures anomaly detection performance using ROC-AUC  
- Tracks variability across multiple iterations for robust statistical insights

It enables systematic comparison of different **quantum configurations**, **datasets**, and **feature sets** for anomaly detection in jet physics experiments.

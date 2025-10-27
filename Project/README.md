# Quantum Machine Learning model based on qutrits for anomaly detection in high-energy physics data: Comparative analysis with qubit-based models and application to CERN's LHC collider. (Master’s Thesis)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)


This repository contains the Final Master’s Thesis (TFM) work on using three-level quantum systems (qutrits) to create a Qantum Autoencoder for anomaly detection in High Energy Physics. 

The identification of anomalous events – not explained by the Standard Model of particle physics – and the possible discovery of exotic physical phenomena pose significant theoretical, experimental and computational challenges. It is anticipated that these challenges will increase significantly with the operation of next-generation colliders, such as the High-Luminosity Large Hadron Collider (HL-LHC). 

At least 140 collisions will be produced each time two particle bunches meet at the heart of the ATLAS and CMS detectors, compared to around 40 collisions at present. Consequently, significant challenges are to be expected in terms of data processing, reconstruction, and analysis. This project sets out to explore the development of unsupervised anomaly detection methods that do not rely on prior knowledge of the underlying physics models.


With this in mind, the project exploits the theoretical and practical advantages of utilizing qutrits in Quantum Machine Learning (QML) models for the purpose of anomaly detection in high-energy physics data, with a particular focus on the context of experiments at CERN’s Large Hadron Collider. The development of a quantum model based on qutrits is proposed, with a comparison with its qubit counterpart undertaken to evaluate its effectiveness in terms of accuracy, scalability and computational efficiency. The objective is threefold: first, to enhance comprehension of multilevel quantum systems and their capacity for the development of more compact quantum algorithms; second, to examine fresh possibilities for the analysis of complex data; and third, to collaborate in the advancement of this field.

To achieve the desired objectives, a high-fidelity autoencoder structure has been utilized as a QAE reference, with CMS real jet data being employed to train the model. This model has been extrapolated to the qutrit state space, with the introduction of novel logic gates according to the parameters of this state space.

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
git clone https://github.com/MirandaCarou/Research-Intership-Memory.git
cd Research-Intership-Memory
```

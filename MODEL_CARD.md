
# Model Card for *"Qutrits for physics at LHC"* work
<!-- Provide a quick summary of what the model is/does. -->
---
The identification of anomalous events, not explained by the Standard Model of particle physics, and the possible discovery of exotic physical phenomena pose significant theoretical, experimental and computational challenges. The task will intensify at next-generation colliders, such as the High- Luminosity Large Hadron Collider (HL-LHC). Consequently, considerable challenges are expected concerning data processing, signal reconstruction, and analysis. This work explores the use of qutrit- based Quantum Machine Learning models for anomaly detection in high-energy physics data, with a focus on LHC applications. We propose the development of a qutrit quantum model and benchmark its performance against qubit-based approaches, assessing accuracy, scalability, and computational efficiency. This study aims to establish whether qutrit architectures can offer an advantage in addressing the computational and analytical demands of future collider experiments.
---

## Model Details 🚀

### Model Description 📑

This model implements a Quantum Autoencoder (QAE) for jet substructure analysis in high-energy physics, using simulated and recorded CMS detector data. A qubit-based QAE is first reproduced and validated against previously published results, demonstrating comparable AUC performance. This reference implementation serves as a benchmark for evaluating alternative quantum representations and ensures consistency with established results.
The model is then extended to a qutrit-based architecture, preserving the overall QAE structure while introducing qutrit-specific rotation gates, controlled operations, and a Majorana-based encoding scheme. The encoding incorporates physically motivated jet observables, including N-subjettiness ratios, energy, momentum, and impact parameters, selected based on their discriminative power. Due to current simulation and memory constraints, the qutrit model is evaluated in a reduced configuration with a single-qutrit latent space. Despite these limitations, the model demonstrates competitive performance and provides a viable proof of principle for qutrit-based quantum machine learning approaches, with scalability to larger systems left for future work.

- 🍄 **Developed by:** Miranda Carou Laiño, Veronika Chobanova, Miriam Lucio Martínez
- 🍄 **Model type:** Qutrit-based Quantum Autoencoder
- 🍄 **Language(s) (NLP):** Python. Pennylane
- 🍄 **License:** https://creativecommons.org/licenses/by/4.0/

### Model Sources 🌲

- 🌱 **Repository:** ➡️ https://github.com/MirandaCarou/Qutrits-for-physics-at-LHC/tree/main/Project/Qutrits-Based_Model
- 📑 **Paper:** ➡️ https://arxiv.org/abs/2510.14001

---

## Uses 

### Direct Use 🏋🏻‍♀️


This model can be used directly to analyze jet substructure in high-energy physics events, using input data that includes particle momenta, energy, and impact parameters. It is intended for researchers or students in quantum computing and particle physics who want to explore quantum machine learning approaches to particle-level event analysis.

### Downstream Use 🏃🏻‍♀️

The model can be fine-tuned or integrated into larger quantum or hybrid classical-quantum workflows for tasks such as jet classification, anomaly detection in HEP datasets, or benchmarking quantum encoding schemes. It may also serve as a reference for developing more complex qutrit-based quantum circuits.

### Out-of-Scope Use ✂️

This model is not intended for use with data outside of particle physics, for predicting real-world phenomena unrelated to jets, or for decision-making with safety-critical consequences. It is also not designed for large-scale deployment on classical systems without appropriate quantum hardware or simulation resources.

## Bias, Risks, and Limitations 🔍

The model is limited by current quantum simulation capabilities, which restrict the number of qutrits that can be simulated efficiently. Results may not fully generalize to larger systems or experimental datasets without further validation. The model relies on simulated or preprocessed experimental data and does not account for all sources of detector noise or event-level variability. Potential misuse includes applying the model to datasets outside its intended scope or misinterpreting probabilistic outputs as deterministic predictions

### Recommendations 📝

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

{{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}}

---

## How to Get Started with the Model 🍾

- **Check** (*Prerequisites*) ➡️ https://github.com/MirandaCarou/Qutrits-for-physics-at-LHC/blob/main/README.md 
- **Check** (*Project*) ➡️ https://github.com/MirandaCarou/Qutrits-for-physics-at-LHC/blob/main/Project/README.md
- **Read** (*Model*) ➡️ https://arxiv.org/abs/2510.14001
- **Read** (*Reference Model*) ➡️  https://journals.aps.org/prd/abstract/10.1103/l8y2-87vq

---
## Training Details

### Training Data 🖥️

The JetClass dataset and the data collected in 2016 by the CMS detector at the LHC and made public have been used separately for the optimal training of the model in different scenarios. The CMS dataset is characterised by having a format focused on Machine Learning and is dominated by Quantum Chromodynamics (QCD) jets with less than 1% contamination from other sources.
The JetClass contains 125 million jets, divided into ten classes, which are split across training, validation, testing, and inference. During the inference phase, when the trained model applies its learned patterns to compress previously unseen data, the simulated dataset JetClass is always employed. JetClass signals originating from decays of particles such as Higgs bosons, W/Z bosons, and top quarks are analysed to evaluate the model’s performance in distinguishing signal from background jets.

Furthermore, for the training phase of the QAE model, a previously sample of the data has been made such as each class has a flat distribution in PT,jet, in the range [500,1000] GeV, in order to prevent the training from being influenced by the jet scale, thus concentrating solely on jet substructure, as in Ref. 

### Training Procedure ⚙️

The qutrit-based Quantum Autoencoder (QAE) was trained using a differentiable quantum circuit implemented in PennyLane with the PyTorch interface. The cost function is defined as the negative fidelity of the reconstructed quantum states. The training optimizes the parameters of the variational layer and qutrit encoding unitaries via gradient-based backpropagation. Due to memory constraints, the model is trained on a reduced circuit with a single-qutrit latent space, trash, reference, and ancilla qutrits.

#### Training Hyperparameters 🐜

- **Training regime**: `torch.float32` (32-bit floating point), gradient-based optimization using backpropagation.
- **Optimizer**: Adam
- **Loss function**: Negative fidelity of the ancilla qutrit measurement.
- **Circuit configuration**: Single-qutrit latent space, three trash qutrits, three reference qutrits, one ancilla qutrit.
- **Input features**: Jet constituent kinematics including `η`, `φ`, `mass`, `energy`, `d0`, `dz`.
- **Number of layers**: Configurable variational layers; each layer includes `TAdd` gates and single-qutrit rotations (`RX`, `RY`, `RZ`).
- **Batching and epochs**: Configurable depending on dataset size and memory constraints.


## Evaluation 📈

### Testing Data, Factors & Metrics

#### Testing Data 📊

The model is evaluated on simulated jet datasets representing both background and signal events. Validation is performed on a held-out set (X_val). Test datasets include:
- Background jets (X_inf)
- H → bb jets (datos_HToBB)
- W → qq jets (datos_WToqq)
- Top-antitop jets (datos_TTBar)

#### Factors 🫧

Performance is disaggregated by the type of jet: background, H → bb, W → qq, and top-antitop (TTBar). Additional factors include the number of constituent particles per jet, with jets containing fewer than the required number of particles excluded from evaluation.

#### Metrics 🫧

The primary evaluation metric is the fidelity of the reconstructed quantum states, computed per jet from the ancilla measurement in the QAE circuit. Additionally, the area under the ROC curve (AUC) is used to quantify discriminative performance between signal and background events.

---

### Results ✨

Check for more details ➡️ https://arxiv.org/abs/2510.14001

The results obtained with qutrits are achieved in a three-step process: first, depending on the case, training is performed using 10,000 events taken from CMS proton-proton collision data or simulated CMS data; this training is validated in a second step using 2,500 different events from the same training dataset; finally, the signal type is inferred for 10,000 events of each type, obtained from simulated samples together with a test with 10,000 new events from the training dataset to make a comparative analysis.

The resulting quantum fidelity distributions are shown in Fig. 1. They enable analysis of the model's ability to compress input data into a latent space. For high fidelity ranges, the qutrit-based model not only has a higher concentration of values close to 97–100%, but it also shows a clearer distinction between the relevant physical signals ($W \rightarrow q\bar{q}$, $H \rightarrow b\bar{b}$ and $t \rightarrow bq\bar{q}$), thus showing the higher expressivity of qutrit systems. The distinction capability was quantified using the Jensen–Shannon (JS) distance as a metric, and the qutrit-based model ultimately exhibited larger distances between the signals, as expected and as illustrated in Table 1.

| Model          | Fid. Distributions | $\sqrt{\textrm{JS}}$ |
|----------------|-----------------|-------------------|
| Qubits R       | (H – W)         | 1.43 × 10⁻²       |
|                | (H – t)         | 1.28 × 10⁻²       |
|                | (W – t)         | 1.35 × 10⁻²       |
| Qutrits R      | (H – W)         | 5.06 × 10⁻²       |
|                | (H – t)         | 9.03 × 10⁻²       |
|                | (W – t)         | 1.02 × 10⁻¹       |
| Qubits S       | (H – W)         | 1.43 × 10⁻²       |
|                | (H – t)         | 1.28 × 10⁻²       |
|                | (W – t)         | 1.39 × 10⁻²       |
| Qutrits (A) S  | (H – W)         | 5.05 × 10⁻²       |
|                | (H – t)         | 9.03 × 10⁻²       |
|                | (W – t)         | 4.87 × 10⁻¹       |
| Qutrits (B) S  | (H – W)         | 5.23 × 10⁻²       |
|                | (H – t)         | 7.88 × 10⁻²       |
|                | (W – t)         | 9.59 × 10⁻²       |
| Qutrits (C) S  | (H – W)         | 5.03 × 10⁻²       |
|                | (H – t)         | 8.59 × 10⁻²       |
|                | (W – t)         | 9.50 × 10⁻²       |
| Qutrits (D) S  | (H – W)         | 4.71 × 10⁻²       |
|                | (H – t)         | 8.81 × 10⁻²       |
|                | (W – t)         | 9.15 × 10⁻²       |

Moreover, the average AUC scores over 100 executions obtained by each model are shown in Tables 2 and 3. Table 2 corresponds to models trained on CMS data, and Table 3 to models trained on simulated CMS Monte Carlo data. The AUC metric is used to analyse the model's capability to detect anomalies, where a higher score indicates better discrimination between signal and background events. 

| Model        | $W \rightarrow q\bar{q}$ | $H \rightarrow b\bar{b}$ | $t \rightarrow bq\bar{q}$ |
|--------------|-------------------------|--------------------------|---------------------------|
| QAE Qubits   | 0.622                   | 0.669                    | 0.776                     |
| QAE Qutrits  | 0.671                   | 0.714                    | 0.793                     |

| Model         | $W \rightarrow q\bar{q}$ | $H \rightarrow b\bar{b}$ | $t \rightarrow bq\bar{q}$ |
|---------------|-------------------------|--------------------------|---------------------------|
| QAE Qubits    | 0.733                   | 0.775                    | 0.846                     |
| QAE Qutrits A | 0.688                   | 0.731                    | 0.811                     |
| QAE Qutrits B | 0.722                   | 0.762                    | 0.833                     |
| QAE Qutrits C | 0.723                   | 0.763                    | 0.833                     |
| QAE Qutrits D | 0.723                   | 0.763                    | 0.833                     |

In the context of anomaly detection, the AUC score establishes a fidelity threshold that allows background events to be distinguished from signal events. High AUC scores reflect more subtle detection of patterns in the input data, allowing greater separation between physics signals and the dominant QCD background. The qutrit-based QAE shows higher performance compared to qubit-based models when the $\tau$-ratios and jet energy are included, or when longitudinal and transverse impact parameters are used for models trained on real CMS data.

As shown in Fig. 1 and Tables 2–3, the $t \rightarrow b q \bar{q}$ decay is identified as the most anomalous signal by all models, consistent with the expected jet topology: at LHC energies, electroweak-scale resonances such as top quarks, $W/Z$ bosons, and the Higgs boson are frequently produced in a boosted regime. Their decay products are collimated into a single large-radius jet. While QCD jets typically exhibit a single-prong core with diffuse soft radiation, boosted $W/Z$ and Higgs bosons yield a two-prong structure, and top quarks produce a three-prong configuration. The richer substructure and higher particle multiplicity in top-initiated jets create larger deviations from the QCD-like background, which the QAE detects as more anomalous events. Including the ratios $\tau_{12}$, $\tau_{23}$, and $\tau_{34}$ with jet energy enhances discrimination for $t \rightarrow b q \bar{q}$ events, leading models B, C, and D to achieve higher AUC values than model A.

---

#### Summary  🔖

A qutrit-based model for anomaly detection in CMS experiment data has been developed, and Majorana encoding for qutrit representation on unit spheres has been proven to be an effective way to represent the information on a unitary sphere for qutrit systems. Generalized gates have been added according to the new ternary paradigm, such as rotation gates and SWAP gates, which have been shown to give robust results. Despite the limitations encountered in PennyLane's ability to simulate qutrit circuits, in terms of memory consumption, similar performance equivalent to the qubit-based model - or in some cases even higher - has been achieved, and our model has stood out for its greater ability to discern between signals. As for future research, further tests should be carried out using data from other LHC experiments, such as ATLAS and LHCb, to evaluate how the new encoding and model behave with different datasets and BSM scenarios. In addition, different encodings and libraries must be studied and tested to find the most suitable combination for LHC data developments using qutrits. 

As for hardware implementation of qudit systems, recent studies, where a quantum algorithm was successfully implemented with a trapped ion qudit, obtaining competitive results, make the hardware implementation of qudit-based models seem like a certainty in the intermediate future. Additionally, the break-even point for Quantum Error Correction of qudit quantum memories, at which the lifetime of a qudit exceeds the lifetime of the constituents of the system, has been beaten in Ref.~\cite{evenBreak}, enhancing the usability conception of qudits in the long run and making studies like this one only the beginning of further developments.

The use of qutrit mixed states, using Majorana's generalised representation for mixed states, as well as higher-level systems such as ququarts, which can potentially provide greater expressiveness and thus improve the performance of our model for anomaly detection, is left for future work. It should be noted that these performance improvements come at an increased computational cost and not-straightforward implementation within the available simulators.

---

## Citation 📌

### BibTeX
```bibtex
@misc{laino2025qutritsphysicslhc,
  title        = {Qutrits for physics at the LHC},
  author       = {Miranda Carou Lai\~no and Veronika Chobanova and Miriam Lucio Mart\'inez},
  year         = {2025},
  eprint       = {2510.14001},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  url          = {https://arxiv.org/abs/2510.14001}
}
```

---

## Model Card Contact ☎️

Miranda Carou Laiño: 📩 micalai@alumni.uv.es
Veronika Chobanova: 📩 veronika.chobanova@cern.ch
Miriam Lucio Martínez: 📩 miriam.lucio.martinez@cern.ch
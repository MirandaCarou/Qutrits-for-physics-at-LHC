
# CESGA Scripts & HPC Model Execution

This repository is organized into two main parts, each corresponding to a different stage of the data processing and model evaluation pipeline used in this project.

---

## 1. Data Processing and Conversion Scripts

The first part of the repository contains the scripts used to process and convert large-scale datasets generated from **real and simulated CMS data**. 

The scripts located in:

Scripts_CESGA/Script_to_convert_data/


are responsible for:

- Loading massive datasets of real and simulated data.
- Cleaning and structuring the data.
- Converting the data into a format and language that can be directly understood and used in our QAEs.
- Automating the entire preprocessing pipeline.

These transformations are essential to ensure that both simulated and real experimental data share a consistent representation before being used for training and evaluation.

---

## 2. Model Execution on HPC (Finisterrae III)

The second part of the repository contains the implementations of the models exported to `.py` format for large-scale testing and benchmarking. These scripts are designed to perform **100 independent executions for each experiment**, allowing robust statistical evaluation of the results.

This section is located in:

Test_models_In_HPC/


It includes:

- Python implementations of qubit and qutrit models.
- Scripts for executing multiple runs automatically.
- Post-processing routines for later computation of performance metrics.

Due to the **high computational cost** of running 100 executions per experiment, all large-scale experiments were performed on **Finisterrae III**, the supercomputer of **CESGA (Centro de Supercomputación de Galicia)**.

---

## Summary

In short, this repository provides:

- ✅ Data preprocessing and conversion tools for massive real and simulated datasets.
- ✅ High-performance computing scripts to execute the quantum machine learning model
- ✅ Infrastructure to compute reliable metrics through large numbers of repeated experiments.
- ✅ Full integration with CESGA’s Finisterrae III supercomputing facilities.



# NLDD-CoT: Faithfulness Decay in Chain-of-Thought

**Official implementation of "Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought Reasoning" (Submitted to ACL 2026).**

This repository contains the mechanistic interpretability pipeline used to quantify **Normalized Logit Difference Decay (NLDD)**, identifying the "Reasoning Horizon" ($k^*$) where Large Language Models cease to rely on their generated Chain-of-Thought (CoT).

## Overview

This pipeline evaluates reasoning faithfulness across three complexity axes: **Syntactic** (Dyck-n), **Logical** (PrOntoQA), and **Arithmetic** (GSM8K). It implements a suite of behavioral and geometric metrics to detect when reasoning turns into post-hoc rationalization.

### Key Metrics Implemented
* **NLDD (Normalized Logit Difference Decay):** Measures the causal drop in model confidence when a specific reasoning step is corrupted, normalized by the model's intrinsic output variance ($S_{model}$).
* **TAS (Trajectory Alignment Score):** Quantifies the geometric efficiency of the reasoning path in the model's latent space.
* **RSA (Representational Similarity Analysis):** Measures the divergence between "clean" and "counterfactual" internal representations to detect computational drift.
* **Reasoning Horizon ($k^*$):** The step index where causal faithfulness (NLDD) peaks before degrading.

## Installation

The pipeline is self-contained. We recommend using a virtual environment with Python 3.10+.

# Install dependencies
pip install torch transformers numpy pandas scipy scikit-learn seaborn matplotlib tqdm datasets accelerate statsmodels

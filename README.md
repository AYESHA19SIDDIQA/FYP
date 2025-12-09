# EEG Classification with NeuroGATE

This repository provides a PyTorch implementation of the NeuroGATE deep learning model for EEG classification. It utilizes the CereProcess library for data loading, preprocessing, model training, and evaluation, demonstrated in a comprehensive example notebook.


---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Project Structure](#project-structure)
* [Dependencies](#dependencies)

---

## Features

* **EEG Preprocessing Pipelines**: Configurable filtering, resampling, and artifact rejection.
* **PyTorch Dataset**: Ready-to-use `torch.utils.data.Dataset` for EEG file formats.
* **NeuroGATE Model**: Reference implementation of the NeuroGATE architecture for classification tasks.
* **Training Utilities**: High-level training loop.

---

## Installation

This package is intended for local development and testing. To install:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/dll-ncai/NeuroGate_EEG_Classification.git
cd NeuroGate_EEG_Classification

#If you already cloned the repository without the flag, you can initialize the submodule manually:
git submodule update --init --recursive

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\\Scripts\\activate  # Windows

# Install requirements
cd cereprocess
pip install -r requirements.txt
```

---

## Quick Start

All code usage is demonstrated in the provided Jupyter notebook. After installation, launch:

```bash
jupyter notebook example.ipynb
```

The notebook walks through:

1. Loading and preprocessing EEG data
2. Initializing and training the NeuroGATE model
3. Inspecting results and metrics

---

## Project Structure

```plaintext
NeuroGate_EEG_Classification/
|-- cereprocess/          # Git submodule for utilities
|-- models
|   |-- __init__.py
|   `-- neurogate.py
|-- example.ipynb
|-- README.md
```

---

## Dependencies

* Python3
* numpy
* pandas
* mne
* torch
* scikit-learn
* tqdm
* matplotlib
* torchmetrics
* ipywidgets

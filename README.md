# Stress Detection from Typing Patterns Using Deep Learning

## Project Overview

This project builds a deep learning system to detect human stress levels based on typing behavior. The model does not use the actual typed content. Instead, it learns patterns from how a person types, such as speed, timing, and rhythm.

The goal is to classify typing sessions into three categories:

* Low stress
* Medium stress
* High stress

The project compares two models:

* A baseline Multi-Layer Perceptron (MLP)
* A Transformer-based model for sequence learning

This comparison helps understand whether sequence modeling improves performance over simple feature-based methods.

---

## Project Structure

```
stress-detection-typing/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── data_loader.py
│   ├── synthetic_data.py
│
├── models/
│   ├── mlp.py
│   ├── transformer.py
│
├── training/
│   ├── trainer.py
│
├── utils/
│   ├── metrics.py
│   ├── plots.py
│   ├── utils.py
│
└── outputs/
```

---

## Setup Instructions

### Step 1: Install Python

Make sure Python 3.8 or higher is installed.

Check version:

```
python --version
```

---

### Step 2: Install Dependencies

Install required libraries using:

```
pip install -r requirements.txt
```

This will install:

* torch
* numpy
* matplotlib
* scikit-learn
* seaborn

---

### Step 3: Verify Folder Structure

Make sure all files are placed exactly as shown in the project structure.
The `outputs/` folder will be created automatically if not present.

---

## How to Run the Project

Run the main script:

```
python main.py
```

This will execute the full pipeline:

1. Generate synthetic typing dataset
2. Train baseline MLP model
3. Train Transformer model
4. Evaluate both models
5. Generate results and visualizations

---

## Outputs Generated

After execution, the following files will be created inside the `outputs/` folder:

1. `mlp_accuracy.png`
   Shows training and validation accuracy for the MLP model

2. `transformer_accuracy.png`
   Shows training and validation accuracy for the Transformer model

3. `confusion_mlp_accuracy.png`
   Confusion matrix for MLP predictions

4. `confusion_transformer_accuracy.png`
   Confusion matrix for Transformer predictions

5. `model_comparison.png`
   Bar chart comparing final validation accuracy of both models

---

## Model Description

### Baseline Model (MLP)

The MLP model treats the typing sequence as a flattened vector.
It learns general patterns but does not capture temporal dependencies.

### Transformer Model

The Transformer processes typing data as a sequence.
It captures relationships across time steps using attention mechanisms, making it better suited for sequential behavior.

---

## Dataset Description

This project uses synthetic typing data generated programmatically.

Each sample represents:

* Time between key presses
* Duration of key presses
* Typing speed

The dataset includes variations to simulate different stress levels.

Data is split into:

* Training set (70%)
* Validation set (15%)
* Test set (15%)

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Confusion matrices are also generated to understand prediction errors.

---

## Notes

* The project runs on CPU by default
* No external datasets are required
* All results are reproducible by running the main script

---

## Summary

This project demonstrates how deep learning can be applied to behavioral data.
It compares simple and advanced models to understand the importance of sequence modeling in stress detection.

---


# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Project Overview

This repository contains the solution for the Neural Networks, Computer Vision, NLP, and AI Solution Design project.

The objective is to build and analyze a feed-forward neural network on a customer churn prediction dataset, demonstrating understanding of forward pass, loss calculation, backpropagation, and parameter tuning.

---

## Dataset

- **File used:** `customer_churn_nn.csv`
- **Rows:** 2,000 customer records
- **Columns:** 17 (16 features + 1 target)
- **Target:** `churn` (0 = retained, 1 = churned)

---

## Repository Structure

```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── evaluation_outputs.png
    ├── model_comparison_table.png
    └── model_comparison_table.csv
```

---

## Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Dataset Understanding — shape, features, missing values, target distribution | ✅ Done |
| Task 2 | Data Preprocessing — encoding, scaling, train/test split, class weights | ✅ Done |
| Task 3 | Neural Network Model Building — feed-forward NN with TensorFlow/Keras | ✅ Done |
| Task 4 | Training and Evaluation — accuracy, loss, confusion matrix | ✅ Done |
| Task 5 | Hyperparameter Experimentation — 4 experiments, comparison table | ✅ Done |
| Task 6 | Final Reflection — conceptual questions answered | ✅ Done |

---

## Model Architecture (Baseline)

```
Input Layer   → 15 features
Hidden Layer 1 → 64 neurons, ReLU activation
Dropout        → 20% (prevents overfitting)
Hidden Layer 2 → 32 neurons, ReLU activation
Dropout        → 20%
Output Layer   → 1 neuron, Sigmoid activation
```

- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (lr=0.001)
- **Total Parameters:** 3,137

---

## Hyperparameter Experiment Results

| Experiment | Train Acc | Test Acc | Churn Recall |
|---|---|---|---|
| Baseline: 2 layers (64-32), ReLU, lr=0.001 | 94.50% | 93.50% | 0.50 |
| Exp 2: Deeper (128-64-32), ReLU, lr=0.001 | 97.37% | 95.75% | 0.50 |
| Exp 3: Tanh activation, lr=0.0005, 80 epochs | 86.06% | 85.00% | 0.83 |
| Exp 4: High LR=0.01 (too aggressive) | 95.25% | 93.25% | 0.33 |

**Best for churn detection:** Experiment 3 (highest recall for minority class)

---

## Key Observations

- The dataset is highly imbalanced (98.5% not churned, 1.5% churned).
- Class weights were applied to prevent the model from ignoring the minority class.
- Exp 3 (lower LR, tanh activation, more epochs) produced the best recall for churners.
- High learning rate (Exp 4) caused unstable validation loss — demonstrating the importance of LR tuning.
- Dropout regularization reduced overfitting (train vs val gap is small).

---

## Final Reflection Summary

- **Weights and biases** are the learnable parameters. Weights determine feature importance; biases allow flexible shifting of activations.
- **Activation functions** (ReLU, Sigmoid) introduce non-linearity, allowing the model to learn complex patterns beyond simple linear relationships.
- **Learning rate too high** → unstable training, oscillating loss. **Too low** → extremely slow convergence.
- **Overfitting/Underfitting:** The baseline model shows mild overfitting (~2.5% train/val gap), controlled by Dropout layers.

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Make sure `customer_churn_nn.csv` is in the same directory as the notebook.

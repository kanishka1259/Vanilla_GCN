# Vanilla Graph Convolutional Network (GCN) from Scratch

A lightweight, transparent Python implementation of a **Vanilla Graph Convolutional Network (GCN)** as formulated by Kipf & Welling (ICLR 2017).

This repository is designed for academic study, coursework assignments, and reports. It exposes every step of the mathematical forward pass without hiding operations inside high-level GNN libraries (such as PyTorch Geometric or DGL).

---

## Features

1. **Part A — Manual Step-by-Step Forward Pass (NumPy)**:
   - Uses pure NumPy matrix operations.
   - Computes and prints every intermediate matrix ($A$, $\tilde{A}$, $\tilde{D}$, $\tilde{D}^{-1/2}$, $\hat{A}$, $X$, $\hat{A}X$, $\hat{A}XW^{(0)}$, $H^{(1)}$, $\hat{A}H^{(1)}W^{(1)}$, $H^{(2)}$).
   - Formats shapes and matrix values with fixed precision (4 decimal places) for manual verification and report copy-pasting.

2. **Part B — Trainable GCN (PyTorch)**:
   - Implements custom `VanillaGCNLayer` and 2-layer `VanillaGCN` `nn.Module` using standard PyTorch tensor ops.
   - Demonstrates gradient-based backpropagation and cross-entropy optimization on a small semi-supervised / supervised node classification task.

3. **Academic Deliverables**:
   - `vanilla_gcn.py`: Complete Python source code.
   - `gcn_output.txt`: Pre-captured console output from a clean execution run.
   - `report_notes.md`: Comprehensive, report-ready academic documentation with theoretical justifications and exact equations.

---

## Requirements & Dependencies

The implementation strictly uses standard numerical and deep learning frameworks:

- Python 3.8+
- `numpy`
- `torch`

To install the required dependencies, run:

```bash
pip install numpy torch
```

---

## How to Run

To execute the complete script (both Part A manual pass and Part B trainable model):

```bash
python vanilla_gcn.py
```

To save the console output to a text file:

```bash
python vanilla_gcn.py > gcn_output.txt
```

---

## File Overview

| File | Description |
| :--- | :--- |
| [`vanilla_gcn.py`](file:///e:/7th%20sem/ML%20with%20Graph/vanilla%20gcn/vanilla_gcn.py) | Runnable Python script containing NumPy manual forward pass and PyTorch trainable model. |
| [`gcn_output.txt`](file:///e:/7th%20sem/ML%20with%20Graph/vanilla%20gcn/gcn_output.txt) | Complete console output log from executing `vanilla_gcn.py`. |
| [`report_notes.md`](file:///e:/7th%20sem/ML%20with%20Graph/vanilla%20gcn/report_notes.md) | Formatted report notes covering theoretical formulas, equations, matrix tables, and explanations. |
| [`README.md`](file:///e:/7th%20sem/ML%20with%20Graph/vanilla%20gcn/README.md) | User guide and repository information. |

---

## Graph Benchmark Used

The example uses a 4-node undirected cycle graph ($N=4$):

```
0 ----- 1
|       |
|       |
2 ----- 3
```

- **Node Features ($X$)**: 2 features per node.
- **Classes**: 2 target classes (Nodes 0 & 1 belong to Class 0; Nodes 2 & 3 belong to Class 1).

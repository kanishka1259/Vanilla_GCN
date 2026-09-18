# Academic Report Notes: Vanilla Graph Convolutional Network (GCN)

This document provides academic reference material, rigorous mathematical derivations, theoretical explanations, and step-by-step numerical verifications for a **Vanilla Graph Convolutional Network (GCN)** based on the landmark paper by Kipf & Welling (ICLR 2017).

---

## 1. Definition of Vanilla GCN

A **Vanilla Graph Convolutional Network (GCN)** is a localized, first-order approximation of spectral graph convolutions on graphs. It generalizes standard convolutional neural networks (CNNs) from regular spatial grids (such as 2D image pixels) to irregular non-Euclidean graph topologies $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.

The core mechanism of a GCN is **neighborhood aggregation** (also called message passing): at every layer, each node updates its representation by aggregating feature vectors from its immediate 1-hop neighbors and combining them via a linear transformation followed by a non-linear activation function.

---

## 2. Formal Mathematical Formulation

The layer-wise propagation rule of a Vanilla GCN is defined as:

$$H^{(l+1)} = \sigma \left( \hat{A} H^{(l)} W^{(l)} \right)$$

Where the symmetric normalized adjacency matrix $\hat{A}$ is constructed via:

$$\tilde{A} = A + I_N$$

$$\tilde{D}_{ii} = \sum_{j=1}^N \tilde{A}_{ij}$$

$$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

---

## 3. Variable Notation & Definitions

| Variable | Dimension | Description |
| :--- | :--- | :--- |
| $N$ | $\mathbb{Z}^+$ | Total number of nodes in the graph ($|\mathcal{V}|$). |
| $A$ | $\mathbb{R}^{N \times N}$ | Unweighted binary adjacency matrix of graph $\mathcal{G}$ ($A_{ij}=1$ if $(i,j)\in\mathcal{E}$, else $0$). |
| $I_N$ | $\mathbb{R}^{N \times N}$ | $N \times N$ Identity matrix representing self-loops. |
| $\tilde{A}$ | $\mathbb{R}^{N \times N}$ | Adjacency matrix with added self-loops ($\tilde{A} = A + I_N$). |
| $\tilde{D}$ | $\mathbb{R}^{N \times N}$ | Diagonal degree matrix of $\tilde{A}$, where $\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$. |
| $\tilde{D}^{-1/2}$ | $\mathbb{R}^{N \times N}$ | Inverse square root diagonal degree matrix ($\tilde{D}^{-1/2}_{ii} = \frac{1}{\sqrt{\tilde{D}_{ii}}}$). |
| $\hat{A}$ | $\mathbb{R}^{N \times N}$ | Symmetrically normalized adjacency matrix ($\hat{A}_{ij} = \frac{\tilde{A}_{ij}}{\sqrt{\tilde{D}_{ii}\tilde{D}_{jj}}}$). |
| $X = H^{(0)}$ | $\mathbb{R}^{N \times F_0}$ | Input node feature matrix, where $F_0$ is the input feature dimension. |
| $H^{(l)}$ | $\mathbb{R}^{N \times F_l}$ | Node hidden representation matrix at layer $l$ ($F_l$ hidden units). |
| $W^{(l)}$ | $\mathbb{R}^{F_l \times F_{l+1}}$ | Trainable weight parameter matrix at layer $l$. |
| $\sigma(\cdot)$ | Function | Element-wise non-linear activation function (e.g., $\text{ReLU}$, $\text{softmax}$). |

---

## 4. Key Theoretical Justifications

### Why Self-Loops are Added ($\tilde{A} = A + I_N$)
In a standard adjacency matrix multiplication $A H^{(l)}$, row $i$ of the product is computed as:

$$(A H^{(l)})_i = \sum_{j \in \mathcal{N}(i)} H^{(l)}_j$$

This aggregates features **only** from neighboring nodes $\mathcal{N}(i)$, completely excluding node $i$'s own feature vector $H^{(l)}_i$. 
Adding the identity matrix $I_N$ creates self-loops ($\tilde{A}_{ii} = 1$), ensuring that during aggregation, a node retains its own features alongside those of its neighbors:

$$(\tilde{A} H^{(l)})_i = H^{(l)}_i + \sum_{j \in \mathcal{N}(i)} H^{(l)}_j$$

### Why Normalization is Performed ($\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$)
1. **Preventing Scale Explosion/Vanishing**: Without normalization, multiplying by $\tilde{A}$ repeatedly increases feature vector magnitudes proportionally to node degrees. High-degree nodes (hubs) produce huge values, causing numerical instability and gradient explosion.
2. **Symmetric Scaling**: The element-wise formula for $\hat{A}$ is:
   $$\hat{A}_{ij} = \frac{\tilde{A}_{ij}}{\sqrt{\tilde{D}_{ii} \tilde{D}_{jj}}}$$
   This normalizes the contribution of node $j$ to node $i$ by the geometric mean of their degrees ($\sqrt{\tilde{D}_{ii} \tilde{D}_{jj}}$), balancing incoming and outgoing feature signals across the network.

### How Neighborhood Aggregation Works
The matrix product $\hat{A} H^{(l)}$ performs weighted local feature aggregation:

$$(\hat{A} H^{(l)})_{i, :} = \sum_{j=1}^N \hat{A}_{ij} H^{(l)}_{j, :}$$

For each node $i$, its new feature vector is a convex combination of its 1-hop neighborhood representations. Stacking $L$ GCN layers enables each node to aggregate information from its $L$-hop neighborhood.

### How Node Classification is Obtained
For a $C$-class classification task:
1. The final layer computes unnormalized logits $Z^{(L)} = \hat{A} H^{(L-1)} W^{(L-1)} \in \mathbb{R}^{N \times C}$.
2. A row-wise softmax activation yields normalized class probabilities $H^{(L)}_{ic} \in [0, 1]$:
   $$H^{(L)}_{ic} = \frac{\exp(Z_{ic}^{(L)})}{\sum_{c'=1}^C \exp(Z_{ic'}^{(L)})}$$
3. The predicted class for node $i$ is:
   $$\hat{y}_i = \arg\max_{c \in \{0, \dots, C-1\}} H^{(L)}_{ic}$$

---

## 5. Complete Numerical Example (4-Node Cycle Graph)

### Input Graph Topology & Features
Consider a 4-node undirected cycle graph with edges $\mathcal{E} = \{(0,1), (1,3), (3,2), (2,0)\}$:

```
0 ----- 1
|       |
|       |
2 ----- 3
```

- **Node Features ($X \in \mathbb{R}^{4 \times 2}$)**:
  $$X = \begin{bmatrix} 1.0 & 0.5 \\ 0.8 & 0.2 \\ 0.1 & 0.9 \\ 0.0 & 1.0 \end{bmatrix}$$

- **Target Labels ($y$)**: Nodes 0 & 1 $\to$ Class 0; Nodes 2 & 3 $\to$ Class 1.

---

### Step-by-Step Matrix Calculations

#### Step 1: Adjacency Matrix ($A$) & Self-Loops ($\tilde{A}$)
$$A = \begin{bmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}, \quad \tilde{A} = A + I_4 = \begin{bmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{bmatrix}$$

#### Step 2: Degree Matrix ($\tilde{D}$) & Inverse Square Root ($\tilde{D}^{-1/2}$)
Since every node in $\tilde{A}$ has 3 connections (2 neighbors + 1 self-loop), $\tilde{D}_{ii} = 3$:

$$\tilde{D} = \begin{bmatrix} 3 & 0 & 0 & 0 \\ 0 & 3 & 0 & 0 \\ 0 & 0 & 3 & 0 \\ 0 & 0 & 0 & 3 \end{bmatrix}, \quad \tilde{D}^{-1/2} = \begin{bmatrix} 0.5774 & 0 & 0 & 0 \\ 0 & 0.5774 & 0 & 0 \\ 0 & 0 & 0.5774 & 0 \\ 0 & 0 & 0 & 0.5774 \end{bmatrix}$$

#### Step 3: Normalized Adjacency Matrix ($\hat{A}$)
$$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} = \begin{bmatrix} 0.3333 & 0.3333 & 0.3333 & 0.0000 \\ 0.3333 & 0.3333 & 0.0000 & 0.3333 \\ 0.3333 & 0.0000 & 0.3333 & 0.3333 \\ 0.0000 & 0.3333 & 0.3333 & 0.3333 \end{bmatrix}$$

#### Step 4: Layer 1 Forward Pass ($H^{(1)} = \text{ReLU}(\hat{A} X W^{(0)})$)
Given fixed weights $W^{(0)} = \begin{bmatrix} 0.5 & -0.2 \\ 0.1 & 0.8 \end{bmatrix}$:

1. **Aggregation ($\hat{A} X$)**:
   $$\hat{A} X = \begin{bmatrix} 0.6333 & 0.5333 \\ 0.6000 & 0.5667 \\ 0.3667 & 0.8000 \\ 0.3000 & 0.7000 \end{bmatrix}$$

2. **Linear Transformation ($\hat{A} X W^{(0)}$)**:
   $$\hat{A} X W^{(0)} = \begin{bmatrix} 0.3700 & 0.3000 \\ 0.3567 & 0.3333 \\ 0.2633 & 0.5667 \\ 0.2200 & 0.5000 \end{bmatrix}$$

3. **Activation ($H^{(1)} = \text{ReLU}(\hat{A} X W^{(0)})$)**:
   $$H^{(1)} = \begin{bmatrix} 0.3700 & 0.3000 \\ 0.3567 & 0.3333 \\ 0.2633 & 0.5667 \\ 0.2200 & 0.5000 \end{bmatrix}$$

#### Step 5: Layer 2 Forward Pass ($H^{(2)} = \text{softmax}(\hat{A} H^{(1)} W^{(1)})$)
Given fixed weights $W^{(1)} = \begin{bmatrix} 1.2 & -0.8 \\ -0.5 & 0.9 \end{bmatrix}$:

1. **Logits ($\hat{A} H^{(1)} W^{(1)}$)**:
   $$\text{Logits} = \begin{bmatrix} 0.1960 & 0.0960 \\ 0.1898 & 0.0876 \\ 0.1136 & 0.1824 \\ 0.1027 & 0.1960 \end{bmatrix}$$

2. **Final Probabilities ($H^{(2)} = \text{softmax}(\text{Logits})$)**:
   $$H^{(2)} = \begin{bmatrix} 0.5250 & 0.4750 \\ 0.5255 & 0.4745 \\ 0.4828 & 0.5172 \\ 0.4767 & 0.5233 \end{bmatrix}$$

#### Step 6: Node Predictions
| Node | Class 0 Probability | Class 1 Probability | Predicted Class | Ground Truth |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.5250 | 0.4750 | **0** | 0 |
| **1** | 0.5255 | 0.4745 | **0** | 0 |
| **2** | 0.4828 | 0.5172 | **1** | 1 |
| **3** | 0.4767 | 0.5233 | **1** | 1 |

---

## 6. Real-World Applications of GCNs

1. **Node Classification in Citation Networks**: Categorizing academic papers (e.g., Cora, Citeseer, PubMed) based on paper content features and citation link graphs.
2. **Molecular Property Prediction**: Modeling chemical molecules as graphs (atoms as nodes, bonds as edges) to predict toxicity, solubility, or drug activity.
3. **Fraud Detection in Financial Networks**: Detecting fraudulent accounts or transactions in banking networks by analyzing transaction links and feature vectors.
4. **Recommendation Systems**: Learning user and item representations in bipartite interaction graphs to recommend products or content.

# Vanilla Graph Convolutional Network (GCN): Mathematical Model and Implementation

---

# 1. INTRODUCTION

In machine learning, conventional algorithms typically process independent tabular data, grid-like images, or sequential text. However, many real-world datasets naturally exist as graphs, such as social networks, citation networks, and chemical molecular structures. Graph data consists of entities (nodes) and connections between them (edges).

Traditional machine learning models process each sample in isolation and ignore topological relationships between data points. Graph Neural Networks (GNNs) address this limitation by combining node feature vectors with structural graph connectivity. A Graph Convolutional Network (GCN) extends standard convolution from regular image grids to irregular graph structures. The Vanilla GCN, introduced by Kipf and Welling (2017), provides a basic yet powerful mathematical formulation for aggregating features from direct neighbors on a graph.

---

# 2. VANILLA GCN

A Vanilla Graph Convolutional Network is a basic variant of GNNs that uses a first-order local spectral graph convolution approximation. The term "vanilla" indicates that the network uses standard neighborhood aggregation with fixed symmetric normalization, without complex mechanisms like attention weights, multi-hop jumping knowledge, or dynamic sampling.

The fundamental idea of a GCN is neighborhood aggregation (also called message passing). In each GCN layer, every node collects feature vectors from its immediate 1-hop neighbors, averages/scales them based on node degrees, applies a linear weight transformation, and passes the result through a non-linear activation function. When multiple GCN layers are stacked, nodes gradually incorporate feature information from higher-hop neighborhoods (e.g., 2-layer GCN aggregates 2-hop neighbor information).

---

# 3. MATHEMATICAL MODEL OF VANILLA GCN

## 3.1 Graph Representation

A graph is represented as:

$$G = (V, E)$$

where:
- $V$: set of $N$ nodes (vertices), where $V = \{0, 1, \dots, N-1\}$.
- $E$: set of edges connecting pairs of nodes.
- $N$: total number of nodes ($N = |V|$).
- $A$: unweighted binary adjacency matrix of size $N \times N$.

The entries of the adjacency matrix $A$ are defined as:

$$A_{ij} = \begin{cases} 1, & \text{if an edge exists between node } i \text{ and node } j \\ 0, & \text{otherwise} \end{cases}$$

For the 4-node undirected cycle graph used in our implementation, the adjacency matrix $A$ is:

$$A = \begin{bmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}$$

---

## 3.2 Node Feature Matrix

Each node in the graph is assigned an initial feature vector. For $N$ nodes where each node has $F$ input features, the node feature matrix $X$ is defined as:

$$X \in \mathbb{R}^{N \times F}$$

In our implementation ($N=4, F=2$), the feature matrix $X$ is:

$$X = \begin{bmatrix} 1.0 & 0.5 \\ 0.8 & 0.2 \\ 0.1 & 0.9 \\ 0.0 & 1.0 \end{bmatrix}$$

Here, row $i$ represents the 2-dimensional feature vector of node $i$.

---

## 3.3 Adding Self-Loops

In standard adjacency matrix multiplication $A X$, a node aggregates features only from its neighbors, ignoring its own feature vector. To fix this, self-loops are added to every node by adding the identity matrix $I$:

$$\tilde{A} = A + I$$

where $I$ is an $N \times N$ identity matrix. The self-loop adjacency matrix $\tilde{A}$ from our implementation is:

$$\tilde{A} = \begin{bmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{bmatrix}$$

Adding self-loops ensures that each node retains its own features during neighborhood aggregation.

---

## 3.4 Degree Matrix

The degree matrix $\tilde{D}$ is a diagonal matrix containing the row sums of $\tilde{A}$ (the degree of each node including its self-loop):

$$\tilde{D}_{ii} = \sum_{j=1}^N \tilde{A}_{ij}$$

For our graph, each node has 2 neighbors plus 1 self-loop, giving a degree of 3 for all nodes:

$$\tilde{D} = \begin{bmatrix} 3 & 0 & 0 & 0 \\ 0 & 3 & 0 & 0 \\ 0 & 0 & 3 & 0 \\ 0 & 0 & 0 & 3 \end{bmatrix}$$

The degree matrix is necessary to compute normalization factors so that high-degree nodes do not cause feature values to explode.

---

## 3.5 Normalized Adjacency Matrix

To prevent scale explosion and balance contributions between connected nodes of different degrees, symmetric normalization is applied:

$$\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$$

First, the inverse square root degree matrix $\tilde{D}^{-\frac{1}{2}}$ is calculated:

$$\tilde{D}^{-\frac{1}{2}} = \begin{bmatrix} 0.5774 & 0 & 0 & 0 \\ 0 & 0.5774 & 0 & 0 \\ 0 & 0 & 0.5774 & 0 \\ 0 & 0 & 0 & 0.5774 \end{bmatrix}$$

Multiplying $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ yields the normalized adjacency matrix $\hat{A}$:

$$\hat{A} = \begin{bmatrix} 0.3333 & 0.3333 & 0.3333 & 0.0000 \\ 0.3333 & 0.3333 & 0.0000 & 0.3333 \\ 0.3333 & 0.0000 & 0.3333 & 0.3333 \\ 0.0000 & 0.3333 & 0.3333 & 0.3333 \end{bmatrix}$$

Each non-zero entry $\hat{A}_{ij} = \frac{1}{\sqrt{\tilde{D}_{ii} \tilde{D}_{jj}}} = \frac{1}{\sqrt{3 \times 3}} = \frac{1}{3} \approx 0.3333$.

---

## 3.6 GCN Layer

The layer propagation rule for Vanilla GCN is given by:

$$H^{(l+1)} = \sigma \left( \hat{A} H^{(l)} W^{(l)} \right)$$

where:
- $H^{(l)}$: node feature matrix at layer $l$ ($H^{(0)} = X$).
- $\hat{A}$: normalized adjacency matrix.
- $W^{(l)}$: trainable weight matrix for layer $l$.
- $\sigma$: non-linear activation function (such as ReLU or Softmax).

Order of operations in a single GCN layer:
1. **Neighborhood Aggregation**: Compute $\hat{A} H^{(l)}$ by multiplying the normalized adjacency matrix with the node representations.
2. **Linear Transformation**: Multiply aggregated features by the weight matrix $W^{(l)}$ to project into a new feature space.
3. **Activation**: Pass the result through activation function $\sigma(\cdot)$.

---

# 4. TWO-LAYER VANILLA GCN MODEL

Our implementation uses a 2-layer GCN architecture for node classification into 2 target classes.

### Layer 1 Equation
$$H^{(1)} = \text{ReLU} \left( \hat{A} X W^{(0)} \right)$$

Using fixed weight matrix $W^{(0)} \in \mathbb{R}^{2 \times 2}$:

$$W^{(0)} = \begin{bmatrix} 0.5 & -0.2 \\ 0.1 & 0.8 \end{bmatrix}$$

The intermediate aggregated feature matrix $\hat{A} X$ is:

$$\hat{A} X = \begin{bmatrix} 0.6333 & 0.5333 \\ 0.6000 & 0.5667 \\ 0.3667 & 0.8000 \\ 0.3000 & 0.7000 \end{bmatrix}$$

Multiplying by $W^{(0)}$ and applying ReLU activation yields $H^{(1)}$:

$$H^{(1)} = \begin{bmatrix} 0.3700 & 0.3000 \\ 0.3567 & 0.3333 \\ 0.2633 & 0.5667 \\ 0.2200 & 0.5000 \end{bmatrix}$$

### Layer 2 Equation
$$H^{(2)} = \text{softmax} \left( \hat{A} H^{(1)} W^{(1)} \right)$$

Using fixed weight matrix $W^{(1)} \in \mathbb{R}^{2 \times 2}$:

$$W^{(1)} = \begin{bmatrix} 1.2 & -0.8 \\ -0.5 & 0.9 \end{bmatrix}$$

The unnormalized logits $\hat{A} H^{(1)} W^{(1)}$ are:

$$\text{Logits} = \begin{bmatrix} 0.1960 & 0.0960 \\ 0.1898 & 0.0876 \\ 0.1136 & 0.1824 \\ 0.1027 & 0.1960 \end{bmatrix}$$

Applying row-wise softmax computes the probability distribution matrix $H^{(2)}$:

$$H^{(2)} = \begin{bmatrix} 0.5250 & 0.4750 \\ 0.5255 & 0.4745 \\ 0.4828 & 0.5172 \\ 0.4767 & 0.5233 \end{bmatrix}$$

### Decision Rule
The predicted class label $\hat{y}_i$ for node $i$ is obtained by selecting the class index with maximum probability:

$$\hat{y}_i = \arg\max_c H^{(2)}_{ic}$$

---

# 5. EXAMPLE SIMULATION

This simulation summarizes the step-by-step numerical output produced by our Python program.

### Step 1: Input Graph
- Nodes ($N$): 4
- Edges: $(0,1), (1,3), (3,2), (2,0)$
- Structure: 4-node cycle graph

### Step 2: Input Features ($X$)
$$X = \begin{bmatrix} 1.0 & 0.5 \\ 0.8 & 0.2 \\ 0.1 & 0.9 \\ 0.0 & 1.0 \end{bmatrix}$$

### Step 3: Adjacency Matrix ($A$)
$$A = \begin{bmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}$$

### Step 4: Self-Loops ($\tilde{A} = A + I$)
$$\tilde{A} = \begin{bmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{bmatrix}$$

### Step 5: Degree Matrix ($\tilde{D}$)
$$\tilde{D} = \begin{bmatrix} 3 & 0 & 0 & 0 \\ 0 & 3 & 0 & 0 \\ 0 & 0 & 3 & 0 \\ 0 & 0 & 0 & 3 \end{bmatrix}$$

### Step 6: Normalization ($\hat{A}$)
$$\hat{A} = \begin{bmatrix} 0.3333 & 0.3333 & 0.3333 & 0.0000 \\ 0.3333 & 0.3333 & 0.0000 & 0.3333 \\ 0.3333 & 0.0000 & 0.3333 & 0.3333 \\ 0.0000 & 0.3333 & 0.3333 & 0.3333 \end{bmatrix}$$

### Step 7: First GCN Layer ($H^{(1)}$)
$$H^{(1)} = \text{ReLU}(\hat{A} X W^{(0)}) = \begin{bmatrix} 0.3700 & 0.3000 \\ 0.3567 & 0.3333 \\ 0.2633 & 0.5667 \\ 0.2200 & 0.5000 \end{bmatrix}$$

### Step 8: Second GCN Layer ($H^{(2)}$)
$$H^{(2)} = \text{softmax}(\hat{A} H^{(1)} W^{(1)}) = \begin{bmatrix} 0.5250 & 0.4750 \\ 0.5255 & 0.4745 \\ 0.4828 & 0.5172 \\ 0.4767 & 0.5233 \end{bmatrix}$$

### Step 9: Final Prediction Table

| Node | Class 0 Probability | Class 1 Probability | Predicted Class |
| :---: | :---: | :---: | :---: |
| **0** | 0.5250 | 0.4750 | **0** |
| **1** | 0.5255 | 0.4745 | **0** |
| **2** | 0.4828 | 0.5172 | **1** |
| **3** | 0.4767 | 0.5233 | **1** |

---

# 6. IMPLEMENTATION

The implementation is written in Python using two core libraries:
- **NumPy**: used for manual matrix operations and step-by-step forward pass verification.
- **PyTorch**: used to define trainable weight parameters and train the network via automatic differentiation.

### Code-to-Equation Mapping

1. **Adjacency & Self-Loops**:
   `A_tilde = A + np.eye(4)` directly computes $\tilde{A} = A + I$.

2. **Degree Matrix & Inverse Square Root**:
   `degrees = np.sum(A_tilde, axis=1)` and `D_tilde_inv_sqrt = np.diag(np.power(degrees, -0.5))` compute $\tilde{D}^{-\frac{1}{2}}$.

3. **Normalized Adjacency**:
   `A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt` computes $\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$.

4. **Layer 1 Forward Pass**:
   `H1 = np.maximum(0, A_hat @ X @ W0)` corresponds to $H^{(1)} = \text{ReLU}(\hat{A} X W^{(0)})$.

5. **Layer 2 & Softmax**:
   `logits = A_hat @ H1 @ W1` followed by `H2 = softmax(logits)` computes $H^{(2)} = \text{softmax}(\hat{A} H^{(1)} W^{(1)})$.

6. **PyTorch Training**:
   In Part B of the program, `VanillaGCNLayer` defines $W$ as a `nn.Parameter`. The model is trained using `nn.CrossEntropyLoss()` and `optim.Adam(lr=0.05)` for 100 epochs, reducing loss from 0.8175 to 0.0510 and reaching 100% accuracy.

---

# 7. RESULTS AND OBSERVATIONS

- **Manual Forward Pass**: With fixed weights, the model correctly assigns Node 0 and Node 1 to Class 0, and Node 2 and Node 3 to Class 1.
- **Trained Model Results**: After 100 training epochs in PyTorch, the predicted class probabilities become highly confident:
  - Node 0: Class 0 Prob = 0.9542 (Class 0)
  - Node 1: Class 0 Prob = 0.9579 (Class 0)
  - Node 2: Class 1 Prob = 0.9299 (Class 1)
  - Node 3: Class 1 Prob = 0.9677 (Class 1)
- **Effect of Neighborhood Aggregation**: Aggregating features across connected nodes smooths feature differences between neighboring nodes, allowing the network to group structural clusters together.
- **Illustrative Nature**: This example uses a small 4-node graph designed to demonstrate mathematical matrix operations clearly, rather than evaluate large-scale benchmark performance.

---

# 8. APPLICATIONS OF GCN

1. **Node Classification**: Categorizing scientific papers in citation graphs based on text features and citation links.
2. **Link Prediction**: Predicting missing connections between users in social networks or items in e-commerce graphs.
3. **Recommendation Systems**: Learning joint user-item embeddings in interaction graphs for personalized recommendations.
4. **Social Network Analysis**: Identifying communities, influential nodes, or interest groups in social graphs.
5. **Traffic Prediction**: Forecasting traffic flow by modeling road intersections as nodes and road segments as edges.
6. **Molecular Property Prediction**: Modeling atoms as nodes and bonds as edges to predict toxicity or chemical reactivity.
7. **Knowledge Graphs**: Predicting missing relations or entities in structured knowledge bases.
8. **Fraud Detection**: Spotting suspicious financial transactions or fake accounts by analyzing graph connection patterns.

---

# 9. ADVANTAGES AND LIMITATIONS

### Advantages
- Combines both node features and structural graph connectivity effectively.
- Uses degree normalization to maintain numerical stability during feature aggregation.
- Learns low-dimensional node representations useful for downstream tasks.
- Parameter sharing across nodes makes GCN computationally efficient compared to fully connected networks.

### Limitations
- Deep GCNs with many layers suffer from over-smoothing, where all node representations become nearly identical.
- Full-batch GCN requires loading the entire adjacency matrix into memory, making it difficult to scale to massive graphs without mini-batch sampling.
- Assumes graph connectivity reflects feature similarity (homophily assumption).
- Fixed edge weighting cannot dynamically assign different importance to different neighbors without attention mechanisms.

---

# 10. CONCLUSION

This assignment presented the mathematical model and step-by-step implementation of a Vanilla Graph Convolutional Network (GCN). By adding self-loops and applying symmetric degree normalization, GCN performs effective neighborhood feature aggregation. Our Python implementation demonstrated the exact mathematical forward pass using NumPy and showed gradient-based optimization using PyTorch on a 4-node graph. The final outputs confirmed how GCN computes node representations and assigns nodes to their correct target classes.

---

# 11. REFERENCES

1. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. International Conference on Learning Representations (ICLR).
2. Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., & Sun, M. (2020). *Graph neural networks: A review of methods and applications*. AI Open, 1, 57-81.

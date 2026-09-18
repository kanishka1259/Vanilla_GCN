"""
Vanilla Graph Convolutional Network (GCN) Implementation from Scratch.

This script demonstrates the complete mathematical inner workings of a Vanilla GCN
as introduced by Kipf & Welling (ICLR 2017).

Part A: Manual Forward-Pass Demonstration (Pure NumPy)
Part B: Trainable GCN Model (PyTorch)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Format printing helper for NumPy matrices
np.set_printoptions(precision=4, suppress=True, linewidth=100)
torch.set_printoptions(precision=4, sci_mode=False)


def softmax_numpy(x):
    """Numerically stable softmax for NumPy array (row-wise)."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu_numpy(x):
    """ReLU activation function for NumPy array."""
    return np.maximum(0, x)


def run_part_a_manual_forward_pass():
    print("=" * 60)
    print("VANILLA GCN — FORWARD PASS (NUMPY DEMONSTRATION)")
    print("=" * 60)

    # ----------------------------------------------------
    # 1. INPUT GRAPH
    # ----------------------------------------------------
    print("\n1. INPUT GRAPH")
    print("-" * 30)
    num_nodes = 4
    edges = [(0, 1), (1, 3), (3, 2), (2, 0)]
    print(f"Number of nodes (N): {num_nodes}")
    print(f"Edges (E): {edges}")
    print("Graph Structure: 4-node cycle graph (0 - 1 - 3 - 2 - 0)")

    # ----------------------------------------------------
    # 2. ADJACENCY MATRIX (A)
    # ----------------------------------------------------
    print("\n2. ADJACENCY MATRIX")
    print("-" * 30)
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    print(f"A shape: {A.shape}")
    print("A (Adjacency Matrix):")
    print(A)
    print("\nExplanation: A[i, j] = 1 if an undirected edge connects node i and node j; otherwise 0.")

    # ----------------------------------------------------
    # 3. ADD SELF-LOOPS (A_tilde)
    # ----------------------------------------------------
    print("\n3. ADJACENCY MATRIX WITH SELF-LOOPS")
    print("-" * 30)
    I = np.eye(num_nodes, dtype=np.float64)
    A_tilde = A + I

    print(f"A_tilde shape: {A_tilde.shape}")
    print("A_tilde = A + I:")
    print(A_tilde)
    print("\nExplanation: Self-loops ensure that each node incorporates its own feature vector during aggregation.")

    # ----------------------------------------------------
    # 4. DEGREE MATRIX (D_tilde)
    # ----------------------------------------------------
    print("\n4. DEGREE MATRIX")
    print("-" * 30)
    degrees = np.sum(A_tilde, axis=1)
    D_tilde = np.diag(degrees)

    print(f"D_tilde shape: {D_tilde.shape}")
    print("D_tilde (Degree Matrix of A_tilde):")
    print(D_tilde)
    print(f"Node degrees (including self-loops): {degrees}")

    # ----------------------------------------------------
    # 5. NORMALIZED ADJACENCY MATRIX (A_hat)
    # ----------------------------------------------------
    print("\n5. NORMALIZED ADJACENCY MATRIX")
    print("-" * 30)
    # D_tilde^(-1/2) calculation
    d_inv_sqrt = np.power(degrees, -0.5)
    D_tilde_inv_sqrt = np.diag(d_inv_sqrt)

    # A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
    A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

    print(f"D^(-1/2) shape: {D_tilde_inv_sqrt.shape}")
    print("D_tilde^(-1/2):")
    print(D_tilde_inv_sqrt)

    print(f"\nA_hat shape: {A_hat.shape}")
    print("A_hat = D_tilde^(-1/2) @ A_tilde @ D_tilde^(-1/2):")
    print(A_hat)
    print("\nExplanation: Symmetric normalization scales edge weights by 1 / sqrt(deg(i) * deg(j)).")

    # ----------------------------------------------------
    # 6. NODE FEATURE MATRIX (X)
    # ----------------------------------------------------
    print("\n6. NODE FEATURES")
    print("-" * 30)
    # Define a simple 4x2 node feature matrix
    X = np.array([
        [1.0, 0.5],   # Node 0
        [0.8, 0.2],   # Node 1
        [0.1, 0.9],   # Node 2
        [0.0, 1.0]    # Node 3
    ], dtype=np.float64)

    print(f"X shape: {X.shape}")
    print("X (Node Feature Matrix):")
    print(X)

    # ----------------------------------------------------
    # 7. FIRST GCN LAYER
    # ----------------------------------------------------
    print("\n7. FIRST GCN LAYER")
    print("-" * 30)
    # Fixed weight matrix W0 (shape: 2x2)
    W0 = np.array([
        [0.5, -0.2],
        [0.1,  0.8]
    ], dtype=np.float64)
    print(f"W0 shape: {W0.shape}")
    print("W0 (Layer 1 Weight Matrix):")
    print(W0)

    # Aggregation: A_hat @ X
    AX = A_hat @ X
    print(f"\nA_hat X shape: {AX.shape}")
    print("A_hat X (Aggregated Node Features):")
    print(AX)

    # Linear transformation: A_hat @ X @ W0
    AXW0 = AX @ W0
    print(f"\nA_hat X W0 shape: {AXW0.shape}")
    print("A_hat X W0 (Linear Transformation):")
    print(AXW0)

    # Non-linear activation: H1 = ReLU(A_hat @ X @ W0)
    H1 = relu_numpy(AXW0)
    print(f"\nH1 shape: {H1.shape}")
    print("H1 = ReLU(A_hat X W0) (Layer 1 Output):")
    print(H1)

    # ----------------------------------------------------
    # 8. SECOND GCN LAYER
    # ----------------------------------------------------
    print("\n8. SECOND GCN LAYER")
    print("-" * 30)
    # Fixed weight matrix W1 (shape: 2x2)
    W1 = np.array([
        [ 1.2, -0.8],
        [-0.5,  0.9]
    ], dtype=np.float64)
    print(f"W1 shape: {W1.shape}")
    print("W1 (Layer 2 Weight Matrix):")
    print(W1)

    # Aggregation and Transformation: A_hat @ H1 @ W1
    AH1W1 = A_hat @ H1 @ W1
    print(f"\nA_hat H1 W1 shape: {AH1W1.shape}")
    print("A_hat H1 W1 (Logits before Softmax):")
    print(AH1W1)

    # Activation: H2 = Softmax(A_hat @ H1 @ W1)
    H2 = softmax_numpy(AH1W1)
    print(f"\nH2 shape: {H2.shape}")
    print("H2 = Softmax(A_hat H1 W1) (Final Probability Matrix):")
    print(H2)

    # ----------------------------------------------------
    # 9. FINAL CLASS PROBABILITIES & PREDICTIONS
    # ----------------------------------------------------
    print("\n9. FINAL CLASS PROBABILITIES & NODE PREDICTIONS")
    print("-" * 30)
    predictions = np.argmax(H2, axis=1)

    print(f"{'Node':^6} | {'Class 0 Prob':^15} | {'Class 1 Prob':^15} | {'Predicted Class':^15}")
    print("-" * 58)
    for node_idx in range(num_nodes):
        p0 = H2[node_idx, 0]
        p1 = H2[node_idx, 1]
        pred = predictions[node_idx]
        print(f"{node_idx:^6} | {p0:^15.4f} | {p1:^15.4f} | {pred:^15}")

    print("\nManual Forward Pass Complete.\n")
    return A_hat, X


# --------------------------------------------------------
# PART B: TRAINABLE GCN IN PYTORCH
# --------------------------------------------------------
class VanillaGCNLayer(nn.Module):
    """Custom Vanilla GCN Layer implementing H^(l+1) = A_hat * H^(l) * W^(l)."""
    def __init__(self, in_features, out_features):
        super(VanillaGCNLayer, self).__init__()
        # Trainable weight parameter W
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, A_hat, H):
        # Matrix multiplication: A_hat @ H @ W
        return torch.matmul(torch.matmul(A_hat, H), self.W)


class VanillaGCN(nn.Module):
    """2-Layer Vanilla Graph Convolutional Network."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(VanillaGCN, self).__init__()
        self.gcn1 = VanillaGCNLayer(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gcn2 = VanillaGCNLayer(hidden_dim, out_dim)

    def forward(self, A_hat, X):
        H1 = self.relu(self.gcn1(A_hat, X))
        logits = self.gcn2(A_hat, H1)
        return logits


def run_part_b_trainable_gcn(A_hat_np, X_np):
    print("=" * 60)
    print("VANILLA GCN — TRAINABLE MODEL (PYTORCH)")
    print("=" * 60)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Convert NumPy matrices to PyTorch tensors
    A_hat_tensor = torch.tensor(A_hat_np, dtype=torch.float32)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    # Ground truth node labels for classification (Nodes 0,1 -> Class 0, Nodes 2,3 -> Class 1)
    y_target = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    print("\nDataset for Training:")
    print(f"Node Features X shape: {X_tensor.shape}")
    print(f"Node Labels y: {y_target.tolist()} (Nodes 0,1: Class 0; Nodes 2,3: Class 1)")

    # Instantiate model
    in_dim = 2
    hidden_dim = 4
    out_dim = 2
    model = VanillaGCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    print("\nModel Architecture:")
    print(model)

    print("\nBeginning Training:")
    print("-" * 50)
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(A_hat_tensor, X_tensor)
        loss = criterion(logits, y_target)

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 20 == 0:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc = (preds == y_target).float().mean().item() * 100.0
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {acc:.1f}%")

    # Evaluation
    model.eval()
    with torch.no_grad():
        final_logits = model(A_hat_tensor, X_tensor)
        final_probs = torch.softmax(final_logits, dim=1)
        final_preds = torch.argmax(final_probs, dim=1)

    print("\nFinal Trained Predictions:")
    print("-" * 58)
    print(f"{'Node':^6} | {'Target':^8} | {'Class 0 Prob':^14} | {'Class 1 Prob':^14} | {'Predicted':^10}")
    print("-" * 58)
    for node_idx in range(4):
        t = y_target[node_idx].item()
        p0 = final_probs[node_idx, 0].item()
        p1 = final_probs[node_idx, 1].item()
        pred = final_preds[node_idx].item()
        print(f"{node_idx:^6} | {t:^8} | {p0:^14.4f} | {p1:^14.4f} | {pred:^10}")

    print("\nTraining Complete.")
    print("=" * 60)


if __name__ == "__main__":
    A_hat_np, X_np = run_part_a_manual_forward_pass()
    run_part_b_trainable_gcn(A_hat_np, X_np)

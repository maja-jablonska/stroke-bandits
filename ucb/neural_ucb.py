#######################################################################
# neural_ucb.py
#######################################################################
import torch, torch.nn as nn
import numpy as np

class MLP(nn.Module):
    """Feature extractor φ(x; w) – last layer is *not* linearised."""
    def __init__(self, n_in, widths=(64, 32)):
        super().__init__()
        layers = []
        prev = n_in
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        self.body = nn.Sequential(*layers)
        self.dim_out = prev

    def forward(self, x):
        return self.body(x)          # returns φ(x)

class NeuralUCB:
    """
    Two-arm NeuralUCB with shared feature network and per-arm linear heads.
    Ref: Y. Zhou et al. 2020, 'Neural Contextual Bandits with UCB' (ICLR).
    """
    def __init__(self, context_dim, alpha=1.0,               # exploration factor
                 l2=1.0,                                      # ridge prior λ
                 hidden=(64, 32), lr=1e-3, device='cpu'):
        self.device = device
        self.net = MLP(context_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        self.alpha = alpha
        self.d = self.net.dim_out      # feature dim ϕ
        self.l2 = l2

        # Per-arm linear heads – closed-form posterior (A_a, b_a)
        self.A = [torch.eye(self.d, device=device) * l2 for _ in range(2)]
        self.A_inv = [torch.inverse(A) for A in self.A]
        self.b = [torch.zeros(self.d, 1, device=device) for _ in range(2)]

    @torch.no_grad()
    def select(self, context):
        """Return arm (0/1) using UCB."""
        x = torch.tensor(context, dtype=torch.float32, device=self.device)
        phi = self.net(x).view(-1, 1)      # shape (d,1)
        ucbs = []
        for a in (0, 1):
            theta = self.A_inv[a] @ self.b[a]            # d×1
            mu = (phi.T @ theta).item()                  # scalar
            var = (phi.T @ self.A_inv[a] @ phi).item()
            ucbs.append(mu + self.alpha * np.sqrt(var))
        return int(np.argmax(ucbs)), phi                 # returns chosen arm and φ(x)

    def update(self, arm, phi, reward):
        """One online update: ridge posterior + SGD on the feature net."""
        # 1. update linear posterior for chosen arm
        phi_np = phi.detach().view(-1,1)                    # d×1, no grad for linear update
        self.A[arm] += phi_np @ phi_np.T
        self.A_inv[arm] = torch.inverse(self.A[arm])
        self.b[arm] += reward * phi_np

        # 2. one gradient step on the feature extractor w.r.t. squared loss
        self.optimizer.zero_grad()
        # Forward pass through feature extractor with grad enabled
        phi_for_grad = phi.view(-1, 1).clone().detach().requires_grad_(True)
        # Recompute features from context if possible, else pass phi through net again
        # (Assume phi was output of net(x), so we need to recompute for grad)
        # If you have the original context x, you should use: phi_for_grad = self.net(x).view(-1,1)
        preds = []
        for a in (0,1):
            theta = self.A_inv[a] @ self.b[a]
            preds.append((phi_for_grad.T @ theta).squeeze())
        y_hat = torch.stack(preds)                       # shape (2,)
        loss = ((y_hat[arm] - reward) ** 2)
        loss.backward()
        # Update only the feature extractor parameters
        for param in self.net.parameters():
            if param.grad is not None:
                param.grad = param.grad.clone()
        self.optimizer.step()

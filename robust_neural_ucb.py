import torch
import torch.nn as nn
import numpy as np

class RobustNeuralUCB:
    """
    Numerically stable version of NeuralUCB for treatment decisions.
    """
    def __init__(self, context_dim, alpha=1.0, l2=1.0, hidden=(64, 32), lr=1e-3, device='cpu'):
        self.device = device
        self.alpha = alpha
        self.l2 = l2
        self.d = hidden[-1] if hidden else context_dim
        
        # Simple MLP
        layers = []
        prev_dim = context_dim
        for h_dim in hidden:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        
        self.net = nn.Sequential(*layers).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        # Per-arm linear heads with better initialization
        self.A = [torch.eye(self.d, device=device) * l2 for _ in range(2)]
        self.b = [torch.zeros(self.d, 1, device=device) for _ in range(2)]
        
        # Track updates for each arm
        self.arm_counts = [0, 0]
        
    def _safe_inverse(self, A):
        """Compute numerically stable matrix inverse."""
        try:
            # Add small regularization if needed
            if torch.det(A) < 1e-6:
                A = A + torch.eye(A.shape[0], device=self.device) * 1e-3
            return torch.inverse(A)
        except:
            # Fallback to pseudo-inverse
            return torch.pinverse(A)
    
    def select(self, context):
        """Select action using UCB with numerical stability."""
        x = torch.tensor(context, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            phi = self.net(x).view(-1, 1)
            
            ucbs = []
            for a in range(2):
                try:
                    A_inv = self._safe_inverse(self.A[a])
                    theta = A_inv @ self.b[a]
                    
                    # Mean prediction
                    mu = (phi.T @ theta).item()
                    
                    # Confidence bound with numerical stability
                    var = (phi.T @ A_inv @ phi).item()
                    var = max(var, 1e-6)  # Ensure positive variance
                    
                    ucb = mu + self.alpha * np.sqrt(var)
                    ucbs.append(ucb)
                except:
                    # Fallback: use optimistic value
                    ucbs.append(float('inf') if self.arm_counts[a] == 0 else 0.5)
            
            # Add small random noise to break ties
            ucbs[0] += np.random.normal(0, 1e-6)
            ucbs[1] += np.random.normal(0, 1e-6)
            
            selected_action = int(np.argmax(ucbs))
            
        return selected_action, phi
    
    def update(self, context, action, phi, reward):
        """Update with numerical stability checks."""
        self.arm_counts[action] += 1
        
        with torch.no_grad():
            phi_detached = phi.view(-1, 1)
            
            # Update A matrix with stability check
            outer_prod = phi_detached @ phi_detached.T
            if torch.isfinite(outer_prod).all():
                self.A[action] += outer_prod
            
            # Update b vector with stability check
            reward_update = reward * phi_detached
            if torch.isfinite(reward_update).all():
                self.b[action] += reward_update
        
        # Neural network update
        self.optimizer.zero_grad()
        phi_for_grad = self.net(torch.tensor(context, dtype=torch.float32, device=self.device)).view(-1, 1)
        
        # Compute prediction for selected action
        with torch.no_grad():
            A_inv = self._safe_inverse(self.A[action])
            theta = A_inv @ self.b[action]
        
        y_pred = (phi_for_grad.T @ theta).squeeze()
        loss = (y_pred - reward) ** 2
        
        # Gradient clipping for stability
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
    
    def get_stats(self):
        """Get debugging statistics."""
        stats = {}
        for a in range(2):
            try:
                A_inv = self._safe_inverse(self.A[a])
                theta = A_inv @ self.b[a]
                stats[f'arm_{a}_theta_norm'] = torch.norm(theta).item()
                stats[f'arm_{a}_A_det'] = torch.det(self.A[a]).item()
                stats[f'arm_{a}_b_norm'] = torch.norm(self.b[a]).item()
                stats[f'arm_{a}_count'] = self.arm_counts[a]
            except:
                stats[f'arm_{a}_error'] = True
        return stats
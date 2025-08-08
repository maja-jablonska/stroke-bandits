import numpy as np
from scipy.linalg import cholesky, solve_triangular

class GPOracle:
    def __init__(self, kernel='rbf', noise=0.1):
        self.kernel = self._get_kernel(kernel)
        self.noise = noise
        self.X_train = []
        self.y_train = []
        self.L = None  # Cholesky decomposition
        self.alpha = None  # Cached weights
        
    def predict(self, context, action):
        if len(self.X_train) == 0:
            return 0, float('inf')  # High uncertainty initially
            
        features = self._create_features(context, action)
        
        # Compute mean prediction
        K_star = self._compute_kernel(features, self.X_train)
        mean = K_star @ self.alpha
        
        # Compute variance (uncertainty)
        v = solve_triangular(self.L, K_star.T, lower=True)
        var = self.kernel(features, features) - np.sum(v**2)
        
        return mean, np.sqrt(var)
    
    def update(self, context, action, reward):
        features = self._create_features(context, action)
        
        # Add to training set
        self.X_train.append(features)
        self.y_train.append(reward)
        
        # Recompute Cholesky decomposition
        K = self._compute_kernel_matrix(self.X_train)
        K += self.noise * np.eye(len(self.X_train))
        self.L = cholesky(K, lower=True)
        self.alpha = solve_triangular(
            self.L.T, 
            solve_triangular(self.L, self.y_train, lower=True)
        )
        
        # Sparse approximation if dataset gets too large
        if len(self.X_train) > 500:
            self._sparsify()

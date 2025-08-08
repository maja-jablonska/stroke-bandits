import numpy as np

class LinearOracle:
    def __init__(self, dim, lambda_reg=1.0):
        """
        Ridge regression oracle with closed-form updates
        
        Args:
            dim: feature dimension
            lambda_reg: regularization parameter
        """
        self.dim = dim
        self.lambda_reg = lambda_reg
        
        # Sufficient statistics for closed-form solution
        self.A = lambda_reg * np.eye(dim)  # Design matrix (X'X + λI)
        self.b = np.zeros(dim)              # Target vector (X'y)
        
        # Current parameters
        self.theta = np.zeros(dim)
        self.num_samples = 0
        
    def predict(self, context, action=None):
        """
        Predict reward for context-action pair
        """
        features = self._create_features(context, action)
        return np.dot(self.theta, features)
    
    def predict_with_uncertainty(self, context, action=None):
        """
        Predict with confidence bounds (useful for UCB)
        """
        features = self._create_features(context, action)
        
        # Mean prediction
        mean = np.dot(self.theta, features)
        
        # Variance (for confidence bound)
        A_inv = np.linalg.inv(self.A)
        variance = np.dot(features, np.dot(A_inv, features))
        std = np.sqrt(variance)
        
        return mean, std
    
    def update(self, context, action, reward):
        """
        Online update using Sherman-Morrison formula
        """
        features = self._create_features(context, action)
        
        # Update sufficient statistics
        self.A += np.outer(features, features)
        self.b += features * reward
        
        # Update parameters (closed-form solution)
        self.theta = np.linalg.solve(self.A, self.b)
        self.num_samples += 1
    
    def _create_features(self, context, action):
        """
        Create feature vector from context and action
        """
        if action is None:
            return context
        
        # Option 1: Concatenate context and one-hot action
        # Option 2: Kronecker product for interaction terms
        # Option 3: Action-specific linear models
        
        # Here we use concatenation
        action_one_hot = np.zeros(self.n_actions)
        action_one_hot[action] = 1
        return np.concatenate([context, action_one_hot])

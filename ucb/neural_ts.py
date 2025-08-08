import numpy as np
from .neural_ucb import NeuralUCB

class NeuralTS(NeuralUCB):
    """
    Neural Thompson Sampling - Bayesian variant of NeuralUCB
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = 0.1  # Noise standard deviation
        
    def select_action(self, context):
        """
        Thompson Sampling: Sample from posterior and act greedily
        """
        sampled_rewards = {}
        
        features = self._get_features(context, use_target=True)
        
        for action in range(self.n_actions):
            # Sample from posterior N(theta, sigma^2 * A^{-1})
            A_inv = np.linalg.inv(self.A[action])
            
            # Mean and covariance of posterior
            mean = self.theta[action]
            cov = self.sigma**2 * A_inv
            
            # Sample parameters
            sampled_theta = np.random.multivariate_normal(mean, cov)
            
            # Compute predicted reward with sampled parameters
            sampled_rewards[action] = np.dot(sampled_theta, features)
        
        # Select action with highest sampled reward
        best_action = max(sampled_rewards, key=sampled_rewards.get)
        
        return best_action, {'sampled_rewards': sampled_rewards}

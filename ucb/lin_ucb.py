import numpy as np
from numpy.linalg import inv
from numpy import sqrt
from numpy import argmax

class LinUCB:
    """
    Traditional LinUCB with built-in linear model
    """
    def __init__(self, dim, alpha=1.0, lambda_reg=1.0):
        self.dim = dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Initialize parameters for each action
        self.A = {}  # Feature covariance matrices
        self.b = {}  # Feature-reward vectors
        self.theta = {}  # Model parameters
        
    def initialize_action(self, action):
        """Initialize parameters for a new action"""
        if action not in self.A:
            self.A[action] = self.lambda_reg * np.eye(self.dim)
            self.b[action] = np.zeros(self.dim)
            self.theta[action] = np.zeros(self.dim)
    
    def select_action(self, context, possible_actions):
        ucb_scores = {}
        
        for action in possible_actions:
            self.initialize_action(action)
            
            # Compute UCB score
            A_inv = inv(self.A[action])
            theta = A_inv @ self.b[action]
            
            # Mean prediction
            mean = context @ theta
            
            # Confidence bound (exploration bonus)
            variance = context @ A_inv @ context
            std = sqrt(variance)
            
            # UCB = exploitation + exploration
            ucb_scores[action] = mean + self.alpha * std
            
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, context, action, reward):
        """
        This is the key LinUCB update!
        Updates Ridge Regression sufficient statistics
        """
        # Ensure action is initialized
        self.initialize_action(action)

        # Make sure context and reward are 1D and scalar, respectively
        if hasattr(reward, "shape") and reward.shape != ():
            reward = np.squeeze(reward)
        if hasattr(context, "shape") and len(context.shape) > 1:
            context = np.squeeze(context)

        # Update sufficient statistics
        self.A[action] += np.outer(context, context)
        self.b[action] += context * reward

        # Update model parameters (could be lazy/cached)
        A_inv = inv(self.A[action])
        self.theta[action] = A_inv @ self.b[action]
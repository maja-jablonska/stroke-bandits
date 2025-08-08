import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class NeuralUCB:
    """
    Neural Upper Confidence Bound for contextual bandits
    """
    def __init__(
        self,
        context_dim,
        n_actions,
        hidden_dims=[128, 64],
        feature_dim=32,
        lambda_reg=1.0,
        nu=0.1,
        learning_rate=0.001,
        batch_size=32,
        replay_buffer_size=10000,
        ucb_alpha=1.0,
        update_freq=10,
        target_update_freq=100
    ):
        """
        Args:
            context_dim: Dimension of context (patient features)
            n_actions: Number of possible actions (treatments)
            hidden_dims: Hidden layer dimensions
            feature_dim: Dimension of learned features (m)
            lambda_reg: Ridge regression regularization
            nu: Neural network L2 regularization
            learning_rate: Learning rate for neural network
            batch_size: Batch size for neural network training
            replay_buffer_size: Size of experience replay buffer
            ucb_alpha: Exploration parameter
            update_freq: How often to update neural network
            target_update_freq: How often to update target network
        """
        self.context_dim = context_dim
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.ucb_alpha = ucb_alpha
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        
        # Neural network for feature extraction
        self.feature_network = self._build_network(
            context_dim, hidden_dims, feature_dim
        )
        
        # Target network for stable feature extraction
        self.target_network = self._build_network(
            context_dim, hidden_dims, feature_dim
        )
        self.target_network.load_state_dict(self.feature_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.feature_network.parameters(), 
            lr=learning_rate,
            weight_decay=nu
        )
        
        # Linear UCB parameters for each action
        self.A = {}  # Design matrices
        self.b = {}  # Target vectors
        self.theta = {}  # Linear parameters in feature space
        
        for a in range(n_actions):
            self.A[a] = lambda_reg * np.eye(feature_dim)
            self.b[a] = np.zeros(feature_dim)
            self.theta[a] = np.zeros(feature_dim)
        
        # Replay buffer for neural network training
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Counters
        self.t = 0
        self.updates = 0
        
    def _build_network(self, input_dim, hidden_dims, output_dim):
        """
        Build feature extraction network
        """
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Final layer (no activation - raw features)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _get_features(self, context, use_target=False):
        """
        Extract features using neural network
        """
        network = self.target_network if use_target else self.feature_network
        
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context)
            if len(context_tensor.shape) == 1:
                context_tensor = context_tensor.unsqueeze(0)
            features = network(context_tensor)
            return features.squeeze().numpy()
    
    def compute_ucb(self, context, action):
        """
        Compute UCB score for a context-action pair
        """
        # Get neural network features
        features = self._get_features(context, use_target=True)
        
        # Linear prediction in feature space
        mean_reward = np.dot(self.theta[action], features)
        
        # Compute confidence bound
        A_inv = np.linalg.inv(self.A[action])
        variance = np.dot(features, np.dot(A_inv, features))
        confidence_bound = self.ucb_alpha * np.sqrt(variance)
        
        ucb_score = mean_reward + confidence_bound
        
        return ucb_score, mean_reward, confidence_bound
    
    def select_action(self, context):
        """
        Select action using UCB strategy
        """
        ucb_scores = {}
        predictions = {}
        uncertainties = {}
        
        for action in range(self.n_actions):
            ucb, mean, conf = self.compute_ucb(context, action)
            ucb_scores[action] = ucb
            predictions[action] = mean
            uncertainties[action] = conf
        
        # Select action with highest UCB
        best_action = max(ucb_scores, key=ucb_scores.get)
        
        return best_action, {
            'ucb_scores': ucb_scores,
            'predictions': predictions,
            'uncertainties': uncertainties
        }
    
    def update(self, context, action, reward):
        """
        Update both neural network and linear models
        """
        self.t += 1
        
        # Add to replay buffer
        self.replay_buffer.append((context, action, reward))
        
        # Update linear UCB parameters with current features
        features = self._get_features(context, use_target=True)
        
        # Update sufficient statistics for the selected action
        self.A[action] += np.outer(features, features)
        self.b[action] += features * reward
        
        # Solve for new linear parameters
        A_inv = np.linalg.inv(self.A[action])
        self.theta[action] = np.dot(A_inv, self.b[action])
        
        # Update neural network periodically
        if self.t % self.update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
            self._update_neural_network()
        
        # Update target network periodically
        if self.t % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.feature_network.state_dict())
    
    def _update_neural_network(self):
        """
        Train neural network on replay buffer
        """
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        contexts, actions, rewards = zip(*batch)
        
        contexts = torch.FloatTensor(contexts)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        # Forward pass
        features = self.feature_network(contexts)
        
        # Compute predictions using current linear parameters
        predictions = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            action = actions[i].item()
            feature = features[i].detach().numpy()
            predictions[i] = np.dot(self.theta[action], feature)
        
        predictions = predictions.detach()
        
        # Compute loss (supervised learning on rewards)
        # We want features that make linear prediction accurate
        final_layer_predictions = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            action = actions[i].item()
            theta_tensor = torch.FloatTensor(self.theta[action])
            final_layer_predictions[i] = torch.dot(features[i], theta_tensor)
        
        loss = F.mse_loss(final_layer_predictions, rewards)
        
        # Add gradient penalty for stable training
        gradient_penalty = 0
        for param in self.feature_network.parameters():
            if param.grad is not None:
                gradient_penalty += torch.sum(param.grad ** 2)
        
        total_loss = loss + 0.01 * gradient_penalty
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.feature_network.parameters(), 1.0)
        
        self.optimizer.step()
        self.updates += 1
        
        return loss.item()

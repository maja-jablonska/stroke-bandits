import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class NeuralOracle:
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.1):
        self.network = self._build_network(input_dim, hidden_dims)
        self.replay_buffer = deque(maxlen=10000)
        self.update_frequency = 10
        self.steps = 0
        self.dropout_rate = dropout_rate
        
    def _build_network(self, input_dim, hidden_dims):
        model = nn.Sequential()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            model.add(nn.Linear(prev_dim, hidden_dim))
            model.add(nn.ReLU())
            model.add(nn.Dropout(self.dropout_rate))  # For uncertainty
            prev_dim = hidden_dim
            
        model.add(nn.Linear(prev_dim, 1))  # Reward prediction
        return model
    
    def predict(self, context, action, n_samples=10):
        features = self._create_features(context, action)
        
        # Monte Carlo dropout for uncertainty
        self.network.eval()
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.network(features)
                predictions.append(pred.item())
        
        return np.mean(predictions), np.std(predictions)
    
    def update(self, context, action, reward):
        self.replay_buffer.append((context, action, reward))
        self.steps += 1
        
        if self.steps % self.update_frequency == 0:
            self._train_on_batch()
    
    def _train_on_batch(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
            
        batch = random.sample(self.replay_buffer, batch_size)
        X, y = self._prepare_batch(batch)
        
        self.network.train()
        optimizer = torch.optim.Adam(self.network.parameters())
        loss_fn = nn.MSELoss()
        
        # Multiple epochs on the batch
        for _ in range(10):
            pred = self.network(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

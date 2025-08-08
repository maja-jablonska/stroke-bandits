import numpy as np
from sklearn.tree import DecisionTreeRegressor
from collections import deque
import random

class OnlineRandomForestOracle:
    def __init__(self, n_trees=100, max_depth=10):
        self.trees = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.buffer = deque(maxlen=1000)  # Experience replay
        
    def predict(self, context, action):
        if not self.trees:
            return 0  # Default prediction
        
        features = self._create_features(context, action)
        predictions = [tree.predict(features) for tree in self.trees]
        
        # Return mean and uncertainty
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        return mean_pred, std_pred
    
    def update(self, context, action, reward):
        self.buffer.append((context, action, reward))
        
        # Periodically retrain trees on buffer
        if len(self.buffer) % 100 == 0:
            self._retrain_trees()
    
    def _retrain_trees(self):
        # Sample bootstrap datasets and train trees
        for i in range(self.n_trees):
            bootstrap_sample = random.choices(self.buffer, k=len(self.buffer))
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            X, y = self._prepare_data(bootstrap_sample)
            tree.fit(X, y)
            
            if len(self.trees) < self.n_trees:
                self.trees.append(tree)
            else:
                # Replace oldest or worst-performing tree
                self.trees[i % self.n_trees] = tree

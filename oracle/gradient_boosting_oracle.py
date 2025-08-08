from sklearn.tree import DecisionTreeRegressor
from collections import deque
import numpy as np

class OnlineGradientBoostingOracle:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.trees = []
        self.learning_rate = learning_rate
        self.residuals_buffer = deque(maxlen=1000)
        self.n_estimators = n_estimators
        self.n_trained = 0  # Track how many trees have been trained
        
    def predict(self, context, action):
        if not self.trees:
            return 0
        
        features = self._create_features(context, action)
        # Each tree predicts a shape (1,) array, so flatten
        prediction = 0
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(features.reshape(1, -1))[0]
        return prediction
    
    def update(self, context, action, reward):
        if self.n_trained >= self.n_estimators:
            return  # Do not train more than n_estimators trees

        features = self._create_features(context, action)
        
        # Compute residual
        current_pred = self.predict(context, action)
        residual = reward - current_pred
        
        self.residuals_buffer.append((features, residual))
        
        # Add new tree fitted to residuals
        if len(self.residuals_buffer) >= 50:  # Minimum samples
            tree = DecisionTreeRegressor(max_depth=3)
            X, residuals = zip(*list(self.residuals_buffer))
            tree.fit(np.array(X), np.array(residuals))
            self.trees.append(tree)
            self.n_trained += 1

    def _create_features(self, context, action):
        # Create features for the oracle
        return np.concatenate([context, [action]])

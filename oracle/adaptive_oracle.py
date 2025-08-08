from .linear_oracle import LinearOracle
from .online_rf_oracle import OnlineRandomForestOracle
from .neural_oracle import NeuralOracle
from .gp_oracle import GPOracle
from .gradient_boosting_oracle import OnlineGradientBoostingOracle
import numpy as np

class AdaptiveOracle:
    """
    Combines multiple oracles and learns which performs best
    """
    def __init__(self):
        self.oracles = {
            'linear': LinearOracle(),
            'rf': OnlineRandomForestOracle(),
            'gradient_boosting': OnlineGradientBoostingOracle(),
            'neural': NeuralOracle(),
            'gp': GPOracle()
        }
        self.weights = {name: 1.0 for name in self.oracles}
        self.performance = {name: [] for name in self.oracles}
        
    def predict(self, context, action):
        predictions = {}
        uncertainties = {}
        
        for name, oracle in self.oracles.items():
            pred = oracle.predict(context, action)
            if isinstance(pred, tuple):
                predictions[name], uncertainties[name] = pred
            else:
                predictions[name] = pred
                uncertainties[name] = 0
        
        # Weighted ensemble
        weights = np.array([self.weights[name] for name in self.oracles])
        weights = weights / weights.sum()
        
        final_pred = sum(w * predictions[name] 
                        for w, name in zip(weights, self.oracles))
        
        return final_pred, predictions, uncertainties
    
    def update(self, context, action, reward):
        # Update all oracles
        for name, oracle in self.oracles.items():
            # Get prediction before update
            pred = oracle.predict(context, action)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # Track performance
            error = abs(pred - reward)
            self.performance[name].append(error)
            
            # Update oracle
            oracle.update(context, action, reward)
        
        # Update weights based on recent performance
        if all(len(perf) > 10 for perf in self.performance.values()):
            for name in self.oracles:
                recent_error = np.mean(self.performance[name][-10:])
                # Inverse error weighting
                self.weights[name] = 1.0 / (recent_error + 1e-6)

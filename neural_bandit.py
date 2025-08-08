import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from robust_neural_ucb import RobustNeuralUCB


class NeuralTreatmentBandit:
    """
    Neural contextual bandit for stroke treatment decisions using NeuralUCB.
    
    Actions:
    - 0: Placebo
    - 1: Treatment (rt-PA)
    
    The bandit uses NeuralUCB for exploration/exploitation and XGBoost models
    as reward predictors to provide feedback on selected actions.
    """
    
    def __init__(self, data_path="action_reward_context_combined_processed.csv", 
                 alpha=1.0, lr=1e-3, hidden=(64, 32)):
        """
        Initialize the neural bandit.
        
        Args:
            data_path: Path to the processed CSV data
            alpha: UCB exploration parameter
            lr: Learning rate for neural network
            hidden: Hidden layer sizes
        """
        self.data_path = data_path
        self.placebo_model = None
        self.treatment_model = None
        self.feature_columns = None
        self.data = None
        
        self._load_models()
        self._load_data()
        
        # Initialize RobustNeuralUCB
        context_dim = len([col for col in self.feature_columns if col != 'treatment'])
        self.neural_ucb = RobustNeuralUCB(
            context_dim=context_dim,
            alpha=alpha,
            lr=lr,
            hidden=hidden,
            device='cpu'
        )
        
        # Track action selections for debugging
        self.action_history = []
        
        print(f"✓ Initialized RobustNeuralUCB with context dim: {context_dim}, alpha: {alpha}")
        
    def _load_models(self):
        """Load the pre-trained XGBoost models."""
        base_path = os.path.dirname(__file__)
        
        placebo_model_path = os.path.join(base_path, "rewarder", "placebo_ohs6_predictor.pkl")
        treatment_model_path = os.path.join(base_path, "rewarder", "treat_ohs6_predictor.pkl")
        
        try:
            self.placebo_model = joblib.load(placebo_model_path)
            self.treatment_model = joblib.load(treatment_model_path)
            print(f"✓ Loaded reward models from rewarder/")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load models: {e}")
    
    def _load_data(self):
        """Load and prepare the data."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Loaded data with shape {self.data.shape}")
            
            # Get feature columns (exclude only target column)
            exclude_cols = ['ohs6']
            self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
            print(f"✓ Using {len(self.feature_columns)} features total")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load data: {e}")
    
    def get_reward(self, context, action):
        """
        Get reward prediction using XGBoost models.
        
        Args:
            context: Patient features (without treatment)
            action: Selected action (0=placebo, 1=treatment)
            
        Returns:
            float: Predicted OHS6 reward
        """
        # Create full feature vector with treatment
        full_context = context.copy()
        full_context['treatment'] = action
        features = full_context[self.feature_columns].values.reshape(1, -1)
        
        if action == 0:  # Placebo
            reward = self.placebo_model.predict(features)[0]
        else:  # Treatment
            reward = self.treatment_model.predict(features)[0]
            
        return reward
    
    def select_action(self, context):
        """
        Select action using NeuralUCB and get reward feedback.
        
        Args:
            context: Patient features (pandas Series, without treatment)
            
        Returns:
            dict: Action selection results
        """
        if isinstance(context, dict):
            context = pd.Series(context)
            
        # Get context features (exclude treatment)
        context_features = [col for col in self.feature_columns if col != 'treatment']
        context_array = context[context_features].values.astype(np.float32)
        
        # Select action using NeuralUCB
        selected_action, phi = self.neural_ucb.select(context_array)
        
        # Track for debugging
        self.action_history.append(selected_action)
        
        # Get reward prediction for selected action
        predicted_reward = self.get_reward(context, selected_action)
        
        # Normalize reward to [0,1] range for numerical stability
        # OHS6 scores are already in [0,1] but let's clip just in case
        predicted_reward = np.clip(predicted_reward, 0.0, 1.0)
        
        # Update NeuralUCB with the predicted reward
        self.neural_ucb.update(context_array, selected_action, phi, predicted_reward)
        
        # Also get prediction for the other action for comparison
        other_action = 1 - selected_action
        other_reward = self.get_reward(context, other_action)
        
        return {
            'action': selected_action,
            'selected_reward_prediction': predicted_reward,
            'other_reward_prediction': other_reward,
            'confidence': abs(predicted_reward - other_reward),
            'recommendation': 'Treatment' if selected_action == 1 else 'Placebo'
        }
    
    def train_online(self, n_samples=None, verbose=True):
        """
        Train the bandit online using the dataset.
        
        Args:
            n_samples: Number of samples to train on (default: all data)
            verbose: Print progress
            
        Returns:
            dict: Training results
        """
        if n_samples:
            train_data = self.data.sample(n_samples, random_state=42).reset_index(drop=True)
        else:
            train_data = self.data.reset_index(drop=True)
        
        total_samples = len(train_data)
        actions_taken = []
        rewards_received = []
        
        if verbose:
            print(f"Training NeuralUCB online on {total_samples} samples...")
        
        for idx, row in train_data.iterrows():
            # Get context without treatment
            context_features = [col for col in self.feature_columns if col != 'treatment']
            context = row[context_features]
            
            # Select action and get reward
            decision = self.select_action(context)
            
            actions_taken.append(decision['action'])
            rewards_received.append(decision['selected_reward_prediction'])
            
            if verbose and (idx + 1) % 100 == 0:
                recent_reward = np.mean(rewards_received[-100:])
                print(f"  Step {idx + 1}/{total_samples} - Recent avg reward: {recent_reward:.3f}")
        
        return {
            'total_samples': total_samples,
            'actions_taken': actions_taken,
            'rewards_received': rewards_received,
            'final_avg_reward': np.mean(rewards_received),
            'placebo_rate': (np.array(actions_taken) == 0).mean(),
            'treatment_rate': (np.array(actions_taken) == 1).mean()
        }
    
    def evaluate_policy(self, n_samples=None, use_trained=True):
        """
        Evaluate the bandit policy.
        
        Args:
            n_samples: Number of samples to evaluate
            use_trained: Use the current trained state (vs fresh bandit)
            
        Returns:
            dict: Evaluation metrics
        """
        if n_samples:
            eval_data = self.data.sample(n_samples, random_state=123).reset_index(drop=True)
        else:
            eval_data = self.data.reset_index(drop=True)
        
        # Create fresh bandit for comparison if needed
        if not use_trained:
            context_dim = len([col for col in self.feature_columns if col != 'treatment'])
            fresh_neural_ucb = RobustNeuralUCB(
                context_dim=context_dim,
                alpha=self.neural_ucb.alpha,
                lr=1e-3,
                hidden=(64, 32),
                device='cpu'
            )
            # Temporarily swap
            original_ucb = self.neural_ucb
            self.neural_ucb = fresh_neural_ucb
        
        bandit_rewards = []
        actual_rewards = []
        bandit_actions = []
        actual_actions = []
        
        for idx, row in eval_data.iterrows():
            context_features = [col for col in self.feature_columns if col != 'treatment']
            context = row[context_features]
            actual_treatment = row['treatment']
            actual_reward = row['ohs6']
            
            # Get bandit's decision
            decision = self.select_action(context)
            recommended_action = decision['action']
            
            bandit_actions.append(recommended_action)
            actual_actions.append(actual_treatment)
            bandit_rewards.append(decision['selected_reward_prediction'])
            actual_rewards.append(actual_reward)
        
        # Restore original if needed
        if not use_trained:
            self.neural_ucb = original_ucb
        
        agreement_rate = (np.array(bandit_actions) == np.array(actual_actions)).mean()
        
        return {
            'agreement_rate': agreement_rate,
            'samples_evaluated': len(eval_data),
            'bandit_mean_reward': np.mean(bandit_rewards),
            'actual_mean_reward': np.mean(actual_rewards),
            'improvement': np.mean(bandit_rewards) - np.mean(actual_rewards),
            'bandit_rewards': bandit_rewards,
            'actual_rewards': actual_rewards,
            'bandit_actions': bandit_actions,
            'actual_actions': actual_actions
        }
    
    def plot_learning_curve(self, training_results, figsize=(12, 6)):
        """Plot learning curve from training results."""
        rewards = training_results['rewards_received']
        actions = training_results['actions_taken']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Cumulative reward
        cumulative_rewards = np.cumsum(rewards)
        ax1.plot(cumulative_rewards, color='blue', linewidth=2)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('NeuralUCB Learning: Cumulative Reward')
        ax1.grid(True, alpha=0.3)
        
        # Action selection over time (smoothed)
        window_size = max(50, len(actions) // 20)
        action_rate = []
        for i in range(window_size, len(actions)):
            recent_actions = actions[i-window_size:i]
            treatment_rate = (np.array(recent_actions) == 1).mean()
            action_rate.append(treatment_rate)
        
        ax2.plot(range(window_size, len(actions)), action_rate, color='red', linewidth=2)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Treatment Selection Rate')
        ax2.set_title(f'Action Selection Evolution (window={window_size})')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cumulative_reward(self, n_samples=None, figsize=(12, 6)):
        """Plot cumulative reward comparison."""
        results = self.evaluate_policy(n_samples)
        
        bandit_rewards = results['bandit_rewards']
        actual_rewards = results['actual_rewards']
        
        bandit_cumulative = np.cumsum(bandit_rewards)
        actual_cumulative = np.cumsum(actual_rewards)
        
        plt.figure(figsize=figsize)
        plt.plot(bandit_cumulative, label=f'NeuralUCB Policy (Mean: {results["bandit_mean_reward"]:.3f})', 
                linewidth=2, color='blue')
        plt.plot(actual_cumulative, label=f'Historical Decisions (Mean: {results["actual_mean_reward"]:.3f})', 
                linewidth=2, color='red', alpha=0.7)
        
        improvement = results['improvement']
        final_diff = bandit_cumulative[-1] - actual_cumulative[-1]
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.xlabel('Patient Number')
        plt.ylabel('Cumulative OHS6 Score')
        plt.title(f'Cumulative Reward: NeuralUCB vs Historical Decisions\n'
                 f'Agreement Rate: {results["agreement_rate"]:.1%} | '
                 f'Improvement: {improvement:+.3f} | '
                 f'Final Difference: {final_diff:+.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics box
        textstr = f'''Key Metrics:
        Samples: {results["samples_evaluated"]:,}
        NeuralUCB Mean: {results["bandit_mean_reward"]:.3f}
        Historical Mean: {results["actual_mean_reward"]:.3f}
        Improvement: {improvement:+.3f}
        Agreement: {results["agreement_rate"]:.1%}'''
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.5, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_action_distribution(self, n_samples=None, figsize=(10, 6)):
        """Plot action selection frequency comparison."""
        results = self.evaluate_policy(n_samples)
        
        bandit_counts = np.bincount(results['bandit_actions'], minlength=2)
        actual_counts = np.bincount(results['actual_actions'], minlength=2)
        
        total_samples = results['samples_evaluated']
        bandit_pct = bandit_counts / total_samples * 100
        actual_pct = actual_counts / total_samples * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar chart
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, actual_pct, width, label='Historical Decisions', 
                       color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, bandit_pct, width, label='NeuralUCB Policy', 
                       color='blue', alpha=0.7)
        
        ax1.set_xlabel('Treatment')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Action Selection Frequency Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Placebo', 'Treatment'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
                        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        # Pie chart for bandit
        labels = ['Placebo', 'Treatment']
        colors = ['lightcoral', 'lightblue']
        ax2.pie([bandit_pct[0], bandit_pct[1]], labels=labels, colors=colors, 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'NeuralUCB Policy\n(n={total_samples})')
        
        # Add summary
        diff_placebo = bandit_pct[0] - actual_pct[0]
        diff_treatment = bandit_pct[1] - actual_pct[1]
        
        summary_text = f"""Action Selection Summary:
        Historical: {actual_pct[0]:.1f}% Placebo, {actual_pct[1]:.1f}% Treatment
        NeuralUCB: {bandit_pct[0]:.1f}% Placebo, {bandit_pct[1]:.1f}% Treatment
        
        Changes:
        Placebo: {diff_placebo:+.1f} percentage points
        Treatment: {diff_treatment:+.1f} percentage points"""
        
        fig.text(0.5, 0.02, summary_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def get_sample_patient(self, idx=None):
        """Get a sample patient for testing."""
        if idx is None:
            idx = np.random.randint(0, len(self.data))
        
        row = self.data.iloc[idx]
        context_features = [col for col in self.feature_columns if col != 'treatment']
        context = row[context_features]
        
        return {
            'patient_id': idx,
            'context': context,
            'actual_treatment': row['treatment'],
            'actual_ohs6': row['ohs6']
        }


if __name__ == "__main__":
    # Example usage
    bandit = NeuralTreatmentBandit(alpha=1.0, lr=1e-3)
    
    # Train the bandit online
    print(f"\n--- Training NeuralUCB ---")
    training_results = bandit.train_online(n_samples=100, verbose=True)
    print(f"Training complete - Final avg reward: {training_results['final_avg_reward']:.3f}")
    print(f"Action distribution - Placebo: {training_results['placebo_rate']:.1%}, Treatment: {training_results['treatment_rate']:.1%}")
    
    # Test on a sample patient
    patient = bandit.get_sample_patient()
    print(f"\n--- Sample Patient {patient['patient_id']} ---")
    print(f"Actual treatment: {patient['actual_treatment']} ({'Treatment' if patient['actual_treatment'] == 1 else 'Placebo'})")
    print(f"Actual OHS6 score: {patient['actual_ohs6']:.3f}")
    
    recommendation = bandit.select_action(patient['context'])
    print(f"\n--- NeuralUCB Recommendation ---")
    print(f"Recommended action: {recommendation['action']} ({recommendation['recommendation']})")
    print(f"Selected prediction: {recommendation['selected_reward_prediction']:.3f}")
    print(f"Other prediction: {recommendation['other_reward_prediction']:.3f}")
    print(f"Confidence: {recommendation['confidence']:.3f}")
    
    # Evaluate policy
    print(f"\n--- Policy Evaluation ---")
    metrics = bandit.evaluate_policy(n_samples=100)
    print(f"Agreement with actual decisions: {metrics['agreement_rate']:.1%}")
    print(f"NeuralUCB mean reward: {metrics['bandit_mean_reward']:.3f}")
    print(f"Actual mean reward: {metrics['actual_mean_reward']:.3f}")
    print(f"Improvement: {metrics['improvement']:.3f}")
    
    # Create visualizations
    print(f"\n--- Creating Visualizations ---")
    os.makedirs('figures', exist_ok=True)
    
    # Learning curve
    fig1 = bandit.plot_learning_curve(training_results)
    plt.savefig('figures/neural_ucb_learning_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved learning curve as 'figures/neural_ucb_learning_curve.png'")
    
    # Cumulative reward
    fig2 = bandit.plot_cumulative_reward(n_samples=100)
    plt.savefig('figures/neural_ucb_cumulative_reward.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cumulative reward as 'figures/neural_ucb_cumulative_reward.png'")
    
    # Action distribution
    fig3 = bandit.plot_action_distribution(n_samples=100)
    plt.savefig('figures/neural_ucb_action_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved action distribution as 'figures/neural_ucb_action_distribution.png'")
    
    plt.show()
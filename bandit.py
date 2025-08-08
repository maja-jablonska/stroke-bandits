import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


class TreatmentBandit:
    """
    Contextual bandit for stroke treatment decisions using pre-trained XGBoost models.
    
    Actions:
    - 0: Placebo
    - 1: Treatment (rt-PA)
    
    The bandit uses separate reward models for each treatment to predict OHS6 scores,
    then selects the treatment with the better predicted outcome.
    """
    
    def __init__(self, data_path="action_reward_context_combined_processed.csv"):
        """
        Initialize the bandit with trained models and data.
        
        Args:
            data_path: Path to the processed CSV data
        """
        self.data_path = data_path
        self.placebo_model = None
        self.treatment_model = None
        self.feature_columns = None
        self.data = None
        
        self._load_models()
        self._load_data()
        
    def _load_models(self):
        """Load the pre-trained XGBoost models."""
        base_path = os.path.dirname(__file__)
        
        placebo_model_path = os.path.join(base_path, "rewarder", "placebo_ohs6_predictor.pkl")
        treatment_model_path = os.path.join(base_path, "rewarder", "treat_ohs6_predictor.pkl")
        
        try:
            self.placebo_model = joblib.load(placebo_model_path)
            self.treatment_model = joblib.load(treatment_model_path)
            print(f"✓ Loaded models from {placebo_model_path} and {treatment_model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load models: {e}")
    
    def _load_data(self):
        """Load and prepare the data."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Loaded data with shape {self.data.shape}")
            
            # Get feature columns (exclude only target column - models need treatment column)
            exclude_cols = ['ohs6']
            self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
            print(f"✓ Using {len(self.feature_columns)} features")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load data: {e}")
    
    def predict_rewards(self, context):
        """
        Predict OHS6 scores for both treatments given patient context.
        
        Args:
            context: Patient features as pandas Series or dict (without treatment column)
            
        Returns:
            tuple: (placebo_prediction, treatment_prediction)
        """
        if isinstance(context, dict):
            context = pd.Series(context)
        
        # Create features for placebo (treatment=0)
        placebo_context = context.copy()
        placebo_context['treatment'] = 0
        placebo_features = placebo_context[self.feature_columns].values.reshape(1, -1)
        
        # Create features for treatment (treatment=1)  
        treatment_context = context.copy()
        treatment_context['treatment'] = 1
        treatment_features = treatment_context[self.feature_columns].values.reshape(1, -1)
        
        placebo_pred = self.placebo_model.predict(placebo_features)[0]
        treatment_pred = self.treatment_model.predict(treatment_features)[0]
        
        return placebo_pred, treatment_pred
    
    def select_action(self, context):
        """
        Select the best treatment based on predicted OHS6 scores.
        Higher OHS6 scores are better outcomes.
        
        Args:
            context: Patient features
            
        Returns:
            dict: Contains selected action, predictions, and confidence
        """
        placebo_pred, treatment_pred = self.predict_rewards(context)
        
        # Select treatment with higher predicted OHS6 score
        selected_action = 1 if treatment_pred > placebo_pred else 0
        confidence = abs(treatment_pred - placebo_pred)
        
        return {
            'action': selected_action,
            'placebo_prediction': placebo_pred,
            'treatment_prediction': treatment_pred,
            'confidence': confidence,
            'recommendation': 'Treatment' if selected_action == 1 else 'Placebo'
        }
    
    def evaluate_policy(self, n_samples=None):
        """
        Evaluate the bandit policy on historical data.
        
        Args:
            n_samples: Number of samples to evaluate (default: all data)
            
        Returns:
            dict: Evaluation metrics
        """
        if n_samples:
            eval_data = self.data.sample(n_samples, random_state=42)
        else:
            eval_data = self.data
        
        correct_decisions = 0
        total_samples = len(eval_data)
        
        bandit_rewards = []
        actual_rewards = []
        bandit_actions = []
        actual_actions = []
        
        for idx, row in eval_data.iterrows():
            # Get context without treatment column
            context_features = [col for col in self.feature_columns if col != 'treatment']
            context = row[context_features]
            actual_treatment = row['treatment']
            actual_reward = row['ohs6']
            
            # Get bandit's decision
            decision = self.select_action(context)
            recommended_action = decision['action']
            
            # Track actions
            bandit_actions.append(recommended_action)
            actual_actions.append(actual_treatment)
            
            # Check if bandit agrees with actual treatment
            if recommended_action == actual_treatment:
                correct_decisions += 1
                bandit_rewards.append(actual_reward)
            else:
                # Estimate what the reward would have been with bandit's choice
                if recommended_action == 0:
                    estimated_reward = decision['placebo_prediction']
                else:
                    estimated_reward = decision['treatment_prediction']
                bandit_rewards.append(estimated_reward)
            
            actual_rewards.append(actual_reward)
        
        agreement_rate = correct_decisions / total_samples
        
        return {
            'agreement_rate': agreement_rate,
            'samples_evaluated': total_samples,
            'bandit_mean_reward': np.mean(bandit_rewards),
            'actual_mean_reward': np.mean(actual_rewards),
            'improvement': np.mean(bandit_rewards) - np.mean(actual_rewards),
            'bandit_rewards': bandit_rewards,
            'actual_rewards': actual_rewards,
            'bandit_actions': bandit_actions,
            'actual_actions': actual_actions
        }
    
    def get_sample_patient(self, idx=None):
        """Get a sample patient for testing."""
        if idx is None:
            idx = np.random.randint(0, len(self.data))
        
        row = self.data.iloc[idx]
        # Get context without treatment column for prediction
        context_features = [col for col in self.feature_columns if col != 'treatment']
        context = row[context_features]
        
        return {
            'patient_id': idx,
            'context': context,
            'actual_treatment': row['treatment'],
            'actual_ohs6': row['ohs6']
        }
    
    def plot_cumulative_reward(self, n_samples=None, figsize=(12, 6)):
        """
        Plot cumulative reward comparison between bandit and actual decisions.
        
        Args:
            n_samples: Number of samples to evaluate (default: all data)
            figsize: Figure size tuple
        """
        # Get evaluation results with individual rewards
        results = self.evaluate_policy(n_samples)
        
        bandit_rewards = results['bandit_rewards']
        actual_rewards = results['actual_rewards']
        
        # Calculate cumulative rewards
        bandit_cumulative = np.cumsum(bandit_rewards)
        actual_cumulative = np.cumsum(actual_rewards)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Plot cumulative rewards
        plt.plot(bandit_cumulative, label=f'Bandit Policy (Mean: {results["bandit_mean_reward"]:.3f})', 
                linewidth=2, color='blue')
        plt.plot(actual_cumulative, label=f'Historical Decisions (Mean: {results["actual_mean_reward"]:.3f})', 
                linewidth=2, color='red', alpha=0.7)
        
        # Add improvement annotation
        improvement = results['improvement']
        final_diff = bandit_cumulative[-1] - actual_cumulative[-1]
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Styling
        plt.xlabel('Patient Number')
        plt.ylabel('Cumulative OHS6 Score')
        plt.title(f'Cumulative Reward: Contextual Bandit vs Historical Decisions\n'
                 f'Agreement Rate: {results["agreement_rate"]:.1%} | '
                 f'Improvement: {improvement:+.3f} | '
                 f'Final Difference: {final_diff:+.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with key metrics
        textstr = f'''Key Metrics:
        Samples: {results["samples_evaluated"]:,}
        Bandit Mean: {results["bandit_mean_reward"]:.3f}
        Historical Mean: {results["actual_mean_reward"]:.3f}
        Improvement: {improvement:+.3f}
        Agreement: {results["agreement_rate"]:.1%}'''
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.5, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_reward_distribution(self, n_samples=None, figsize=(10, 6)):
        """
        Plot reward distribution comparison.
        
        Args:
            n_samples: Number of samples to evaluate
            figsize: Figure size tuple
        """
        results = self.evaluate_policy(n_samples)
        
        plt.figure(figsize=figsize)
        
        # Create histograms
        plt.hist(results['actual_rewards'], bins=30, alpha=0.7, label='Historical Decisions', 
                color='red', density=True)
        plt.hist(results['bandit_rewards'], bins=30, alpha=0.7, label='Bandit Policy', 
                color='blue', density=True)
        
        # Add mean lines
        plt.axvline(results['actual_mean_reward'], color='red', linestyle='--', 
                   label=f'Historical Mean: {results["actual_mean_reward"]:.3f}')
        plt.axvline(results['bandit_mean_reward'], color='blue', linestyle='--',
                   label=f'Bandit Mean: {results["bandit_mean_reward"]:.3f}')
        
        plt.xlabel('OHS6 Score')
        plt.ylabel('Density')
        plt.title('Reward Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_action_distribution(self, n_samples=None, figsize=(10, 6)):
        """
        Plot action selection frequency comparison.
        
        Args:
            n_samples: Number of samples to evaluate
            figsize: Figure size tuple
        """
        results = self.evaluate_policy(n_samples)
        
        # Count actions
        bandit_counts = np.bincount(results['bandit_actions'], minlength=2)
        actual_counts = np.bincount(results['actual_actions'], minlength=2)
        
        # Calculate percentages
        total_samples = results['samples_evaluated']
        bandit_pct = bandit_counts / total_samples * 100
        actual_pct = actual_counts / total_samples * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar chart comparison
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, actual_pct, width, label='Historical Decisions', 
                       color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, bandit_pct, width, label='Bandit Policy', 
                       color='blue', alpha=0.7)
        
        ax1.set_xlabel('Treatment')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Action Selection Frequency')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Placebo', 'Treatment'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Pie charts
        labels = ['Placebo', 'Treatment']
        colors = ['lightcoral', 'lightblue']
        
        # Historical pie chart
        ax2.pie([actual_pct[0], actual_pct[1]], labels=labels, colors=colors, 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Historical Decisions\n(n={total_samples})')
        
        # Create second subplot for bandit pie chart
        plt.figure(figsize=figsize)
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize)
        
        # Copy the bar chart
        bars3 = ax3.bar(x - width/2, actual_pct, width, label='Historical Decisions', 
                       color='red', alpha=0.7)
        bars4 = ax3.bar(x + width/2, bandit_pct, width, label='Bandit Policy', 
                       color='blue', alpha=0.7)
        
        ax3.set_xlabel('Treatment')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Action Selection Frequency Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Placebo', 'Treatment'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
        for bar in bars4:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Bandit pie chart
        ax4.pie([bandit_pct[0], bandit_pct[1]], labels=labels, colors=colors, 
               autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Bandit Policy\n(n={total_samples})')
        
        # Add summary text
        diff_placebo = bandit_pct[0] - actual_pct[0]
        diff_treatment = bandit_pct[1] - actual_pct[1]
        
        summary_text = f"""Action Selection Summary:
        Historical: {actual_pct[0]:.1f}% Placebo, {actual_pct[1]:.1f}% Treatment
        Bandit: {bandit_pct[0]:.1f}% Placebo, {bandit_pct[1]:.1f}% Treatment
        
        Changes:
        Placebo: {diff_placebo:+.1f} percentage points
        Treatment: {diff_treatment:+.1f} percentage points"""
        
        fig2.text(0.5, 0.02, summary_text, fontsize=9, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        return fig2


if __name__ == "__main__":
    # Example usage
    bandit = TreatmentBandit()
    
    # Get a sample patient
    patient = bandit.get_sample_patient()
    print(f"\n--- Sample Patient {patient['patient_id']} ---")
    print(f"Actual treatment: {patient['actual_treatment']} ({'Treatment' if patient['actual_treatment'] == 1 else 'Placebo'})")
    print(f"Actual OHS6 score: {patient['actual_ohs6']:.3f}")
    
    # Get bandit recommendation
    recommendation = bandit.select_action(patient['context'])
    print(f"\n--- Bandit Recommendation ---")
    print(f"Recommended action: {recommendation['action']} ({recommendation['recommendation']})")
    print(f"Placebo prediction: {recommendation['placebo_prediction']:.3f}")
    print(f"Treatment prediction: {recommendation['treatment_prediction']:.3f}")
    print(f"Confidence: {recommendation['confidence']:.3f}")
    
    # Evaluate policy
    print(f"\n--- Policy Evaluation ---")
    metrics = bandit.evaluate_policy(n_samples=1000)
    print(f"Agreement with actual decisions: {metrics['agreement_rate']:.1%}")
    print(f"Bandit mean reward: {metrics['bandit_mean_reward']:.3f}")
    print(f"Actual mean reward: {metrics['actual_mean_reward']:.3f}")
    print(f"Improvement: {metrics['improvement']:.3f}")
    
    # Create plots
    print(f"\n--- Creating Visualizations ---")
    
    # Ensure figures directory exists
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Plot cumulative reward
    fig1 = bandit.plot_cumulative_reward(n_samples=1000)
    plt.savefig('figures/cumulative_reward_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cumulative reward plot as 'figures/cumulative_reward_comparison.png'")
    
    # Plot reward distribution
    fig2 = bandit.plot_reward_distribution(n_samples=1000) 
    plt.savefig('figures/reward_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved reward distribution plot as 'figures/reward_distribution_comparison.png'")
    
    # Plot action distribution
    fig3 = bandit.plot_action_distribution(n_samples=1000)
    plt.savefig('figures/action_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved action distribution plot as 'figures/action_distribution_comparison.png'")
    
    plt.show()
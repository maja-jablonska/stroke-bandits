import numpy as np

def evaluate_policy(policy, data, reward_model):
    """
    Off-policy evaluation using:
    1. Direct Method (DM): Use reward model predictions
    2. Inverse Propensity Scoring (IPS): Weight by behavior policy
    3. Doubly Robust (DR): Combine DM and IPS
    """
    rewards = []
    for context, true_action, true_reward in data:
        selected_action = policy.select(context)
        if selected_action == true_action:
            # Observed reward
            reward = true_reward
        else:
            # Imputed reward from model
            reward = reward_model.predict(context, selected_action)
        rewards.append(reward)
    return np.mean(rewards)

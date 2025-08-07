
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# for BL Regression class


class BayesianLinearRegression:
    def __init__(self, n_features, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.posterior_mean = np.zeros(n_features)
        self.posterior_cov = np.eye(n_features) / alpha

    def update(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        cov_inv = np.linalg.inv(self.posterior_cov)
        self.posterior_cov = np.linalg.inv(cov_inv + self.beta * X.T @ X)
        self.posterior_mean = self.posterior_cov @ (
            cov_inv @ self.posterior_mean + self.beta * X.T @ y
        )

    def sample_weights(self):
        return np.random.multivariate_normal(self.posterior_mean, self.posterior_cov)

    def predict(self, X, weights):
        return X @ weights


# SAS data file
data_path = "C:/Users/LENOVO/Desktop/Hackathon/Model/datashare_aug2015.sas7bdat"
df = pd.read_sas(data_path, format="sas7bdat", encoding="latin1")

#  dataset stuffs
print("\nAll available columns:")
print(df.columns.tolist())
print("Loaded dataset with shape:", df.shape)
print(df.head())

# cols
columns_needed = [
    'age', 'gender', 'pretrialexp', 'treatment', 'surv18',
    'nihss', 'randdelay', 'sbprand', 'dbprand', 'glucose',
    'gcs_score_rand', 'weight'
]
df_clean = df[columns_needed].copy()
df_clean.dropna(subset=columns_needed, inplace=True)

# reward
df_clean['treatment_encoded'] = df_clean['treatment'].map(
    {'rt-PA': 1, 'Placebo': 0})
df_clean['reward'] = df_clean['surv18'] / 548  # Normalize survival time

print("Cleaned dataset preview:")
print(df_clean.head())

#  Random Forest Regressor

feature_cols = [
    'age', 'gender', 'pretrialexp', 'nihss', 'randdelay',
    'sbprand', 'dbprand', 'glucose', 'gcs_score_rand', 'weight',
    'treatment_encoded'
]
X = df_clean[feature_cols].values
y = df_clean['reward'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
r2_score = model.score(X_test, y_test)
print(f"\nModel R² score on test set: {r2_score:.4f}")

# Greedy cb
context_cols = [
    'age', 'gender', 'pretrialexp', 'nihss', 'randdelay',
    'sbprand', 'dbprand', 'glucose', 'gcs_score_rand', 'weight'
]
contexts = df_clean[context_cols].values

cumulative_reward = 0
reward_history = []

for i in range(len(contexts)):
    context = contexts[i]
    predicted_rewards = []
    for action in [0, 1]:
        input_features = np.array(list(context) + [action]).reshape(1, -1)
        predicted_reward = model.predict(input_features)[0]
        predicted_rewards.append(predicted_reward)

    best_action = np.argmax(predicted_rewards)
    best_predicted_reward = predicted_rewards[best_action]

    cumulative_reward += best_predicted_reward
    reward_history.append(cumulative_reward)

average_rewards = [r / (i + 1) for i, r in enumerate(reward_history)]

# Thompson Sampling
actions = df_clean['treatment_encoded'].values
rewards = df_clean['reward'].values

n_features = len(context_cols) + 1
model_0 = BayesianLinearRegression(n_features)
model_1 = BayesianLinearRegression(n_features)

cumulative_reward_ts = 0
reward_history_ts = []

for i in range(len(contexts)):
    context = contexts[i]
    x0 = np.array(list(context) + [0])
    x1 = np.array(list(context) + [1])

    w0 = model_0.sample_weights()
    w1 = model_1.sample_weights()

    r0 = model_0.predict(x0, w0)
    r1 = model_1.predict(x1, w1)

    if r1 > r0:
        chosen_action = 1
        x = x1
    else:
        chosen_action = 0
        x = x0

    if actions[i] == chosen_action:
        actual_reward = rewards[i]
    else:
        actual_reward = 0

    if chosen_action == 0:
        model_0.update(x, actual_reward)
    else:
        model_1.update(x, actual_reward)

    cumulative_reward_ts += actual_reward
    reward_history_ts.append(cumulative_reward_ts)

average_ts = [r / (i + 1) for i, r in enumerate(reward_history_ts)]


plt.figure(figsize=(10, 5))
sns.lineplot(x=range(len(average_rewards)),
             y=average_rewards, label="Greedy Policy")
sns.lineplot(x=range(len(average_ts)), y=average_ts, label="Thompson Sampling")
plt.title("Contextual Bandit - Average Reward Over Time")
plt.xlabel("Patient #")
plt.ylabel("Average Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

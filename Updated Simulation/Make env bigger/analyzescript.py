# analyze_model.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from training2 import CaneEnv

# =========================
# Configuration
# =========================
MODEL_PATH = "dqn_cane_model_01.zip"
N_EPISODES = 100  # number of episodes to evaluate
GUI = False      # Set True if you want to visually watch

# =========================
# Initialize environment
# =========================
env = CaneEnv(gui=GUI)

# =========================
# Load trained model
# =========================
model = DQN.load(MODEL_PATH)

# =========================
# Run evaluation
# =========================
episode_rewards = []
episode_steps = []
episode_collisions = []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    collisions = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1
        if info.get("collision_detected"):
            collisions += 1

    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    episode_collisions.append(collisions)

# =========================
# Convert to numpy for convenience
# =========================
episode_rewards = np.array(episode_rewards)
episode_steps = np.array(episode_steps)
episode_collisions = np.array(episode_collisions)

# =========================
# Print basic stats
# =========================
print(f"Mean reward: {episode_rewards.mean():.2f} ± {episode_rewards.std():.2f}")
print(f"Mean steps: {episode_steps.mean():.2f} ± {episode_steps.std():.2f}")
print(f"Mean collisions: {episode_collisions.mean():.2f} ± {episode_collisions.std():.2f}")

# =========================
# Plotting
# =========================
plt.figure(figsize=(12, 4))

# Episode reward
plt.subplot(1, 3, 1)
plt.plot(episode_rewards, marker='o')
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")

# Episode steps
plt.subplot(1, 3, 2)
plt.plot(episode_steps, marker='o', color='orange')
plt.title("Episode Steps")
plt.xlabel("Episode")
plt.ylabel("Steps")

# Collisions per episode
plt.subplot(1, 3, 3)
plt.plot(episode_collisions, marker='o', color='red')
plt.title("Collisions per Episode")
plt.xlabel("Episode")
plt.ylabel("Collisions")

plt.tight_layout()
plt.savefig("model_analysis.png", dpi=300)
plt.show()

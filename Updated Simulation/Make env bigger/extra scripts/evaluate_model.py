import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from training2 import CaneEnv

MODEL_PATH = "dqn_cane_model"
N_EPISODES = 200

env = CaneEnv(gui=False)
model = DQN.load(MODEL_PATH)

rewards = []
lengths = []
collisions = []
successes = []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False

    total_reward = 0
    steps = 0
    ep_collisions = 0
    success = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        ep_collisions += info.get("collision_detected", 0)

        if info.get("goal_reached", False):
            success = 1

    rewards.append(total_reward)
    lengths.append(steps)
    collisions.append(ep_collisions)
    successes.append(success)

# =========================
# METRICS
# =========================

print("\n===== EVALUATION RESULTS =====")
print(f"Success rate: {np.mean(successes)*100:.2f}%")
print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"Mean steps: {np.mean(lengths):.2f}")
print(f"Mean collisions: {np.mean(collisions):.2f}")

# =========================
# PLOTS
# =========================

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.hist(rewards, bins=20)
plt.title("Reward Distribution")

plt.subplot(1,3,2)
plt.hist(lengths, bins=20)
plt.title("Episode Length")

plt.subplot(1,3,3)
plt.hist(collisions, bins=20)
plt.title("Collisions per Episode")

plt.tight_layout()
plt.show()
from training2 import CaneEnv
from stable_baselines3 import DQN
import time

# === 1. Create GUI environment for visual run ===
env = CaneEnv(gui=True)

# === 2. Load trained model ===
model = DQN.load("dqn_cane_model_25")

# === 3. Reset environment ===
obs, info = env.reset()
done = False
total_reward = 0
step_count = 0

# === 4. Run manual visual test ===
while not done:
    # Predict action (greedy)
    action, _ = model.predict(obs, deterministic=True)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Episode done if terminated or truncated
    done = terminated or truncated

    total_reward += reward
    step_count += 1

    # Slow down for visual clarity
    #time.sleep(5/60)

print(f"Episode finished in {step_count} steps")
print(f"Cumulative reward: {total_reward}")

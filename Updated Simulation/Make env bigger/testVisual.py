from gridBaseObstacles import CaneEnv
from stable_baselines3 import DQN
import time

# Create GUI env for visual run
env = CaneEnv(gui=True)

# Load trained model
model = DQN.load("dqn_cane_model_13.zip")

obs, _ = env.reset()
done = False
total_reward = 0
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    total_reward += reward
    step_count += 1

    time.sleep(1/60)  # Slow down for visual clarity

print(f"Episode finished in {step_count} steps")
print(f"Cumulative reward: {total_reward}")

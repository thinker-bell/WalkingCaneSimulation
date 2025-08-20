import gymnasium as gym  # or gym if you're using classic gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from gridBaseObstacles import CaneEnv  # Your custom environment

# === 1. Create and wrap the environment ===
def make_env():
    env = CaneEnv()
    env = Monitor(env)  # Required for proper episode logging
    return env

env = DummyVecEnv([make_env])  # SB3 expects a vectorized env

# === 2. Load the trained model ===
model = DQN.load("dqn_cane_model.zip")

# === 3. Evaluate (both deterministic and exploratory) ===
print("Greedy :")
mean_det, std_det = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
print(f"Mean reward (deterministic): {mean_det}")
print(f"Standard deviation: {std_det}\n")

print("Exploratory:")
mean_stoch, std_stoch = evaluate_policy(model, env, n_eval_episodes=50, deterministic=False)
print(f"Mean reward (exploratory): {mean_stoch}")
print(f"Standard deviation: {std_stoch}\n")

# === 4. Optional: Run one test episode manually and print reward ===
print("Running one manual episode:")
obs = env.reset()
done = [False]
total_reward = 0

while not done[0]:
    action, _ = model.predict(obs, deterministic=True)  # Set to False for exploratory behavior
    obs, reward, done, _ = env.step(action)
    total_reward += reward[0]

print(f"Total reward from manual episode: {total_reward}")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from training_PPO import CaneEnv


# === 1. Create environment ===
def make_env():
    env = CaneEnv(gui=False)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])  # 1-env vector for evaluation


# === 2. Load PPO model ===
model = PPO.load("ppo_cane_model_03", env=env)


# === 3. Evaluate policy ===
print("Greedy (deterministic):")
mean_det, std_det = evaluate_policy(
    model,
    env,
    n_eval_episodes=50,
    deterministic=True
)
print(f"Mean reward: {mean_det:.2f}")
print(f"Std reward : {std_det:.2f}\n")


print("Exploratory (stochastic):")
mean_stoch, std_stoch = evaluate_policy(
    model,
    env,
    n_eval_episodes=50,
    deterministic=False
)
print(f"Mean reward: {mean_stoch:.2f}")
print(f"Std reward : {std_stoch:.2f}\n")


# === 4. Manual rollout (Gymnasium-correct) ===
print("Running one manual episode...")
obs = env.reset()
done = [False]  # vectorized env returns lists
total_reward = 0.0

while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]  # index 0 because VecEnv returns arrays

print(f"Total reward (manual episode): {total_reward:.2f}")

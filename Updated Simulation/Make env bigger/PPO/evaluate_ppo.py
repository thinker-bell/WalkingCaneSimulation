import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
import pybullet as p
# Import your environment
from training2 import CaneEnv     # <-- Replace with your filename


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_TYPE = "PPO"               # Change to "DQN" when evaluating DQN
MODEL_PATH = "PPO_attempts_01.zip"
NUM_EPISODES = 200
GUI = False


# ==========================================================
# LOAD ENVIRONMENT
# ==========================================================

env = CaneEnv(gui=GUI)

if MODEL_TYPE == "PPO":
    model = PPO.load(MODEL_PATH)

elif MODEL_TYPE == "DQN":
    model = DQN.load(MODEL_PATH)

else:
    raise ValueError("MODEL_TYPE must be PPO or DQN")


results = []

print(f"\nEvaluating {MODEL_TYPE}...\n")


# ==========================================================
# EVALUATION LOOP
# ==========================================================

for episode in range(NUM_EPISODES):

    obs, info = env.reset()

    terminated = False
    truncated = False

    episode_reward = 0
    inference_times = []

    while not (terminated or truncated):

        start = time.perf_counter()

        action, _ = model.predict(obs, deterministic=True)

        inference_times.append(time.perf_counter() - start)

        obs, reward, terminated, truncated, info = env.step(action)


        episode_reward += reward

    final_pos, _ = p.getBasePositionAndOrientation(env.cane_id)
    final_distance = np.linalg.norm(np.array(final_pos) - env.goal_location)

    print(
        f"Episode {episode+1} | "
        f"Distance to goal: {final_distance:.2f}"
    )

    success = int(info["goal_reached"])

    results.append({

        "Episode": episode + 1,

        "Success": success,

        "Reward": episode_reward,

        "EpisodeLength": info["steps_taken"],

        "Collisions": env.episode_collisions,

        "InferenceTime(ms)": np.mean(inference_times) * 1000,

        "StartX": env.cane_start_pos[0],
        "StartY": env.cane_start_pos[1],

        "GoalX": env.goal_location[0],
        "GoalY": env.goal_location[1]

    })

    print(
        f"Episode {episode+1:3d} | "
        f"Success: {success} | "
        f"Reward: {episode_reward:8.2f} | "
        f"Steps: {info['steps_taken']:3d} | "
        f"Collisions: {env.episode_collisions}"
    )


# ==========================================================
# SAVE RESULTS
# ==========================================================

df = pd.DataFrame(results)

csv_name = f"{MODEL_TYPE}_evaluation.csv"

df.to_csv(csv_name, index=False)

print(f"\nSaved evaluation to {csv_name}")


# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

print("\n==============================")
print("Evaluation Summary")
print("==============================")

print(f"Episodes               : {NUM_EPISODES}")

print(f"Success Rate (%)       : {df['Success'].mean()*100:.2f}")

print(f"Average Reward         : {df['Reward'].mean():.2f}")

print(f"Reward Std             : {df['Reward'].std():.2f}")

print(f"Average Episode Length : {df['EpisodeLength'].mean():.2f}")

print(f"Episode Length Std     : {df['EpisodeLength'].std():.2f}")

print(f"Average Collisions     : {df['Collisions'].mean():.2f}")

print(f"Collision Std          : {df['Collisions'].std():.2f}")

print(f"Average Inference Time : {df['InferenceTime(ms)'].mean():.3f} ms")


summary = pd.DataFrame({

    "Metric": [
        "Success Rate (%)",
        "Average Reward",
        "Reward Std",
        "Average Episode Length",
        "Episode Length Std",
        "Average Collisions",
        "Collision Std",
        "Average Inference Time (ms)"
    ],

    "Value": [

        df["Success"].mean()*100,

        df["Reward"].mean(),

        df["Reward"].std(),

        df["EpisodeLength"].mean(),

        df["EpisodeLength"].std(),

        df["Collisions"].mean(),

        df["Collisions"].std(),

        df["InferenceTime(ms)"].mean()

    ]

})

summary.to_csv(f"{MODEL_TYPE}_summary.csv", index=False)

print(f"Saved summary to {MODEL_TYPE}_summary.csv")


env.close()
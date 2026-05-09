"""
validation.py

Batch validation framework for:
- Collision detection correctness
- Goal detection correctness
- Environment consistency

Run:
    python validation.py
"""

import numpy as np
import pandas as pd
from datetime import datetime
from validation_script import CaneEnv
import os

# 🔁 CHANGE THIS IMPORT TO MATCH YOUR PROJECT
#from env.your_env import YourEnv   # <-- UPDATE THIS


# =========================================================
# 🔧 CONFIGURATION
# =========================================================

SCENARIOS = [
    "direct_collision",
    "near_miss",
    "goal_only",
    "edge_goal",
    "edge_collision"
]

EPISODES_PER_SCENARIO = 2000
MAX_STEPS = 200


# =========================================================
# 🧪 VALIDATION RUNNER
# =========================================================

def run_validation(env, scenarios, episodes_per_scenario=50):

    results = []

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}")

        for ep in range(episodes_per_scenario):

            obs, _ = env.reset(test_config={"type": scenario})

            done = False
            steps = 0

            metrics = {
                "scenario": scenario,
                "episode": ep,
                "true_collision": 0,
                "detected_collision": 0,
                "false_positive": 0,
                "missed_collision": 0,
                "goal_reached": 0,
                "final_distance": None,
                "steps": 0
            }

            while not done:
                
                # 🔁 SIMPLE TEST POLICY (MODIFY IF NEEDED)
                action = get_test_action(env, obs)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # -------------------------------
                # VALIDATION LOGIC
                # -------------------------------
                true_col = info.get("true_collision", False)
                det_col = info.get("detected_collision", False)

                if true_col:
                    metrics["true_collision"] += 1

                if det_col:
                    metrics["detected_collision"] += 1

                if det_col and not true_col:
                    metrics["false_positive"] += 1

                if true_col and not det_col:
                    metrics["missed_collision"] += 1

                if true_col and not det_col:
                    metrics["missed_collision"] += 1

                if info.get("goal_reached", False):
                    metrics["goal_reached"] = 1

                metrics["final_distance"] = info.get("distance_to_goal", None)

                steps += 1

                if steps >= MAX_STEPS:
                    break

            metrics["steps"] = steps
            results.append(metrics)

    return pd.DataFrame(results)


# =========================================================
# 🎮 TEST POLICY
# =========================================================

def get_test_action(env, obs):

    # 70% forward movement, 30% rotation
    if np.random.rand() < 0.7:
        return np.random.choice([0, 1, 2])
    else:
        return np.random.choice([4, 5, 6, 7, 8, 9, 10])


# =========================================================
# 📊 METRICS SUMMARY
# =========================================================

def summarize_results(df):

    summary = {}

    # Collision accuracy
    summary["collision_accuracy"] = (
        (df["true_collision"] == df["detected_collision"]).mean()
    )

    # False positives
    summary["false_positive_rate"] = (
        df["false_positive"].sum() / (df.shape[0] * MAX_STEPS)
    )

    # Missed collisions
    summary["missed_collision_rate"] = (
        df["missed_collision"].sum() / len(df)
    )

    # Goal success
    summary["goal_success_rate"] = df["goal_reached"].mean()

    # Avg steps
    summary["avg_steps"] = df["steps"].mean()

    return summary


# =========================================================
# 💾 SAVE RESULTS
# =========================================================

def save_results(df, summary):

    os.makedirs("results", exist_ok=True)  # ✅ THIS FIXES IT

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = f"results/validation_results_{timestamp}.csv"
    summary_file = f"results/validation_summary_{timestamp}.txt"

    df.to_csv(results_file, index=False)

    with open(summary_file, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"\nSaved results to: {results_file}")
    print(f"Saved summary to: {summary_file}")


# =========================================================
# 🚀 MAIN
# =========================================================

def main():

    print("Starting validation...")

    env = CaneEnv()

    df = run_validation(
        env,
        scenarios=SCENARIOS,
        episodes_per_scenario=EPISODES_PER_SCENARIO
    )

    summary = summarize_results(df)

    print("\n========== SUMMARY ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")

    save_results(df, summary)


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ============================================================
# SETTINGS
# ============================================================

EVENT_FILES = [
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\PPO\events.out.tfevents.1784566265.DESKTOP-US2JO9N.4068.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\PPO\events.out.tfevents.1784746967.DESKTOP-US2JO9N.11772.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\PPO\events.out.tfevents.1784812210.DESKTOP-US2JO9N.28400.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\PPO\events.out.tfevents.1784818533.DESKTOP-US2JO9N.8488.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\PPO\events.out.tfevents.1784825433.DESKTOP-US2JO9N.3464.0"
]

# TensorBoard scalar to plot
SCALAR_TAG = "custom/ep_reward"
# Or:
# SCALAR_TAG = "rollout/ep_rew_mean"

NUM_POINTS = 500
SMOOTH_WINDOW = 25

# ============================================================
# MOVING AVERAGE (works for even and odd window sizes)
# ============================================================

def moving_average(data, window):
    kernel = np.ones(window) / window

    left = (window - 1) // 2
    right = window // 2

    padded = np.pad(data, (left, right), mode="edge")

    return np.convolve(padded, kernel, mode="valid")

# ============================================================
# LOAD RUNS
# ============================================================

runs = []

for file in EVENT_FILES:

    print(f"\nLoading: {file}")

    ea = event_accumulator.EventAccumulator(file)
    ea.Reload()

    print("Available Scalars:")
    print(ea.Tags()["scalars"])

    if SCALAR_TAG not in ea.Tags()["scalars"]:
        raise ValueError(f"{SCALAR_TAG} not found in {file}")

    events = ea.Scalars(SCALAR_TAG)

    steps = np.array([e.step for e in events])
    rewards = np.array([e.value for e in events])

    runs.append((steps, rewards))

# ============================================================
# COMMON TRAINING AXIS
# ============================================================

max_common_step = min(run[0][-1] for run in runs)

common_steps = np.linspace(
    0,
    max_common_step,
    NUM_POINTS
)

# ============================================================
# INTERPOLATE RUNS
# ============================================================

reward_matrix = []

for steps, rewards in runs:

    interp = np.interp(
        common_steps,
        steps,
        rewards
    )

    reward_matrix.append(interp)

reward_matrix = np.array(reward_matrix)

# ============================================================
# STATISTICS
# ============================================================

mean_reward = np.mean(reward_matrix, axis=0)
std_reward = np.std(reward_matrix, axis=0)

# ============================================================
# SMOOTH
# ============================================================

mean_reward = moving_average(mean_reward, SMOOTH_WINDOW)
std_reward = moving_average(std_reward, SMOOTH_WINDOW)

# Safety check
assert len(common_steps) == len(mean_reward)
assert len(common_steps) == len(std_reward)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    common_steps,
    mean_reward,
    linewidth=2
)

plt.fill_between(
    common_steps,
    mean_reward - std_reward,
    mean_reward + std_reward,
    alpha=0.10
)

plt.xlabel("Training Timesteps", fontsize=11)
plt.ylabel("Episode Reward", fontsize=11)

plt.title(
    "Mean PPO Training Reward Across Five Training Runs",
    fontsize=13,
    fontweight="bold"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "PPO_Training_Curve_5_Seeds.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(8,5))

for i, (steps, rewards) in enumerate(runs):
    plt.plot(steps, rewards, alpha=0.7, label=f"Seed {i+1}")

plt.xlabel("Training Timesteps")
plt.ylabel("Episode Reward")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
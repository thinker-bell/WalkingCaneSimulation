import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ============================================================
# SETTINGS
# ============================================================

EVENT_FILES = [
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\DQN\events.out.tfevents.1784610268.DESKTOP-US2JO9N.15560.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\DQN\events.out.tfevents.1784812230.DESKTOP-US2JO9N.16588.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\DQN\events.out.tfevents.1784819509.DESKTOP-US2JO9N.27752.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\DQN\events.out.tfevents.1784821871.DESKTOP-US2JO9N.20084.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\DQN\events.out.tfevents.1784824266.DESKTOP-US2JO9N.12168.0"
]

# Choose ONE:
SCALAR_TAG = "custom/ep_reward"
# SCALAR_TAG = "rollout/ep_rew_mean"

# Number of interpolation points
NUM_POINTS = 500

# Moving average window
SMOOTH_WINDOW = 25

# ============================================================
# LOAD ALL RUNS
# ============================================================

runs = []

for file in EVENT_FILES:

    ea = event_accumulator.EventAccumulator(file)
    ea.Reload()

    print(f"\nLoaded: {file}")
    print("Available Scalars:")
    print(ea.Tags()["scalars"])

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
# INTERPOLATE EACH RUN
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
# COMPUTE MEAN ± STD
# ============================================================

mean_reward = reward_matrix.mean(axis=0)
std_reward = reward_matrix.std(axis=0)

# ============================================================
# SMOOTH ONLY THE MEAN
# ============================================================

kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW

mean_reward = np.convolve(
    mean_reward,
    kernel,
    mode="same"
)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    common_steps,
    mean_reward,
    linewidth=2,
    label="Mean Reward"
)

plt.fill_between(
    common_steps,
    mean_reward - std_reward,
    mean_reward + std_reward,
    alpha=0.15,
    label="±1 Standard Deviation"
)

plt.xlabel("Training Timesteps ", fontsize=11)
plt.ylabel("Episode Reward", fontsize=11)

plt.title(
    "Mean DQN Training Reward Across Five Training Runs",
    fontsize=13
)

plt.grid(True, alpha=0.3)

plt.legend(frameon=False)

plt.tight_layout()

plt.savefig(
    "DQN_Training_Curve_5_Seeds.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
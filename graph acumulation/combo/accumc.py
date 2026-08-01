import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ============================================================
# SETTINGS
# ============================================================

DQN_EVENT_FILES = [
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\DQN\events.out.tfevents.1784610268.DESKTOP-US2JO9N.15560.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\DQN\events.out.tfevents.1784812230.DESKTOP-US2JO9N.16588.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\DQN\events.out.tfevents.1784819509.DESKTOP-US2JO9N.27752.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\DQN\events.out.tfevents.1784821871.DESKTOP-US2JO9N.20084.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\DQN\events.out.tfevents.1784824266.DESKTOP-US2JO9N.12168.0"
]

PPO_EVENT_FILES = [
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\PPO\events.out.tfevents.1784566265.DESKTOP-US2JO9N.4068.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\PPO\events.out.tfevents.1784746967.DESKTOP-US2JO9N.11772.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\PPO\events.out.tfevents.1784812210.DESKTOP-US2JO9N.28400.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\PPO\events.out.tfevents.1784818533.DESKTOP-US2JO9N.8488.0",
    r"C:\Users\Ruchelle\Desktop\WalkingCane\WalkingCaneSimulation\graph acumulation\combo\PPO\events.out.tfevents.1784825433.DESKTOP-US2JO9N.3464.0"
]

# Use the SAME scalar for both algorithms
SCALAR_TAG = "custom/ep_reward"
# Or:
# SCALAR_TAG = "rollout/ep_rew_mean"

NUM_POINTS = 500
SMOOTH_WINDOW = 25

# ============================================================
# MOVING AVERAGE
# ============================================================

def moving_average(data, window):
    kernel = np.ones(window) / window

    left = (window - 1) // 2
    right = window // 2

    padded = np.pad(data, (left, right), mode="edge")

    return np.convolve(padded, kernel, mode="valid")

# ============================================================
# LOAD FUNCTION
# ============================================================

def load_algorithm(event_files):

    runs = []

    for file in event_files:

        ea = event_accumulator.EventAccumulator(file)
        ea.Reload()

        events = ea.Scalars(SCALAR_TAG)

        steps = np.array([e.step for e in events])
        rewards = np.array([e.value for e in events])

        runs.append((steps, rewards))

    max_common_step = min(run[0][-1] for run in runs)

    common_steps = np.linspace(
        0,
        max_common_step,
        NUM_POINTS
    )

    reward_matrix = []

    for steps, rewards in runs:

        interp = np.interp(
            common_steps,
            steps,
            rewards
        )

        reward_matrix.append(interp)

    reward_matrix = np.array(reward_matrix)

    mean_reward = reward_matrix.mean(axis=0)
    std_reward = reward_matrix.std(axis=0)

    mean_reward = moving_average(
        mean_reward,
        SMOOTH_WINDOW
    )

    std_reward = moving_average(
        std_reward,
        SMOOTH_WINDOW
    )

    return common_steps, mean_reward, std_reward

# ============================================================
# LOAD BOTH ALGORITHMS
# ============================================================

dqn_steps, dqn_mean, dqn_std = load_algorithm(DQN_EVENT_FILES)

ppo_steps, ppo_mean, ppo_std = load_algorithm(PPO_EVENT_FILES)

# ============================================================
# COMMON X AXIS
# ============================================================

max_step = min(dqn_steps[-1], ppo_steps[-1])

common_steps = np.linspace(0, max_step, NUM_POINTS)

dqn_mean = np.interp(common_steps, dqn_steps, dqn_mean)
dqn_std = np.interp(common_steps, dqn_steps, dqn_std)

ppo_mean = np.interp(common_steps, ppo_steps, ppo_mean)
ppo_std = np.interp(common_steps, ppo_steps, ppo_std)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(9,6))

# ---------------- DQN ----------------

plt.plot(
    common_steps,
    dqn_mean,
    linewidth=2,
    label="DQN"
)

plt.fill_between(
    common_steps,
    dqn_mean - dqn_std,
    dqn_mean + dqn_std,
    alpha=0.15
)

# ---------------- PPO ----------------

plt.plot(
    common_steps,
    ppo_mean,
    linewidth=2,
    label="PPO"
)

plt.fill_between(
    common_steps,
    ppo_mean - ppo_std,
    ppo_mean + ppo_std,
    alpha=0.15
)

plt.title(
    "Comparison of DQN and PPO Learning Behaviour",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Training Timesteps")
plt.ylabel("Episode Reward")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Learning_Behaviour_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
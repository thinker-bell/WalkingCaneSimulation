import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Monitor CSV ---
log_file = "./logs/cane_monitor_02.csv.monitor.csv"
df = pd.read_csv(
    "./logs/cane_monitor_02.csv.monitor.csv",
    skiprows=1
)

# --- 2. Compute moving averages ---
window = 50  # smoothing window
df['reward_ma'] = df['r'].rolling(window).mean()
df['length_ma'] = df['l'].rolling(window).mean()

# Optional: if you logged collisions in info dict
if 'collisions' in df.columns:
    df['collisions_ma'] = df['collisions'].rolling(window).mean()

window = 50
df["reward_smoothed"] = df["r"].rolling(window).mean()

plt.figure(figsize=(10, 5))
plt.scatter(range(len(df)), df["r"], s=8, alpha=0.3, label="Episode reward")
plt.plot(df["reward_smoothed"], linewidth=2, label=f"Rolling mean ({window})")
plt.xlabel("Training Episode")
plt.ylabel("Total Episode Reward")
plt.title("Learning Behaviour Over Training")
plt.legend()
plt.grid(True)
plt.show()


# --- 3. Plot episode reward ---
plt.figure(figsize=(10,6))
plt.plot(df['t'], df['reward_ma'], label=f'Smoothed Reward (window={window})')
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('DQN Training: Episode Reward vs Timesteps')
plt.legend()
plt.grid(True)
plt.show()

# --- 4. Plot episode length ---
plt.figure(figsize=(10,6))
plt.plot(df['t'], df['length_ma'], label=f'Smoothed Episode Length (window={window})', color='orange')
plt.xlabel('Timesteps')
plt.ylabel('Episode Length (timesteps)')
plt.title('Episode Length vs Timesteps')
plt.legend()
plt.grid(True)
plt.show()

# --- 5. Plot collisions if available ---
if 'collisions' in df.columns:
    plt.figure(figsize=(10,6))
    plt.plot(df['t'], df['collisions_ma'], label=f'Smoothed Collisions per Episode (window={window})', color='red')
    plt.xlabel('Timesteps')
    plt.ylabel('Collisions')
    plt.title('Collisions During Training')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 6. Optional: cumulative reward over time ---
df['cumulative_reward'] = df['r'].cumsum()
plt.figure(figsize=(10,6))
plt.plot(df['t'], df['cumulative_reward'], label='Cumulative Reward', color='green')
plt.xlabel('Timesteps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward During Training')
plt.legend()
plt.grid(True)
plt.show()

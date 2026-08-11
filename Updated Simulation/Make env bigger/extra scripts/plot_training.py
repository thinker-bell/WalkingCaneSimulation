# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# log = pd.read_csv("./logs/cane_26.monitor.csv", skiprows=1)

# rewards = log["r"]
# lengths = log["l"]

# # Rolling mean reward
# window = 50
# rolling = rewards.rolling(window).mean()

# plt.figure(figsize=(12,5))

# plt.plot(rewards, alpha=0.3, label="Episode reward")
# plt.plot(rolling, linewidth=2, label="Rolling mean (50)")

# plt.title("Training Reward Curve")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.legend()
# plt.savefig("training_reward_curve.png", dpi=300)
# plt.show()

import pandas as pd

df = pd.read_csv("./logs/cane_26.monitor.csv", skiprows=0, on_bad_lines="skip")
print(df.head())
print(df.columns)
print(df.shape)
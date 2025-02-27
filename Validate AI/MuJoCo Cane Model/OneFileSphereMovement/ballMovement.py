import mujoco as mj
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# XML Model Definition for a Cane
CANE_XML = """
<mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <!-- Floor -->
        <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
        
        <!-- Cane -->
        <body name="cane" pos="0 0 0.5">
            <joint name="cane_joint_x" type="slide" axis="1 0 0"/>
            <joint name="cane_joint_y" type="slide" axis="0 1 0"/>
            <joint name="cane_joint_rot" type="hinge" axis="0 0 1"/>
            <geom name="cane_geom" type="capsule" fromto="0 0 0.5 0 0 -0.5" size="0.05" mass="1" rgba="0 0 1 1"/>
        </body>
    </worldbody>
    
    <actuator>
        <motor name="x_motor" joint="cane_joint_x" ctrlrange="-1 1"/>
        <motor name="y_motor" joint="cane_joint_y" ctrlrange="-1 1"/>
        <motor name="rot_motor" joint="cane_joint_rot" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

class CaneEnv(gym.Env):
    """Custom Gymnasium environment for a moving cane in MuJoCo."""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.model = mj.MjModel.from_xml_string(CANE_XML)
        self.data = mj.MjData(self.model)
        self.viewer = None
        
        self.action_space = spaces.Discrete(6)  # 0: Stop, 1: Forward, 2: Backward, 3: Left, 4: Right, 5: Rotate
        self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)

    def step(self, action):
        if action == 1:
            self.data.ctrl[1] = 1  # Forward
        elif action == 2:
            self.data.ctrl[1] = -1  # Backward
        elif action == 3:
            self.data.ctrl[0] = -1  # Left
        elif action == 4:
            self.data.ctrl[0] = 1  # Right
        elif action == 5:
            self.data.ctrl[2] = 1  # Rotate
        else:
            self.data.ctrl[:] = 0  # Stop
        
        for _ in range(10):  # Simulate multiple steps for smooth motion
            mj.mj_step(self.model, self.data)
        
        obs = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]])
        reward = -np.linalg.norm(obs[:2])  # Reward based on proximity to origin
        done = False
        return obs, reward, done, {}

    def reset(self, seed=None, options=None):
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        return np.array([0.0, 0.0, 0.0]), {}

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mj.viewer.launch_passive(self.model, self.data)
        while self.viewer.is_running():
            mj.mj_step(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

env = CaneEnv()
env.reset()
env.render()

action_sequence = [1, 2, 3, 4, 5, 0]
for action in action_sequence:
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Obs: {obs}, Reward: {reward}")

'''
current cane only moves in the 5 action spaces 
putting in the swimg component is quite tough to consider
'''



import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from gymnasium import spaces



class CaneEnv(gym.Env):
    def __init__(self):
        super(CaneEnv, self).__init__()

        # Connect to PyBullet in GUI mode and set the search path
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.resetSimulation()

        # Load a plane for visual reference
        self.plane_id = p.loadURDF("plane.urdf")

        # Define cane properties
        self.cane_radius = 0.05    # Radius of the cylinder (cane)
        self.cane_height = 2.0     # Total length of the cane
        self.cane_mass = 1.0       # Mass of the cane

        # To have the cane near the ground when tilted 45°,
        # the vertical offset from its center to its bottom is:
        # (cane_height / 2) * cos(45°)
        vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))
        # Set the cane's center so that its lower tip is almost at z=0.
        self.cane_start_pos = [0, 0, vertical_offset]

        # Set initial orientation: 45° tilt about the X-axis.
        self.initial_orientation = p.getQuaternionFromEuler([math.radians(45), 0, 0])

        # Create the cane multi-body
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                 radius=self.cane_radius,
                                                 height=self.cane_height)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                           radius=self.cane_radius,
                                           length=self.cane_height,
                                           rgbaColor=[0, 0, 0.85, 1])
        self.cane_id = p.createMultiBody(baseMass=self.cane_mass,
                                         baseCollisionShapeIndex=collision_shape,
                                         baseVisualShapeIndex=visual_shape,
                                         basePosition=self.cane_start_pos,
                                         baseOrientation=self.initial_orientation)

        # Define a discrete action space:
        # 0: Move forward (+Y)
        # 1: Move backward (-Y)
        # 2: Move left (-X)
        # 3: Move right (+X)
        # 4: Stop (no movement)
        self.action_space = spaces.Discrete(5)
        # Observation: the (x, y, z) position of the cane's center.
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0]),
            high=np.array([10, 10, 10]),
            dtype=np.float32
        )

    def step(self, action):
        # Retrieve current position (ignore orientation here as it stays fixed)
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        step_size = 0.5

        if action == 0:  # Move Forward (+Y)
            new_pos = [pos[0], pos[1] + step_size, pos[2]]
        elif action == 1:  # Move Backward (-Y)
            new_pos = [pos[0], pos[1] - step_size, pos[2]]
        elif action == 2:  # Move Left (-X)
            new_pos = [pos[0] - step_size, pos[1], pos[2]]
        elif action == 3:  # Move Right (+X)
            new_pos = [pos[0] + step_size, pos[1], pos[2]]
        elif action == 4:  # Stop (no change)
            new_pos = pos

        # Update the cane's position while keeping its orientation constant
        p.resetBasePositionAndOrientation(self.cane_id, new_pos, self.initial_orientation)

        # Simple reward function: for example, reward forward movement
        reward = new_pos[1]
        done = False

        return np.array(new_pos, dtype=np.float32), reward, done, {}

    def reset(self):
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, self.initial_orientation)
        return np.array(self.cane_start_pos, dtype=np.float32)

    def render(self, mode="human"):
        pass  # GUI is already active

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = CaneEnv()
    env.reset()

    try:
        while True:
            action = env.action_space.sample()  # Replace with your own policy
            env.step(action)
            time.sleep(0.2)  # Slow down to observe movements
    except KeyboardInterrupt:
        env.close()

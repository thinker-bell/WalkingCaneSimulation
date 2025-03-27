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

        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        # Load the plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Cane properties
        self.cane_radius = 0.05
        self.cane_height = 2.0
        self.cane_mass = 1.0

        # Fixed baseline
        self.baseline_roll = math.radians(45)
        self.baseline_pitch = 0
        self.current_swing_deg = 0  # Initial swing

        # Compute vertical offset for starting position
        vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))
        self.cane_start_pos = [0, 0, vertical_offset]

        # Initial orientation
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )

        # Create the cane
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                 radius=self.cane_radius,
                                                 height=self.cane_height)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                           radius=self.cane_radius,
                                           length=self.cane_height,
                                           rgbaColor=[1, 0.75, 0.8, 1])  # Pink color.

        # **Move Center of Mass to Handle**
        self.cane_id = p.createMultiBody(
            baseMass=self.cane_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.cane_start_pos,
            baseOrientation=initial_orientation,
            baseInertialFramePosition=[0, 0, self.cane_height * 0.4],  # Move CoM UP
            baseInertialFrameOrientation=[0, 0, 0, 1]
        )

        # Action & Observation Space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0]),
            high=np.array([10, 10, 10]),
            dtype=np.float32
        )

        self.dt = 1.0 / 240.0  # Time step

    def step(self, action):
        self.swing_cycle()

        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        step_size = 0.3

        if action == 0:  # Move forward (+Y)
            new_pos = pos + np.array([0, step_size, 0])
        elif action == 1:  # Move backward (-Y)
            new_pos = pos + np.array([0, -step_size, 0])
        elif action == 2:  # Move left (-X)
            new_pos = pos + np.array([-step_size, 0, 0])
        elif action == 3:  # Move right (+X)
            new_pos = pos + np.array([step_size, 0, 0])
        else:  # Stop
            new_pos = pos

        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), final_orientation)

        reward = new_pos[1]
        done = False
        return new_pos, reward, done, {}

    def swing_cycle(self):
        swing_up = np.linspace(0, 80, num=10)
        swing_down = np.linspace(80, -80, num=20)
        swing_return = np.linspace(-80, 0, num=10)
        full_cycle = np.concatenate((swing_up, swing_down, swing_return))

        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)

        for angle in full_cycle:
            self.current_swing_deg = angle
            new_orientation = p.getQuaternionFromEuler(
                [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
            )
            p.resetBasePositionAndOrientation(self.cane_id, pos.tolist(), new_orientation)
            p.stepSimulation()
            time.sleep(self.dt)

        self.current_swing_deg = 0
        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, pos.tolist(), final_orientation)

    def reset(self):
        self.current_swing_deg = 0
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, initial_orientation)
        return np.array(self.cane_start_pos, dtype=np.float32)

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    env = CaneEnv()
    env.reset()

    try:
        while True:
            action = env.action_space.sample()
            env.step(action)
            time.sleep(0.6)
    except KeyboardInterrupt:
        env.close()

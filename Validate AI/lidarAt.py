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

        # Connect to PyBullet in GUI mode and set up simulation.
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        # Load a plane for reference.
        self.plane_id = p.loadURDF("plane.urdf")

        # Cane properties.
        self.cane_radius = 0.05  
        self.cane_height = 2.0   
        self.cane_mass = 1.0     

        # LiDAR properties
        self.lidar_range_near = 1.0  # 1 meter range for near-space LiDAR
        self.lidar_range_far = 5.0   # 5 meters for far-space LiDAR

        # Cane starting orientation
        self.baseline_roll = math.radians(45)
        self.baseline_pitch = 0
        self.current_swing_deg = 0  

        # Compute vertical offset
        vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))
        self.cane_start_pos = [0, 0, vertical_offset]

        # Initial orientation
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )

        # Create the cane as a cylinder
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                 radius=self.cane_radius,
                                                 height=self.cane_height)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                           radius=self.cane_radius,
                                           length=self.cane_height,
                                           rgbaColor=[1, 0.75, 0.8, 1])
        self.cane_id = p.createMultiBody(baseMass=self.cane_mass,
                                         baseCollisionShapeIndex=collision_shape,
                                         baseVisualShapeIndex=visual_shape,
                                         basePosition=self.cane_start_pos,
                                         baseOrientation=initial_orientation)

        # Define action space
        self.action_space = spaces.Discrete(5)

        # Observation space (includes LiDAR readings)
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0, 0, 0]),  # LiDAR values start from 0
            high=np.array([10, 10, 10, self.lidar_range_near, self.lidar_range_far]),
            dtype=np.float32
        )

        self.dt = 1.0 / 240.0

    def get_lidar_readings(self):
        """
        Simulate LiDAR readings using PyBullet ray tracing.
        Also visualizes the LiDAR rays.
        """
        pos, orientation = p.getBasePositionAndOrientation(self.cane_id)
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        

        # Extract forward direction
        forward_vector = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])

        # Near LiDAR (bottom of the cane)
        near_start = np.array(pos) + forward_vector * 0.1  # Small offset from the cane
        near_end = near_start + forward_vector * self.lidar_range_near

        # Far LiDAR (higher up, looking further ahead)
        far_start = np.array(pos) + forward_vector * 1.0  # Offset forward
        far_end = far_start + forward_vector * self.lidar_range_far

        # Perform ray tests
        near_hit = p.rayTest(near_start.tolist(), near_end.tolist())[0]
        far_hit = p.rayTest(far_start.tolist(), far_end.tolist())[0]

        # Extract distances (if no hit, return max range)
        near_distance = near_hit[2] if near_hit[0] != -1 else self.lidar_range_near
        far_distance = far_hit[2] if far_hit[0] != -1 else self.lidar_range_far

        # Debug visualization
        p.addUserDebugLine(near_start, near_end, [1, 0, 0], lineWidth=2, lifeTime=0.1)  # Red for near LiDAR
        p.addUserDebugLine(far_start, far_end, [0, 1, 0], lineWidth=2, lifeTime=0.1)   # Green for far LiDAR

        return [near_distance, far_distance]


    def step(self, action):
        self.swing_cycle()

        # Movement
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        step_size = 0.3

        if action == 0:      # Move forward (+Y)
            new_pos = pos + np.array([0, step_size, 0])
        elif action == 1:    # Move backward (-Y)
            new_pos = pos + np.array([0, -step_size, 0])
        elif action == 2:    # Move left (-X)
            new_pos = pos + np.array([-step_size, 0, 0])
        elif action == 3:    # Move right (+X)
            new_pos = pos + np.array([step_size, 0, 0])
        elif action == 4:    # Stop
            new_pos = pos

        time.sleep(self.dt * 3)  # Multiply dt by 3 to slow it down more


        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), final_orientation)

        # Get LiDAR readings
        lidar_near, lidar_far = self.get_lidar_readings()

        # Observation includes position and LiDAR data
        observation = np.concatenate((new_pos, [lidar_near, lidar_far]))
        reward = -lidar_near  # Penalize being too close to obstacles
        done = False

        return observation, reward, done, {}

    def reset(self):
        self.current_swing_deg = 0
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, initial_orientation)
        lidar_near, lidar_far = self.get_lidar_readings()
        return np.concatenate((np.array(self.cane_start_pos, dtype=np.float32), [lidar_near, lidar_far]))

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

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = CaneEnv()
    env.reset()

    try:
        while True:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            print(f"Observation: {obs}, Reward: {reward}")
            time.sleep(0.6)
    except KeyboardInterrupt:
        env.close()

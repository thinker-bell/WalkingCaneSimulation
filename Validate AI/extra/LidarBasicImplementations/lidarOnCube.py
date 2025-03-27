import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from gymnasium import spaces

class CubeSwingEnv(gym.Env):
    def __init__(self):
        super(CubeSwingEnv, self).__init__()
        
        # Connect to PyBullet in GUI mode and set up simulation.
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        # Load a plane for reference.
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Cube properties.
        self.cube_size = 0.2      # Side length of the cube.
        self.cube_mass = 1.0       
        
        # Fixed baseline: 45° tilt about the X-axis and 0° pitch.
        self.baseline_roll = math.radians(0)
        self.baseline_pitch = 0
        self.current_swing_deg = 0  # Yaw (swing) in degrees.
        
        # Compute a vertical offset so that when the cube is tilted,
        # its bottom nearly touches the ground.
        # For a cube, the distance from its center to the top face is half the side length.
        # After a 45° rotation, the vertical offset becomes:
        vertical_offset = (self.cube_size / 2) * math.cos(math.radians(45))
        self.cube_start_pos = [0, 0, vertical_offset]
        
        # Compute the initial orientation using Euler angles: [roll, pitch, yaw]
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )
        
        # Create the cube as a pink box.
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.cube_size/2]*3)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.cube_size/2]*3,
                                           rgbaColor=[1, 0.75, 0.8, 1])  # Pink color.
        self.cube_id = p.createMultiBody(baseMass=self.cube_mass,
                                         baseCollisionShapeIndex=collision_shape,
                                         baseVisualShapeIndex=visual_shape,
                                         basePosition=self.cube_start_pos,
                                         baseOrientation=initial_orientation)
        
        # Define the original action space (movement only):
        # 0: Move forward (+Y)
        # 1: Move backward (-Y)
        # 2: Move left (-X)
        # 3: Move right (+X)
        # 4: Stop (no movement)
        self.action_space = spaces.Discrete(5)
        
        # Observation space: we'll return the cube's center position.
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0]),
            high=np.array([10, 10, 10]),
            dtype=np.float32
        )
        
        # Simulation time step for the swing cycle.
        self.dt = 1.0 / 240.0
        
        # Define the lidar properties.
        self.num_rays = 30
        self.ray_length = 5
        self.scan_angle = math.pi / 4  # Scan angle of 45 degrees in front of the robot
        

    def swing_cycle(self):
        """
        Perform a full swing cycle covering a total of 160° before taking a step.
        The cube swings from 0° to +80°, then to –80°, and finally returns to 0°.
        """
        # Create swing angle sequences.
        swing_up = np.linspace(0, 80, num=10)
        swing_down = np.linspace(80, -80, num=20)
        swing_return = np.linspace(-80, 0, num=10)
        full_cycle = np.concatenate((swing_up, swing_down, swing_return))
        
        # Get the current position (keep it constant during the swing).
        pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        pos = np.array(pos)
        
        for angle in full_cycle:
            self.current_swing_deg = angle
            new_orientation = p.getQuaternionFromEuler(
                [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
            )
            p.resetBasePositionAndOrientation(self.cube_id, pos.tolist(), new_orientation)
            
            # Perform lidar scan.
            self.perform_lidar_scan()
            
            p.stepSimulation()
            time.sleep(self.dt)
        
        # After the cycle, reset the swing angle to 0.
        self.current_swing_deg = 0
        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )
        p.resetBasePositionAndOrientation(self.cube_id, pos.tolist(), final_orientation)
        
    def perform_lidar_scan(self):
        """
        Perform a lidar scan in front of the cube.
        """
        # Define the lidar properties.
        num_rays = self.num_rays
        ray_length = self.ray_length
        scan_angle = self.scan_angle
        
        # Update lidar ray origins and destinations.
        ray_froms = []
        ray_tos = []
        base_pos, base_orient = p.getBasePositionAndOrientation(self.cube_id)
        yaw = p.getEulerFromQuaternion(base_orient)[2]  # Get the cube's yaw angle.
        for j in range(num_rays):
            angle = -scan_angle / 2 + j * scan_angle / (num_rays - 1)
            rotated_vector = [ray_length * math.cos(yaw + angle), ray_length * math.sin(yaw + angle), 0]
            ray_from = base_pos
            ray_to = [base_pos[0] + rotated_vector[0], base_pos[1] + rotated_vector[1], base_pos[2]]
            ray_froms.append(ray_from)
            ray_tos.append(ray_to)
        
        # Perform ray casting.
        ray_results = p.rayTestBatch(ray_froms, ray_tos)
        
        # Visualize the lidar scan.
        for j, ray_result in enumerate(ray_results):
            if ray_result[0] != -1:
                p.addUserDebugLine(ray_froms[j], ray_result[3], [1, 0, 0], 1)  # Red for hits.
            else:
                p.addUserDebugLine(ray_froms[j], ray_tos[j], [0, 1, 0], 1)  # Green for misses.

    def step(self, action):
        # First, run a full swing cycle (160° total swing) before moving.
        self.swing_cycle()
        
        # Now, update the cube's position based on the movement action.
        pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        pos = np.array(pos)
        step_size = 0.1
        
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
        
        # Update the cube's position; reset orientation to baseline (swing = 0).
        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cube_id, new_pos.tolist(), final_orientation)
        
        # For observation, we return the new position.
        reward = new_pos[1]  # For example, reward forward progress.
        done = False
        return new_pos, reward, done, {}

    def reset(self):
        self.current_swing_deg = 0
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cube_id, self.cube_start_pos, initial_orientation)
        return np.array(self.cube_start_pos, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = CubeSwingEnv()
    env.reset()
    
    try:
        while True:
            # For testing, sample a random action from the movement action space.
            action = env.action_space.sample()
            env.step(action)
            time.sleep(0.3)
    except KeyboardInterrupt:
        env.close()
    
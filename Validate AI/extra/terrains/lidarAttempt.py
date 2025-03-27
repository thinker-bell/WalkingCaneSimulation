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
        self.cane_radius = 0.05     # Radius of the cane (cylinder)
        self.cane_height = 2.0      # Cane length
        self.cane_mass = 1.0        # Cane mass
        
        # Fixed baseline: 45° tilt about the X-axis and 0° pitch.
        self.baseline_roll = math.radians(45)
        self.baseline_pitch = 0
        
        # We'll let the cane have an additional yaw (swing) that is updated during the swing cycle.
        self.current_swing_deg = 0  # in degrees
        
        # Compute vertical offset so that the bottom nearly touches the ground.
        vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))
        self.cane_start_pos = [0, 0, vertical_offset]
        
        # Initial orientation: baseline roll and zero yaw.
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )
        
        # Create the cane as a pink cylinder.
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                 radius=self.cane_radius,
                                                 height=self.cane_height)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                           radius=self.cane_radius,
                                           length=self.cane_height,
                                           rgbaColor=[1, 0.75, 0.8, 1])  # Pink color.
        self.cane_id = p.createMultiBody(baseMass=self.cane_mass,
                                         baseCollisionShapeIndex=collision_shape,
                                         baseVisualShapeIndex=visual_shape,
                                         basePosition=self.cane_start_pos,
                                         baseOrientation=initial_orientation)
        
        # Define the original action space (movement only).
        # 0: Move forward (+Y)
        # 1: Move backward (-Y)
        # 2: Move left (-X)
        # 3: Move right (+X)
        # 4: Stop (no movement)
        self.action_space = spaces.Discrete(5)
        
        # Observation space: we'll return the cane’s center position and the straight line sensor data.
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0, 0]),  
            high=np.array([10, 10, 10, 10]),  
            dtype=np.float32
        )
        
        # Simulation time step for the swing cycle.
        self.dt = 1.0 / 240.0
        self.line_ids = {}
    '''
    def get_straight_line_data(self):
        # Get the straight line sensor data
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)

        ray_from = pos + np.array([0, 0, -self.cane_height / 2])
        ray_to_down = ray_from + np.array([0, 0, -1])
        ray_to_45 = ray_from + np.array([math.cos(math.radians(45)), math.sin(math.radians(45)), 0])

        result_down = p.rayTest(ray_from, ray_to_down)
        result_45 = p.rayTest(ray_from, ray_to_45)

        if result_down[0] != -1 and len(result_down) >= 3:
            straight_line_data_down = result_down[2]
        else:
            straight_line_data_down = 10  # maximum distance

        if result_45[0] != -1 and len(result_45) >= 3:
            straight_line_data_45 = result_45[2]
        else:
            straight_line_data_45 = 10  # maximum distance

        return straight_line_data_down, straight_line_data_45
    '''
    def swing_cycle(self):
        """
        Perform a full swing cycle (covering a total of 160°) before taking a step.
        We'll swing from 0° to +80°, then from +80° to –80°, and finally back to 0°.
        """
        # Create a sequence of swing angles.
        # For example, 10 steps from 0 to 80, 20 steps from

              # For example, 10 steps from 0 to 80, 20 steps from 80 to -80, and 10 steps from -80 to 0.
        swing_up = np.linspace(0, 80, num=10)
        swing_down = np.linspace(80, -80, num=20)
        swing_return = np.linspace(-80, 0, num=10)
        full_cycle = np.concatenate((swing_up, swing_down, swing_return))

        # Get the current position (we keep it constant during the swing).
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

        # After the cycle, reset swing angle to 0.
        self.current_swing_deg = 0
        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, pos.tolist(), final_orientation)
    
    def get_straight_line_data(self):
        # Get the straight line sensor data
        pos, orn = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        orn = np.array(orn)

        # Calculate direction of lidar
        yaw_angle = p.getEulerFromQuaternion(orn)[2]
        forward_vector = np.array([math.cos(yaw_angle), math.sin(yaw_angle), -math.sin(math.radians(45))])

        # Lidar position and direction
        #ray_from = pos + np.array([0, 0, -self.cane_height / 4 + 0.5])
        ray_from = pos + np.array([0, 0, -self.cane_height / 2 + 0.5])
        ray_to = ray_from + forward_vector

        # Draw lidar line
        if 'lidar' in self.line_ids:
            p.removeUserDebugItem(self.line_ids['lidar'])
        self.line_ids['lidar'] = p.addUserDebugLine(ray_from, ray_to, [0, 1, 0], 0.1)  # Green line

        # Return dummy value for now
        return 1.0, 1.0
    

    
    

    
        '''
        if result_down[0] != -1 and len(result_down) >= 3:
            straight_line_data_down = result_down[2]
            hit_point_down = ray_from + straight_line_data_down * (ray_to_down - ray_from)
            p.addUserDebugLine(ray_from, hit_point_down, [1, 0, 0], 2)  # Red line
        else:
            straight_line_data_down = 10  # maximum distance

        if result_45[0] != -1 and len(result_45) >= 3:
            straight_line_data_45 = result_45[2]
            hit_point_45 = ray_from + straight_line_data_45 * (ray_to_45 - ray_from)
            p.addUserDebugLine(ray_from, hit_point_45, [0, 1, 0], 2)  # Green line
        else:
            straight_line_data_45 = 10  # maximum distance

        return straight_line_data_down, straight_line_data_45
        '''


    def step(self, action):
        # First, run a full swing cycle (160° total swing) before moving.
        self.swing_cycle()

        # Now, update the cane's position based on the original movement action.
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

        # Update the cane's position (orientation reset to baseline with 0 swing).
        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), final_orientation)

        # Get the straight line sensor data
        straight_line_data_down, straight_line_data_45 = self.get_straight_line_data()

        # For observation, we return the cane’s center position and the straight line sensor data.
        observation = np.concatenate((new_pos, [straight_line_data_down, straight_line_data_45]))
        reward = new_pos[1]  # For example, reward based on forward progress.
        done = False
        return observation, reward, done, {}

    def reset(self):
        self.current_swing_deg = 0
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, initial_orientation)
        return np.concatenate((self.cane_start_pos, [10, 10]))  # straight line sensor data initialized to maximum distance

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = CaneEnv()
    observation = env.reset()

    try:
        while True:
            # For testing, use the original action space (movement only).
            # For instance, randomly choose an action.
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            print(observation)
            time.sleep(0.6)
            if done:
                observation = env.reset()
    except KeyboardInterrupt:
        env.close()
          
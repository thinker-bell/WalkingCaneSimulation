''' 
This implementation considers swinging before stepping
Problem: the cane swings from a middle axis, for 
Current working implementation of cane swining
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
        
        # Connect to PyBullet in GUI mode and set up simulation.
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        # Load a plane for reference.
        self.plane_id = p.loadURDF("plane.urdf")
        

        self.cane_radius = 0.05     # Radius of the cane (cylinder)
        self.cane_height = 2.0      # Cane length
        self.cane_mass = 1.0        # Cane mass
        
        # Fixed baseline: 45° tilt about the X-axis and 0° pitch.
        self.baseline_roll = math.radians(45)
        self.baseline_pitch = 0
        
        
        self.current_swing_deg = 0  # in degrees
        
        # Compute vertical offset so that the bottom nearly touches the ground.
        vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))
        self.cane_start_pos = [0, 0, vertical_offset]
        
        # Initial orientation
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )
        
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
        
        """# Lines to add create the LiDAR sensor: 
        lidar_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        lidar_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 0, 0, 1])
        self.lidar_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=lidar_collision_shape, baseVisualShapeIndex=lidar_visual_shape)
        
        # Attach the LiDAR to the cane

        #self.lidar_start_pos = [0, 0, self.cane_height/2]
        
        self.lidar_start_pos, self.lidar_start_orientation = p.getBasePositionAndOrientation(self.lidar_id)
        lidar_start_pos = [0, 0, self.cane_height/2]
        lidar_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.lidar_id, lidar_start_pos, lidar_start_orientation)
        constraint_id = p.createConstraint(self.cane_id, -1, self.lidar_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])"""

        # Create the LiDAR cube
        lidar_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        lidar_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 0, 0, 1])
        self.lidar_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=lidar_collision_shape, baseVisualShapeIndex=lidar_visual_shape)

        # Create a fixed constraint between the cane and the LiDAR cube
        constraint_id = p.createConstraint(self.cane_id, -1, self.lidar_id, -1, p.JOINT_FIXED, [0, 0, -self.cane_height/2], [0, 0, 0], [0, 0, 0])

        # Set the initial position of the LiDAR cube
        lidar_start_pos = [0, 0, -self.cane_height/2]
        lidar_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.lidar_id, lidar_start_pos, lidar_start_orientation)

        self.observation_space = spaces.Box(
        low=np.array([-10, -10, 0, 0]),  # closest obstacle, clear path width
        high=np.array([10, 10, 10, 10]),
        dtype=np.float32
        )


        # Define the original action space (movement only).
        # 0: Move forward (+Y)
        # 1: Move backward (-Y)
        # 2: Move left (-X)
        # 3: Move right (+X)
        # 4: Stop (no movement)
        self.action_space = spaces.Discrete(5)
        
        # Observation space: we'll return the cane’s center position.
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0]),
            high=np.array([10, 10, 10]),
            dtype=np.float32
        )
        
        # Simulation time step for the swing cycle.
        self.dt = 1.0 / 240.0
        
        
        
    def swing_cycle(self):
        """
        Perform a full swing cycle (covering a total of 160°) before taking a step.
        We'll swing from 0° to +80°, then from +80° to –80°, and finally back to 0°.
        """
        # Create a sequence of swing angles.
        # For example, 10 steps from 0 to 80, 20 steps from 80 to -80, and 10 steps from -80 to 0.
        swing_up = np.linspace(0, 80, num=50)
        swing_down = np.linspace(80, -80, num=100)
        swing_return = np.linspace(-80, 0, num=50)
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
            
            # Get LiDAR data
            closest_obstacle, _ = self.get_lidar_data()
            print(f"LiDAR Distance: {closest_obstacle:.2f} meters")
            
            p.stepSimulation()
            time.sleep(self.dt)


        # After the cycle, reset swing angle to 0.
        self.current_swing_deg = 0
        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)]
        )
        cane_pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        lidar_pos = [cane_pos[0], cane_pos[1], cane_pos[2] + self.cane_height/2]
        
    # get the LiDAR data so that I can eventually use it or save it for processing.    
    
    def get_lidar_data(self):
        # Remove previous beam
        if hasattr(self, 'beam_id'):
            p.removeUserDebugItem(self.beam_id)

        # Get the current position and orientation of the LiDAR cube
        lidar_pos, lidar_orientation = p.getBasePositionAndOrientation(self.lidar_id)
        lidar_roll, lidar_pitch, lidar_yaw = p.getEulerFromQuaternion(lidar_orientation)

        # Calculate the direction of the LiDAR cube
        direction = [np.cos(lidar_yaw), np.sin(lidar_yaw), 0]

        # Calculate the end point of the beam
        step_size = 0.3
        num_steps = 2
        beam_end = [lidar_pos[0] + num_steps * step_size * direction[0],
                    lidar_pos[1] + num_steps * step_size * direction[1],
                    lidar_pos[2] + num_steps * step_size * direction[2]]

        # Visualize the beam
        self.beam_id = p.addUserDebugLine(lidar_pos, beam_end, [1, 0, 0], 2, 0.1)

        # Perform ray casting and collision detection
        result = p.rayTest(lidar_pos, beam_end)

        # Check if the ray hit an object
        if result[0] != -1:  # if collision detected
            if len(result) > 2:  # Check if result has at least 3 elements
                closest_obstacle = result[2]  # record distance
            else:
                closest_obstacle = num_steps * step_size  # record maximum distance if no collision
        else:
            closest_obstacle = num_steps * step_size  # record maximum distance if no collision

        return closest_obstacle, None
        

    def step(self, action):
        # First, run a full swing cycle (160° total swing) before moving.
        self.swing_cycle()

        
        # Create a fixed constraint between the cane and the LiDAR cube
        constraint_id = p.createConstraint(self.cane_id, -1, self.lidar_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, self.cane_height/2])
        
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

        # Calculate the new position of the LiDAR cube
        cane_pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        lidar_pos = [cane_pos[0], cane_pos[1], cane_pos[2] + self.cane_height/2]
        lidar_orientation = p.getQuaternionFromEuler([0, 0, math.radians(0)])

        #lidar_pos = [new_pos[0], new_pos[1], new_pos[2] + self.cane_height/2] 
        closest_obstacle, clear_path_width = self.get_lidar_data()
        observation = np.array([pos[0], pos[1], closest_obstacle, clear_path_width])

        # For observation, we return the cane's new center position.
        reward = new_pos[1]  # For example, reward based on forward progress.
        #print(reward)
        done = False
        return new_pos, reward, done, {}
    
    def reset(self):
        self.current_swing_deg = 0
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, initial_orientation)
        return np.array(self.cane_start_pos, dtype=np.float32)
    
    def render(self, mode="human"):
        pass
    
    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = CaneEnv()
    env.reset()
    
    try:
        while True:
            # Testing if my github works
            # For testing, use the original action space (movement only).
            # For instance, randomly choose an action.
            #print(env.get_lidar_data())
            action = env.action_space.sample()
            env.step(action)
            time.sleep(0.6)
    except KeyboardInterrupt:
        env.close()

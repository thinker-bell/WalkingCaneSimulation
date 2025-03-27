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
        self.cane_start_pos = [0, 0, vertical_offset + 0.75]
        
        # Initial orientation: baseline roll and zero yaw.
        # Initial orientation: baseline roll, zero pitch, and 90 degrees yaw (pointing forward).
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(90)]
        )
        # Create the cane as a pink cylinder.
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                             radius=self.cane_radius,
                                             height=self.cane_height)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                            radius=self.cane_radius,
                                            length=self.cane_height,
                                            rgbaColor=[1, 0.75, 0.8, 1])  # Pink color.

        # Shift the center of mass upward to the top 1/8
        com_height = self.cane_height - (self.cane_height / 8)
        inertial_pos = [0, 0, com_height / 2]

        # Adjust the base position to move the cane up
        base_pos = [0, 0, self.cane_height / 2 + 0.1]  # Add 0.1 to move the cane above the floor

        # Create the cane multi-body with the shifted center of mass
        self.cane_id = p.createMultiBody(baseMass=self.cane_mass,
                                            baseCollisionShapeIndex=collision_shape,
                                            baseVisualShapeIndex=visual_shape,
                                            basePosition=base_pos,
                                            baseOrientation=initial_orientation,
                                            baseInertialFramePosition=inertial_pos)
        
         
        self.lidar_start_pos = [0, 0, self.cane_height / 8]


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
    '''
    def get_lidar_data(self):
        # Remove previous beam
        if hasattr(self, 'beam_id'):
            p.removeUserDebugItem(self.beam_id)
        
        # Get the current position and orientation of the cane
        cane_pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)
        
        # Calculate the position of the LiDAR beam (move down the cane shaft)
        lidar_offset = [0, -self.cane_radius, -self.cane_height / 4]  # Move down 1/4 of the cane height and offset from the center
        rotated_offset = [
            lidar_offset[0] * np.cos(cane_pitch) - lidar_offset[2] * np.sin(cane_pitch),
            lidar_offset[1],
            lidar_offset[0] * np.sin(cane_pitch) + lidar_offset[2] * np.cos(cane_pitch)
        ]
        lidar_pos = [
            cane_pos[0] + rotated_offset[0] * np.cos(cane_yaw) - rotated_offset[1] * np.sin(cane_yaw),
            cane_pos[1] + rotated_offset[0] * np.sin(cane_yaw) + rotated_offset[1] * np.cos(cane_yaw),
            cane_pos[2] + rotated_offset[2]
        ]
        
        # Calculate the direction of the beam (rotate 90 degrees to the left)
        direction = [-np.sin(cane_yaw), np.cos(cane_yaw), -1]
        
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
        
    '''
    '''
    def get_lidar_data(self):
        # Remove previous beam
        if hasattr(self, 'beam_id'):
            p.removeUserDebugItem(self.beam_id)
        
        # Get the current position and orientation of the cane
        cane_pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)
        
        
        # LiDAR should be positioned at the lower 1/8th of the cane
        lidar_offset = [0, 0, -self.cane_height / 8]  

        # Rotate the offset to match the cane's orientation
        rotated_offset = [
            lidar_offset[0] * np.cos(cane_pitch) - lidar_offset[2] * np.sin(cane_pitch),
            lidar_offset[1],
            lidar_offset[0] * np.sin(cane_pitch) + lidar_offset[2] * np.cos(cane_pitch)
        ]

        # Compute the LiDAR world position
        lidar_pos = [
            cane_pos[0] + rotated_offset[0] * np.cos(cane_yaw) - rotated_offset[1] * np.sin(cane_yaw),
            cane_pos[1] + rotated_offset[0] * np.sin(cane_yaw) + rotated_offset[1] * np.cos(cane_yaw),
            cane_pos[2] + rotated_offset[2]
        ]

        # Make sure the LiDAR beam points straight forward (+Y direction in the cane's local frame)
        direction = [np.cos(cane_yaw), np.sin(cane_yaw), 0]  # Only in the X-Y plane

        # Calculate the end point of the beam
        step_size = 0.5  # Increase step size if needed
        num_steps = 4
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
    '''
    def get_lidar_data(self):
        # Remove previous beam
        if hasattr(self, 'beam_id'):
            p.removeUserDebugItem(self.beam_id)
        
        # Get the current position and orientation of the cane
        cane_pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)

        # Correct LiDAR offset (1/8th from the bottom)
        lidar_offset_z = -1.5 #- (self.cane_height / 2) + (self.cane_height / 100)  # Position from the bottom

        # Adjust for cane’s rotation to ensure LiDAR remains properly aligned
        lidar_offset = [0, 0, lidar_offset_z]
        rotated_offset = p.rotateVector(cane_orientation, lidar_offset)

        # Final LiDAR position relative to the cane
        lidar_pos = [
            cane_pos[0] + rotated_offset[0],
            cane_pos[1] + rotated_offset[1],
            cane_pos[2] + rotated_offset[2]
        ]

        # LiDAR beam direction: should be straight ahead, regardless of cane tilt
        beam_direction = [
            # Forward direction
            -math.sin(cane_yaw),  
            math.cos(cane_yaw),# Side direction
            -math.sin(45)                    # Keep level with the ground
        ]

        # Compute end point of the LiDAR beam
        step_size = 0.3
        num_steps = 1.2
        beam_end = [
            lidar_pos[0] + num_steps * step_size * beam_direction[0],
            lidar_pos[1] + num_steps * step_size * beam_direction[1],
            lidar_pos[2] + num_steps * step_size * beam_direction[2]
        ]

        #p.addUserDebugLine(lidar_pos, beam_end, [1, 0, 0], 2, 0.1)

        # Draw the beam in PyBullet
        self.beam_id = p.addUserDebugLine(lidar_pos, beam_end, [1, 0, 0], 2, 0.1)

        # Perform ray casting for object detection
        result = p.rayTest(lidar_pos, beam_end)

        # Check for obstacles
        if result[0] != -1:  # if collision detected
            closest_obstacle = result[2] if len(result) > 2 else num_steps * step_size
        else:
            closest_obstacle = num_steps * step_size  # Max range if no collision

        return closest_obstacle, None


    def swing_cycle(self):
        """
        Perform a full swing cycle (covering a total of 160°) before taking a step.
        We'll swing from 0° to +80°, then from +80° to –80°, and finally back to 0°.
        """
        # Create a sequence of swing angles.
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
                [self.baseline_roll, self.baseline_pitch, math.radians(90 + self.current_swing_deg)]
            )
            p.resetBasePositionAndOrientation(self.cane_id, pos.tolist(), new_orientation)
            
            # Get LiDAR data during the swing
            closest_obstacle, _ = self.get_lidar_data()
            print(f"LiDAR Distance at {angle}°: {closest_obstacle:.2f} meters")
            
            p.stepSimulation()
            time.sleep(self.dt)
        
        # After the cycle, reset swing angle to 0.
        self.current_swing_deg = 0
        final_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(90)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, pos.tolist(), final_orientation)
        
    def step(self, action):
        # First, run a full swing cycle (160° total swing) before moving.
        self.swing_cycle()
        
        # Now, update the cane's position based on the original movement action.
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        step_size = 0.3
        
        if action == 0:      # Move forward (+Y)
            print("forward")
            new_pos = pos + np.array([0, step_size, 0])
        elif action == 1:    # Move backward (-Y)
            print("backward")
            new_pos = pos + np.array([0, -step_size, 0])
        elif action == 2:    # Move left (-X)
            print("left")
            new_pos = pos + np.array([-step_size, 0, 0])
        elif action == 3:    # Move right (+X)
            print("right")
            new_pos = pos + np.array([step_size, 0, 0])
        elif action == 4:    # Stop
            print("no move")
            new_pos = pos
        
        # Update the cane's position (orientation reset to baseline with 0 swing).
        # Update the cane's position (orientation reset to baseline with 90-degree yaw offset).
        final_orientation = p.getQuaternionFromEuler(
        [self.baseline_roll, self.baseline_pitch, math.radians(90)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), final_orientation)
        
        # For observation, we return the cane's new center position.
        reward = new_pos[1]  # For example, reward based on forward progress.
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
            action = env.action_space.sample()
            env.step(action)
            time.sleep(0.6)
    except KeyboardInterrupt:
        env.close()

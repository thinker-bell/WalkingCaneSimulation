''' 
This implementation considers swinging before stepping
Problem: the cane swings from a middle axis, for 
Current working implementation of cane swining
'''

#### TEST ####


import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import os
from gymnasium import spaces
from stable_baselines3 import PPO




class CaneEnv(gym.Env):
    MAX_TIMESTEPS = 100  # Set a max limit for each episode
    def __init__(self):
        super(CaneEnv, self).__init__()
        
        # Connect to PyBullet in GUI mode and set up simulation.
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        # Load a plane for reference.
        #self.plane_id = p.loadURDF("custom_plane.urdf")
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Cane properties.
        self.cane_radius = 0.025     # Radius of the cane (cylinder)
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
        self.cane_id = 1  # Add this line
         
        self.lidar_start_pos = [0, 0, self.cane_height / 8]

        ############################ GOAL LOCATION ################################
        self.goal_location = np.array([-2.0, 2.0, 1.4])
        self.goal_visual_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.1]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.1],
                rgbaColor=[0, 1, 0, 1]  # Green color
            ),
            basePosition=self.goal_location
        )

        # Define the original action space (movement only).
        # 0: Move forward (+Y)
        # 1: Move backward (-Y)
        # 2: Move left (-X)
        # 3: Move right (+X)
        # 4: Stop (no movement)
        self.action_space = spaces.Discrete(5)
        
        # Simulation time step for the swing cycle.
        self.dt = 1.0 / 240.0
        
        ########### NON FUNCTIONING ##########
        # 19 lidar angles * 2 beams = 38 + 6 (cane_x, cane_y, cane_z, cane_yaw, yaw, distance_to_goal) = 44
        #low_obs = np.array([0.0] * 38 + [-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, 0.0], dtype=np.float32)
        #high_obs = np.array([10.0] * 38 + [np.inf, np.inf, np.inf, np.pi, np.pi, 20.0], dtype=np.float32)

        #self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
       
        # Ensure the 'low' and 'high' observations have a consistent length and data type.
        low_obs = np.array([0.0] * 79 + [-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, 0.0], dtype=np.float32)
        high_obs = np.array([10.0] * 79 + [np.inf, np.inf, np.inf, np.pi, np.pi, 20.0], dtype=np.float32)

        # Create the observation space using Box
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)



        # ----------------- OBSTACLE 1 -----------------
        self.obstacle_location = np.array([-2.5, 1.0, 0.5])  # Adjust height to be half the box height
        self.obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.3, 0.3, 0.5]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.3, 0.3, 0.5],
                rgbaColor=[0.7, 0.2, 0.2, 1]  # Reddish color
            ),
            basePosition=self.obstacle_location
        )
        
        # ----------------- OBSTACLE 2 -----------------
        self.obstacle_location = np.array([1.5, 2.0, 0.5])  # Adjust height to be half the box height
        self.obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                halfExtents=[0.3, 0.3, 0.5]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                halfExtents=[0.3, 0.3, 0.5],
                rgbaColor=[0.7, 0.2, 0.2, 3]  # Reddish color
            ),
            basePosition=self.obstacle_location
        )
        
    def get_observation_with_swing(self):
        # Initial position and orientation of the cane
        pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)

        # Define the swing angles for the full cycle
        swing_up = np.linspace(0, 80, num=10)
        swing_down = np.linspace(80, -80, num=20)
        swing_return = np.linspace(-80, 0, num=10)
        full_cycle = np.concatenate((swing_up, swing_down, swing_return))

        # Goal location (assumed to be provided somewhere in the class)
        goal_location = self.goal_location

        # Observation list to accumulate the data
        obs_list = []

        # Loop through the swing cycle and gather data at each angle
        for angle in full_cycle:
            #print(angle)
            self.current_swing_deg = angle

            # Set new orientation based on the swing angle
            new_orientation = p.getQuaternionFromEuler(
                [self.baseline_roll, self.baseline_pitch, cane_yaw + math.radians(self.current_swing_deg)]
            )
            
            # Update cane's position and orientation
            p.resetBasePositionAndOrientation(self.cane_id, pos, new_orientation)

            # Get LiDAR data during the swing
            primary_lidar, secondary_lidar = self.get_lidar_data()

            # Handle None lidar readings (e.g., if no obstacle detected)
            if primary_lidar is None:
                primary_lidar = 3.6  # or whatever the maximum lidar range is
            if secondary_lidar is None:
                secondary_lidar = 3.6
            
            # idk if I should add the norm angle
            #obs_list.append(primary_lidar)
            #obs_list.append(secondary_lidar)
            obs_list.append([primary_lidar,secondary_lidar])


        '''
        cane_x, cane_y, cane_z = pos  # Extract the z-coordinate
        distance_to_goal = np.linalg.norm(np.array(self.goal_location) - np.array([cane_x, cane_y, cane_z]))  # 3D distance

        obs_list.extend([
            cane_yaw,  # Using cane_yaw directly as the heading
            cane_x, cane_y, cane_z,  # Including z-coordinate of the cane
            cane_yaw,  # yaw as a duplicate or you can choose another metric
            distance_to_goal
        ])

        #print(np.array(obs_list, dtype=np.float32))
        '''


        # Distance to goal
        #distance_to_goal = np.linalg.norm(np.array(goal_location) - np.array(pos))
        # Append the observation for this cycle step
        #obs_list.append([distance_to_goal])

        # Check for collisions during the swing
        contacts = p.getContactPoints(bodyA=self.cane_id)
        for contact in contacts:
            # Only report if it's meaningful
            if contact[8] < 0.01:  # Close contact distance
                print("Cane hits obstacle")
            p.changeVisualShape(self.cane_id, -1, rgbaColor=[1, 0, 0, 1])  # Red color
        p.changeVisualShape(self.cane_id, -1, rgbaColor=[0, 1, 0, 1])  # Green color   

        # Step the simulation forward
        p.stepSimulation()
        time.sleep(self.dt)  # Assuming self.dt is the time step for the simulation

        # Extract the cane's position (x, y, z)
        cane_x, cane_y, cane_z = pos  

        # Calculate distance to goal (3D distance)
        distance_to_goal = np.linalg.norm(np.array(self.goal_location) - np.array([cane_x, cane_y, cane_z]))

        # Add the cane's position, orientation (yaw), and distance to goal to the observation list
        obs_list.append([cane_yaw, cane_x, cane_y, cane_z, distance_to_goal])

        #print(obs_list)

        obs_array = np.concatenate([np.array(obs) for obs in obs_list])


        # Flatten the collected observations into a single 1D array
        #obs_array = np.array(obs_list).flatten()

        #obs_array = np.array(obs_list, dtype=np.float64)

        print(obs_array)

        return obs_array

   


    def get_lidar_data(self):
        # Remove previous beam
        if hasattr(self, 'beam_id_primary'):
            p.removeUserDebugItem(self.beam_id)
        if hasattr(self, 'beam_id_secondary'):
            p.removeUserDebugItem(self.beam_id_secondary)    
        
        # Get the current position and orientation of the cane
        cane_pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)

        # Primary Lidar 
        lidar_offset_z = -1.5 #- (self.cane_height / 2) + (self.cane_height / 100)  # Position from the bottom
        lidar_offset = [0, 0, lidar_offset_z]
        rotated_offset = p.rotateVector(cane_orientation, lidar_offset)

        # Primary lidar pos
        lidar_pos = [
            cane_pos[0] + rotated_offset[0],
            cane_pos[1] + rotated_offset[1],
            cane_pos[2] + rotated_offset[2]
        ]

        # primary LiDAR beam direction: 
        beam_direction = [
            # Forward direction
            -math.sin(cane_yaw),  
            math.cos(cane_yaw),# Side direction
            -math.sin(45)                    # Keep level with the ground
        ]

        # Compute end point of the primary LiDAR beam
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

        ### SECONDARY LIDAR #########
        # Secondary LiDAR position with an offset of -0.5 along the X-axis
        lidar_offset_y = -0.7
        secondary_lidar_offset = [0, 0, lidar_offset_y]  # -0.5 along X and same Z offset
        rotated_secondary_offset = p.rotateVector(cane_orientation, secondary_lidar_offset)

        # Secondary LiDAR position
        secondary_lidar_pos = [
            cane_pos[0] + rotated_secondary_offset[0],
            cane_pos[1] + rotated_secondary_offset[1],
            cane_pos[2] + rotated_secondary_offset[2]
        ]

        secondary_beam_direction = [-math.sin(cane_yaw), math.cos(cane_yaw), 0]  # Same beam direction as primary

        # Compute the end point of the secondary LiDAR beam
        step_size = 0.3
        num_steps = 6
        beam_end_secondary = [
            secondary_lidar_pos[0] + num_steps * step_size * secondary_beam_direction[0],
            secondary_lidar_pos[1] + num_steps * step_size * secondary_beam_direction[1],
            secondary_lidar_pos[2] + num_steps * step_size * secondary_beam_direction[2]
        ]


        self.beam_id_secondary = p.addUserDebugLine(secondary_lidar_pos, beam_end_secondary, [1, 0, 0], 2, 0.1)

        # Perform ray casting for object detection
        result_primary = p.rayTest(lidar_pos, beam_end)
        result_secondary = p.rayTest(secondary_lidar_pos, beam_end_secondary)
        #print("prim ",result_primary)
        #print("sec ",result_secondary)


        # Primary LiDAR
        if result_primary[0][0] == -1:
            lidar1_value = 0.0  # No hit
        else:
            hit_position = result_primary[0][3]
            lidar1_value = math.dist(lidar_pos, hit_position)

        # Secondary LiDAR
        if result_secondary[0][0] == -1 or result_secondary[0][0] == self.cane_id:
            lidar2_value = 0.0  # No hit or hit own cane
        else:
            hit_position = result_secondary[0][3]
            lidar2_value = math.dist(secondary_lidar_pos, hit_position)

        #print("Prim",lidar1_value)
        #print("Sec",lidar2_value)

        return lidar1_value, lidar2_value


    
    def step(self, action):

        # First, run a full swing cycle (160° total swing) before moving.
        #self.swing_cycle()
        observation = self.get_observation_with_swing()

        # Now, update the cane's position based on the original movement action.
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        step_size = 0.3

        # Define the rotation angles
        rotation_angles = {
            2: math.radians(30),  # short left
            3: math.radians(60),  # medium left
            4: math.radians(90),  # hard left
            6: math.radians(-30),  # short right
            7: math.radians(-60),  # medium right
            8: math.radians(-90),  # hard right
            9: math.radians(180)  # 180 deg turn around
        }

        collision_detected = False  # Initialize collision_detected variable

        if action == 0:  # Take 1 step forward
            print("\n\nforward\n\n")
            _, orientation = p.getBasePositionAndOrientation(self.cane_id)
            roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
            
            # Calculate new position based on current orientation
            step_size = 0.3
            new_pos = pos + np.array([-step_size * math.sin(yaw), step_size * math.cos(yaw), 0])
            #print(new_pos)
            
            # Check for collisions at the new position
            temp_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
            p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), temp_orientation)
            contacts = p.getContactPoints(bodyA=self.cane_id)
            for contact in contacts:
                # Only report if it's meaningful
                if contact[8] < 0.01:  # Close contact distance
                    collision_detected = True
                    break
            
            # If a collision is detected, don't move the cane
            if collision_detected:
                #p.changeVisualShape(self.cane_id, -1, rgbaColor=[1, 0, 0, 1])  # Red color
                p.resetBasePositionAndOrientation(self.cane_id, pos, orientation)
                print("Collision detected, cannot move through obstacle")
            else:
                #p.changeVisualShape(self.cane_id, -1, rgbaColor=[0, 1, 0, 1])  # Green color
                _, new_orientation = p.getBasePositionAndOrientation(self.cane_id)

        elif action == 1:  # Stop
            print("\n\nno move\n\n")
            new_pos = pos
            _, new_orientation = p.getBasePositionAndOrientation(self.cane_id)
        elif action in rotation_angles:  # Rotate
            print("\n\nrotate")
            print(action,"\n\n")
            new_pos = pos
            #print(new_pos)
            new_orientation = p.getQuaternionFromEuler(
                [self.baseline_roll, self.baseline_pitch, rotation_angles[action]]
            )
        else:
            raise ValueError("Invalid action")

        # Update the cane's position and orientation.
        if not collision_detected or action != 0:
            p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), new_orientation)

        # For observation, we return the cane's new center position.
        distance_to_goal = np.linalg.norm(new_pos - self.goal_location)
        #print(distance_to_goal)

        # Check if the cane has reached the goal location
        goal_location = False
        if distance_to_goal < 0.8 :  # Adjust the threshold value as needed
            # Stop the cane
            p.resetBaseVelocity(self.cane_id, [0, 0, 0], [0, 0, 0])

            # Display a notification
            print("Location Reached! Stopping the cane.")
            goal_location = True
            # You can also add a notification using tkinter or other GUI libraries
            # if you want a pop-up window.

            # Return done=True to indicate that the episode has ended
            print(100)
            return observation, 100, True, False, {}

        ################  CREATE REWARD VALUES #################

        # variables I need to consider: 
        # goal_location = False: boolean
        # distance_to_goal: float
        # previous distance to goal: float
        # collision_detected: boolean

        reward = self.compute_reward(goal_location, distance_to_goal, self.prev_distance_to_goal, collision_detected)
        self.prev_distance_to_goal = distance_to_goal

        print(reward)

        #reward = -distance_to_goal
        #done = False



        #obs = self.get_observation()
        #print("Obs: ",obs)
        ############### Current OBS values ################
        # cane yaw angle
        # primary lidar distance
        # secondary lidar distance
        # distance to goal 
        print(observation)

        self.current_timestep += 1  # Increment timestep
        done = self.current_timestep >= CaneEnv.MAX_TIMESTEPS

        return observation, reward, done,False, {}
    
    def compute_reward(self,goal_location, distance_to_goal, prev_distance_to_goal, collision_detected):
        reward = 0.0

        if goal_location:
            reward += 100.0
        else:
            reward -= 0.9  # Step penalty
            reward += (prev_distance_to_goal - distance_to_goal) * 10

        if collision_detected:
            reward -= 50.0

        return reward


    def reset(self, **kwargs):
        self.current_timestep = 0 
        self.current_swing_deg = 0
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, initial_orientation)
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        self.prev_distance_to_goal = np.linalg.norm(np.array(pos) - np.array(self.goal_location))
        
        obs = np.zeros(85, dtype=np.float32)  # Replace with real values later
        return obs, {}


    
    def render(self, mode="human"):
        pass
    
    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env=CaneEnv()

    model = PPO("MlpPolicy",env,verbose=1)

    model.learn(total_timesteps=10)

    model.save("ppo_cane_model")
    
    try:
        while True:
            # Testing if my github works
            # For testing, use the original action space (movement only).
            # For instance, randomly choose an action.
            #action = env.action_space.sample()
            #env.step(action)
            

            obs, _ = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, _, _ = env.step(action)
                time.sleep(1.0 / 30.0)  # Slow down for visualization



            '''contact_points = p.getContactPoints(bodyA=env.cane_id, bodyB=env.obstacle_id)
            for contact in contact_points:
                pos = contact[5]  # Contact position in world coordinates
                normal = contact[7]  # Contact normal
                print(f"Tip hit obstacle at {pos} with normal {normal}")
            '''

            time.sleep(0.6)
    except KeyboardInterrupt:
        env.close

    #TEst
    
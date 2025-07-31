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
import random
import math
from gymnasium import spaces
from collections import deque
from stable_baselines3 import DQN
#from stable_baselines3 import PPO
#from stable_baselines3.common.monitor import Monitor




class CaneEnv(gym.Env):
    MAX_TIMESTEPS = 300  # Set a max limit for each episode
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
        self.cumulative_reward = 0.0

        ############################ GOAL LOCATION ################################
        self.goal_location = np.array([2.0, -2.0, 1.4])
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
        self.action_space = spaces.Discrete(9)
        
        # Simulation time step for the swing cycle.
        self.dt = 1.0 / 240.0

        # Scaling down the observation space since it is much smaller now, need to think how to keep this consistent dynamically 
        #low_obs = np.full(13, -np.inf, dtype=np.float32)
        #high_obs = np.full(13, np.inf, dtype=np.float32)

        low_obs = np.array([0.0]*20 + [-np.pi, 0.0, -np.pi], dtype=np.float32)
        high_obs = np.array([10.0]*20 + [np.pi, 20.0, np.pi], dtype=np.float32)



        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)


        self.obstacle_ids = []
        num_obstacles = 100
        min_dist = 1.5  # Minimum spacing between any two

        positions = []
        bounds = (-20, 20)

        while len(positions) < num_obstacles:
            x = random.uniform(*bounds)
            y = random.uniform(*bounds)
            z = 0.5

            # Check that this point is far enough from others
            if all(np.linalg.norm(np.array([x, y]) - np.array([px, py])) > min_dist for px, py, _ in positions):
                positions.append((x, y, z))

                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=p.createCollisionShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=[0.3, 0.3, z]
                    ),
                    baseVisualShapeIndex=p.createVisualShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=[0.3, 0.3, z],
                        rgbaColor=[0.8, 0.3, 0.3, 1]
                    ),
                    basePosition=[x, y, z]
                )
                self.obstacle_positions = [(x, y) for x, y, z in positions]
                self.obstacle_ids.append(obstacle_id)


    ''' 
        Revamping to swig based on paramaters: 
        N = number of past and current readings to consider (keep a list where you push and pop values) 
        K = timesteps (how many timesteps before we make a decision)
        T = how many degrees does the cane swing at a time before taking a reading (maybe start with keeping it stagnant at 2, and then later convert to a randomization process. In the range of (0.5 to 3 degrees) 

    '''  
    
    def get_observation_with_swing(self, T=-2, K=10, N=10):
        # Buffer to store the last N observations
        if not hasattr(self, 'swing_observation_buffer') or self.swing_observation_buffer.maxlen != N:
            self.swing_observation_buffer = deque(maxlen=N)

        
        collision = False
        pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)

        angle = 0  # Start at 0
        for step in range(K):
            angle += T  # Step angle by T degrees

            self.current_swing_deg = angle

            new_orientation = p.getQuaternionFromEuler([
                self.baseline_roll, 
                self.baseline_pitch, 
                cane_yaw + math.radians(self.current_swing_deg)
            ])

            p.resetBasePositionAndOrientation(self.cane_id, pos, new_orientation)

            primary_lidar, secondary_lidar = self.get_lidar_data()
            if primary_lidar is None:
                primary_lidar = 3.6
            if secondary_lidar is None:
                secondary_lidar = 3.6

            # Add reading to buffer
            self.swing_observation_buffer.append([primary_lidar, secondary_lidar])

            # Optional: Visual feedback
            contacts = p.getContactPoints(bodyA=self.cane_id)
            for contact in contacts:
                if contact[8] < 0.01:
                    #print("Cane hits obstacle")
                    collision = True
                    T = -T
                p.changeVisualShape(self.cane_id, -1, rgbaColor=[1, 0, 0, 1])
            p.changeVisualShape(self.cane_id, -1, rgbaColor=[0, 1, 0, 1])

            p.stepSimulation()
            time.sleep(self.dt)

        # After K steps, find the angle to goal
        # get x and y of both goal location and cane base
        cane_x, cane_y, cane_z = pos
        goal_x, goal_y = self.goal_location[:2]

        dx = goal_x - cane_x
        dy = goal_y - cane_y

        distance_to_goal = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx) 

        # adding the subtraction of which direction the cane is facing
        #goal_angle = math.atan2(dy, dx)
        #angle_to_goal = goal_angle - cane_yaw

        # Normalize to [-π, π]
        #angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi


        # Add position & direction to goal
        # Updated observation space:
        position_info = [cane_yaw, distance_to_goal, angle_to_goal]
        # position_info = [cane_yaw, cane_x, cane_y, cane_z, dx, dy, dz]

        # Flatten last N readings + position info
        flattened_readings = np.array(self.swing_observation_buffer).flatten()
        full_obs = np.concatenate((flattened_readings, position_info))
        

        #print("Final observation shape:", full_obs.shape)


        return full_obs, collision, angle_to_goal

    def get_lidar_data(self):
        # Remove previous debug beams (if they exist)
        try:
            if hasattr(self, 'beam_id'):
                p.removeUserDebugItem(self.beam_id)
            if hasattr(self, 'beam_id_secondary'):
                p.removeUserDebugItem(self.beam_id_secondary)
        except:
            pass  # In case it's already removed

        # Get cane position and orientation
        cane_pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)

        # ========== PRIMARY LIDAR ==========
        lidar_offset_z = -1.5
        lidar_offset = [0, 0, lidar_offset_z]
        rotated_offset = p.rotateVector(cane_orientation, lidar_offset)

        lidar_pos = [
            cane_pos[0] + rotated_offset[0],
            cane_pos[1] + rotated_offset[1],
            cane_pos[2] + rotated_offset[2]
        ]

        beam_direction = [
            -math.sin(cane_yaw),
            math.cos(cane_yaw),
            -math.sin(math.radians(45))  # Clean 45-degree downward pitch
        ]

        step_size = 0.3
        num_steps = 1.2
        beam_end = [
            lidar_pos[0] + num_steps * step_size * beam_direction[0],
            lidar_pos[1] + num_steps * step_size * beam_direction[1],
            lidar_pos[2] + num_steps * step_size * beam_direction[2]
        ]

        self.beam_id = p.addUserDebugLine(lidar_pos, beam_end, [1, 0, 0], 2, 0.1)

        # ========== SECONDARY LIDAR ==========
        lidar_offset_y = -0.7
        secondary_lidar_offset = [0, 0, lidar_offset_y]
        rotated_secondary_offset = p.rotateVector(cane_orientation, secondary_lidar_offset)

        secondary_lidar_pos = [
            cane_pos[0] + rotated_secondary_offset[0],
            cane_pos[1] + rotated_secondary_offset[1],
            cane_pos[2] + rotated_secondary_offset[2]
        ]

        secondary_beam_direction = [
            -math.sin(cane_yaw),
            math.cos(cane_yaw),
            0  # No pitch
        ]

        num_steps_secondary = 6
        beam_end_secondary = [
            secondary_lidar_pos[0] + num_steps_secondary * step_size * secondary_beam_direction[0],
            secondary_lidar_pos[1] + num_steps_secondary * step_size * secondary_beam_direction[1],
            secondary_lidar_pos[2] + num_steps_secondary * step_size * secondary_beam_direction[2]
        ]

        self.beam_id_secondary = p.addUserDebugLine(secondary_lidar_pos, beam_end_secondary, [1, 0, 0], 2, 0.1)

        # ========== RAYCAST ==========

        result_primary = p.rayTest(lidar_pos, beam_end)
        result_secondary = p.rayTest(secondary_lidar_pos, beam_end_secondary)

        # Primary LiDAR
        if result_primary[0][0] == -1:
            lidar1_value = 0.0
        else:
            hit_position = result_primary[0][3]
            lidar1_value = math.dist(lidar_pos, hit_position)

        # Secondary LiDAR
        if result_secondary[0][0] == -1 or result_secondary[0][0] == self.cane_id:
            lidar2_value = 0.0
        else:
            hit_position = result_secondary[0][3]
            lidar2_value = math.dist(secondary_lidar_pos, hit_position)

        return lidar1_value, lidar2_value

    def step(self, action):
        # Step 1: Get observation and current state
        observation, collision, angle_to_goal = self.get_observation_with_swing()
        pos, orientation = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        step_size = 0.3
        self.last_cane_position, self.last_cane_orientation = p.getBasePositionAndOrientation(self.cane_id)


        # Step 2: Define rotation angles
        rotation_angles = {
            2: math.radians(30),   # short left
            3: math.radians(-30),  # short right
            4: math.radians(60),   # medium left
            5: math.radians(-60),  # medium right
            6: math.radians(90),   # hard left
            7: math.radians(-90),  # hard right
            8: math.radians(180)   # turn around
        }

        # Step 3: Compute proposed action
        new_pos = np.array(pos)
        new_yaw = yaw
        

        if isinstance(action, np.ndarray):
            action = action.item()

        if action == 0:  # Step forward
            new_pos = pos + np.array([
                -step_size * math.sin(yaw),
                step_size * math.cos(yaw),
                0
            ])

        elif action == 1:  # Stop
            pass  # Keep position and orientation unchanged

        elif action in rotation_angles:  # Rotate
            new_yaw += rotation_angles[action]

        else:
            raise ValueError("Invalid action")

        # Step 4: Check for collision at proposed pose
        proposed_orientation = p.getQuaternionFromEuler([roll, pitch, new_yaw])
        p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), proposed_orientation)
        contacts = p.getContactPoints(bodyA=self.cane_id)
        collision_detected = any(contact[8] < 0.01 for contact in contacts)

        # Step 5: Apply or revert based on collision
        if collision_detected and action == 0:
            # Revert position and orientation
            p.resetBasePositionAndOrientation(self.cane_id, pos.tolist(), orientation)
            p.stepSimulation() 
            new_pos = pos
            #T = -T
            new_orientation = orientation
        else:
            new_orientation = proposed_orientation
            p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), new_orientation)

        

        # Step 6: Check if goal reached
        distance_to_goal = np.linalg.norm(new_pos - self.goal_location)
        goal_location = False
        if distance_to_goal < 0.5:
            p.resetBaseVelocity(self.cane_id, [0, 0, 0], [0, 0, 0])
            print("Location Reached! Stopping the cane.")
            goal_location = True
            return observation, 100, True, False, {
                "goal_reached": True,
                "collision": collision_detected,
                "steps_taken": self.current_timestep,
                "reward": 100
            }
        #else self.current_step >= self.MAX_TIMESTEPS:
        #    done = True

        # Step 7: Compute reward
        reward = self.compute_reward(
            goal_location,
            distance_to_goal,
            self.prev_distance_to_goal,
            collision,
            angle_to_goal,
            self.prev_angle_to_goal,
            action
        )
        self.prev_distance_to_goal = distance_to_goal
        self.prev_angle_to_goal = angle_to_goal

        # Step 8: Prepare info and check episode done
        self.current_timestep += 1
        done = self.current_timestep >= CaneEnv.MAX_TIMESTEPS
        self.cumulative_reward += reward

        info = {
            "goal_reached": goal_location,
            "collision": collision_detected,
            "steps_taken": self.current_timestep,
            #"reward": reward,
            "cumulative reward": self.cumulative_reward
        }

        print(info)
        #print(observation)

        return observation, reward, done, False, info

    def compute_reward(self, goal_location, distance_to_goal, prev_distance_to_goal, collision_detected,angle_to_goal,prev_angle_to_goal,action):
        reward = 0.0

        if goal_location:
            reward += 100.0
        else:
            # penalty for moving away from goal
            reward += (prev_distance_to_goal - distance_to_goal) * 10
            print("Test:",reward)
            if distance_to_goal > prev_distance_to_goal:
                reward -= 0.5  # penalty for moving away from the goal

        if collision_detected:
            reward -= 2.0 

        #reward -= 0.5 #small time penalty
        
        #considering the angles and turning away from the goal
        # will this still be relevant with my updated angle ?? 
        # if abs(angle_to_goal) > abs(prev_angle_to_goal):
        #     reward -= 0.5  # Penalize turning away from the goal

        # else:
        #     # Reward turning to the goal
        #     reward += 0.2  # Reward turning toward goal 

        print("Action:", action, "Reward:", reward)

        return reward

    def random_starting_pos(self,obstacles, safe_radius=1.0):
        bounds = (-20, 20)
        for _ in range(20):
            x = random.uniform(*bounds)
            y = random.uniform(*bounds)
            vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))

            if all(math.hypot(x - ox, y - oy) >= safe_radius for ox, oy in obstacles):
                return [x, y, vertical_offset + 0.75]  # Z is height
        raise RuntimeError("Could not find valid spawn position")

         

    def reset(self, **kwargs):
        #print("\n RESET \n")
        self.current_timestep = 0
        self.cumulative_reward = 0.0 
        self.current_swing_deg = 0

        self.cane_start_pos = self.random_starting_pos(
            obstacles=self.obstacle_positions,
            safe_radius=1.0
        )

        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(0)]
        )
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, initial_orientation)
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        self.prev_distance_to_goal = np.linalg.norm(np.array(pos) - np.array(self.goal_location))
        self.prev_angle_to_goal = 0

        # Changed to 13 to meet the new array requirements. 
        # now includes polar coordinates
        obs = np.zeros(23, dtype=np.float32)  # Replace with real values later
        return obs, {}


    
    def render(self, mode="human"):
        pass
    
    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env=CaneEnv()
    #env = Monitor(env)

    
    model = DQN("MlpPolicy",env,verbose=1, exploration_initial_eps=0.8, exploration_final_eps=0.02, exploration_fraction=0.2, )
    #model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_tensorboard/")

    model.learn(total_timesteps=1000)

    model.save("dqn_cane_model")
    '''
    model = PPO("MlpPolicy",env,verbose=1)
    #model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_tensorboard/")

    model.learn(total_timesteps=1000)

    model.save("ppo_cane_model")
    '''
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
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                '''
                action, _states = model.predict(obs)
                obs, reward, done, _, _ = env.step(action)
                time.sleep(1.0 / 30.0)  # Slow down for visualization
                '''

            time.sleep(0.6)
    except KeyboardInterrupt:
        env.close


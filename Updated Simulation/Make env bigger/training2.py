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
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

class CaneEnv(gym.Env):
    MAX_TIMESTEPS = 200 # 30 seconds worth of steps



    #MAX_TIMESTEPS = 300  # Set a max limit for each episode
    def __init__(self, gui=False):
        super(CaneEnv, self).__init__()
        
        # Connect to PyBullet in GUI mode and set up simulation.
        #self.physics_client = p.connect(p.GUI)

        if p.isConnected():
            p.disconnect()

        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) /1
        
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

        ###########
        self.last_safe_pos = inertial_pos
        self.last_safe_orientation = initial_orientation
        self.collision_count = 0
        self.safe_steps_count = 0

        ############################ GOAL LOCATION ################################
        self.goal_location = np.array([0, 0, 20])
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.1]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.1],
                rgbaColor=[0, 1, 0, 1]
            ),
            basePosition=self.goal_location
        )

        self.action_space = spaces.Discrete(11)
        
        # Simulation time step for the swing cycle.
        self.dt = 1.0 / 240.0

        low_obs = np.array([0.0]*20 + [-np.pi, 0.0, -np.pi], dtype=np.float32)
        high_obs = np.array([10.0]*20 + [np.pi, 20.0, np.pi], dtype=np.float32)

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)


        self.obstacle_ids = []
        num_obstacles = 40
        min_dist = 1.5  # Minimum spacing between any two

        positions = []
        bounds = (-10, 10)

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

                    ############ Test collision on SWING ###################
                    p.resetBasePositionAndOrientation(self.cane_id, pos, cane_orientation)
                    break
                p.changeVisualShape(self.cane_id, -1, rgbaColor=[1, 0, 0, 1])
            p.changeVisualShape(self.cane_id, -1, rgbaColor=[0, 1, 0, 1])

            #p.stepSimulation()
            #time.sleep(self.dt)

        # After K steps, find the angle to goal
        # get x and y of both goal location and cane base
        cane_x, cane_y, cane_z = pos
        goal_x, goal_y = self.goal_location[:2]

        dx = goal_x - cane_x
        dy = goal_y - cane_y

        distance_to_goal = math.hypot(dx, dy)
        goal_angle = math.atan2(dy, dx) 

        # adding the subtraction of which direction the cane is facing
        #goal_angle = math.atan2(dy, dx)
        angle_to_goal = goal_angle - cane_yaw

        # Normalize to [-π, π]
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi


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

        if isinstance(action, np.ndarray):
            action = int(action.item())  # Convert numpy array to plain int


        # Step 1: Get observation and current state
        observation, collision, angle_to_goal = self.get_observation_with_swing()
        
        collision_detected = collision
        pos, orientation = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        step_size = 0.5
        self.last_cane_position, self.last_cane_orientation = p.getBasePositionAndOrientation(self.cane_id)

        # Example: extend action space
        # 0 = small forward, 1 = medium forward, 2 = big forward,
        # 3 = stop, 4–10 = turns
        
        self.action_space = spaces.Discrete(11)



        step_sizes = {
            0: 0.3,   # small forward
            1: 0.6,   # medium forward
            2: 1.0    # big forward
        }

        rotation_angles = {
            4: math.radians(30),   # short left
            5: math.radians(-30),  # short right
            6: math.radians(60),   # medium left
            7: math.radians(-60),  # medium right
            8: math.radians(90),   # hard left
            9: math.radians(-90),  # hard right
            10: math.radians(180)  # turn around
        }

        # In step():

        new_yaw = yaw
        if action in step_sizes:
            step_size = step_sizes[action]
            new_pos = pos + np.array([
                -step_size * math.sin(yaw),
                step_size * math.cos(yaw),
                0
            ])
        elif action == 3:  # stop
            new_pos = pos
        elif action in rotation_angles:
            new_yaw = yaw + rotation_angles[action]
            proposed_orientation = p.getQuaternionFromEuler([roll, pitch, new_yaw])
            # Keep same position, just rotate
            new_pos = pos
        else:
            raise ValueError(f"Invalid action: {action}")

        # new placement for test nr 21
        observation, collision, angle_to_goal = self.get_observation_with_swing()
        collision_detected = collision

        # Step 4: Check for collision at proposed pose
        proposed_orientation = p.getQuaternionFromEuler([roll, pitch, new_yaw])
        p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), proposed_orientation)
        contacts = p.getContactPoints(bodyA=self.cane_id)
        collision_detected = any(contact[8] < 0.01 for contact in contacts)

        if collision_detected:
            self.collision_count += 1
            # Push the cane slightly backwards along its facing direction
            if self.collision_count >= 4:
                # Too many collisions – reset to last safe pos
                p.resetBasePositionAndOrientation(self.cane_id, self.last_safe_pos, self.last_safe_orientation)
                self.collision_count = 0  # Reset tracker
            else:
                backoff_distance = 0.3
                backoff_vector = [
                    backoff_distance * math.sin(yaw),
                    -backoff_distance * math.cos(yaw),
                    0
                ]
                escape_pos = [
                    pos[0] + backoff_vector[0],
                    pos[1] + backoff_vector[1],
                    (self.cane_height / 2) * math.cos(math.radians(45)) + 0.75
                ]
                p.resetBasePositionAndOrientation(self.cane_id, escape_pos, orientation)

                self.collision_count = 0


        # Step 6: Check if goal reached
        #distance_to_goal = np.linalg.norm(new_pos - self.goal_location)
        
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        distance_to_goal = np.linalg.norm(pos - self.goal_location)
       

        '''
        if distance_to_goal < 0.5:
            p.resetBaseVelocity(self.cane_id, [0, 0, 0], [0, 0, 0])
            #print("Location Reached! Stopping the cane.")
            goal_location = True
            return observation, 100, True, False, {
                "goal_reached": True,
                "collision": collision_detected,
                "steps_taken": self.current_timestep,
                "reward": 100
            }
        '''
        goal_location = False
        done = False
        terminated = False

        #observation, collision, angle_to_goal = self.get_observation_with_swing()

        if distance_to_goal < 0.5:  # within goal radius
            goal_location = True
            terminated = True       # stop episode immediately
            truncated = False       # not truncated
            reward = 100            # give final reward for reaching goal
            self.cumulative_reward += reward
            self.current_timestep += 1
            info = {
                "goal_reached": True,
                "collision_detected": collision_detected,
                "steps_taken": self.current_timestep,
                "cumulative reward": self.cumulative_reward
            }
            return observation, reward, terminated, truncated, info  # exit immediately

        #observation, collision, angle_to_goal = self.get_observation_with_swing()

        # Step 7: Compute reward
        reward = self.compute_reward(
            goal_location,
            distance_to_goal,
            self.prev_distance_to_goal,
            collision_detected,
            angle_to_goal,
            self.prev_angle_to_goal,
            action
        )

        

        # Step 8: Prepare info and check episode done
        self.current_timestep += 1
        truncated = self.current_timestep >= CaneEnv.MAX_TIMESTEPS
        self.cumulative_reward += reward

        if not terminated and not truncated:
            self.prev_distance_to_goal = distance_to_goal
            self.prev_angle_to_goal = angle_to_goal

        info = {
            "goal_reached": goal_location,
            "collision_detected": collision_detected,
            "steps_taken": self.current_timestep,
            #"reward": reward,
            "cumulative reward": self.cumulative_reward
        }

        #print(info)
        #print(observation)
        

        

        return observation, reward,terminated, truncated, info
    
    def compute_reward(self, goal_location, distance_to_goal, prev_distance_to_goal, collision_detected,angle_to_goal,prev_angle_to_goal,action):
        reward = 0.0

        if goal_location:
            return 100
        
        if distance_to_goal < 0.5:  # small threshold around the goal
            angle_to_goal = 0.0
        
        progress = (prev_distance_to_goal - distance_to_goal)
        reward +=  progress * 2.5
        
        if abs(progress) < 0.001:
            reward -= 0.2
        
        #reward -= abs(angle_to_goal) * 0.5

        #angle_progress = abs(prev_angle_to_goal) - abs(angle_to_goal)
        #reward += angle_progress * 0.3

        #if action in [0, 1, 2] and progress > 0:
        #    angle_progress = abs(prev_angle_to_goal) - abs(angle_to_goal)
        #    reward += angle_progress * 0.3

        angle_diff = angle_to_goal  # signed, not abs
        reward += math.cos(angle_diff) * 2.0  


        if collision_detected:
            reward -= 5.0 

        reward -= 0.3
 
        
        return reward

    #if distance_to_goal < 0.7:  # small threshold around the goal
        #    angle_to_goal = 0.0

        #if distance_to_goal > prev_distance_to_goal:
        #        reward -= 0.5
        
        #angle_diff = prev_angle_to_goal - angle_to_goal
        #reward += angle_diff * 0.2

    def random_starting_pos(self,obstacles, safe_radius=1.0):
        bounds = (-10, 10)
        for _ in range(20):
            x = random.uniform(*bounds)
            y = random.uniform(*bounds)
            vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))

            if all(math.hypot(x - ox, y - oy) >= safe_radius for ox, oy in obstacles):
                return [x, y, vertical_offset + 0.75]  # Z is height
        raise RuntimeError("Could not find valid spawn position")

    def random_goal_pos(self, safe_radius=1.0):
        bounds = (-10, 10)
        for _ in range(50):
            x = random.uniform(*bounds)
            y = random.uniform(*bounds)
            z = 1.4  # keep same height for visibility

            # Ensure goal isn’t too close to obstacles
            if all(math.hypot(x - ox, y - oy) >= safe_radius for ox, oy in self.obstacle_positions):
                return [x, y, z]
        raise RuntimeError("Could not find valid goal position")

    def reset(self, **kwargs):
        if not p.isConnected():
            self.physics_client = p.connect(p.DIRECT)

        self.current_timestep = 0
        self.cumulative_reward = 0.0 
        self.current_swing_deg = 0

        # Random cane start
        self.cane_start_pos = self.random_starting_pos(
            obstacles=self.obstacle_positions,
            safe_radius=1.0
        )
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, 0]
        )
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, initial_orientation)


        ########################## GOAL LOCATION (IN reset()) ##########################
        self.goal_location = self.random_starting_pos(
            obstacles=self.obstacle_positions, #+ [self.cane_start_pos],  # avoid both
            safe_radius=1.0  # increase if needed
        )

        p.resetBasePositionAndOrientation(
            self.goal_id,
            self.goal_location,
            [0, 0, 0, 1]
        )




        ####################


        # Init tracking vars
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        self.prev_distance_to_goal = np.linalg.norm(np.array(pos) - self.goal_location)
        self.prev_angle_to_goal = 0

        obs = np.zeros(23, dtype=np.float32)
        return obs, {}

    
    def render(self, mode="human"):
        pass
    
    def close(self):
        #if p.isConnected():
        p.disconnect()


def make_env():
    """
    Helper function to create a monitored environment for parallel processing.
    """
    env = CaneEnv(gui=False)  # GUI must be False for parallel workers
    env = Monitor(env, filename=f"{log_dir}/cane_23")
    return env

if __name__ == "__main__":
    #env=CaneEnv()

    num_cpu = 8 
    #vec_env = SubprocVecEnv([CaneEnv for _ in range(num_cpu)])

    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    #env = Monitor(env, filename=f"{log_dir}/cane_monitor_02.csv")
    #env = Monitor(env)

    #random.seed(1001)
    
    #model = DQN("MlpPolicy",env,verbose=1, exploration_initial_eps=0.8, exploration_final_eps=0.02, exploration_fraction=0.2, )
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-4, 
        buffer_size=100_000,
        learning_starts=50_000,
        batch_size=64,
        tau=1.0,  # Hard update (DQN default)
        train_freq=4,
        target_update_interval=500,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,  # decay over 20% of training
        gamma=0.95,
    )

    #model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_tensorboard/")

    model.learn(total_timesteps=9000 * CaneEnv.MAX_TIMESTEPS)

    model.save("dqn_cane_model_23")
    print("Model saved after training.")


    
    '''
    model = PPO("MlpPolicy",env,verbose=1)
    #model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_tensorboard/")

    model.learn(total_timesteps=1000)

    model.save("ppo_cane_model")
    
    env=CaneEnv()
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
                action, _states = model.predict(obs, deterministic=True)
                #action, _states = model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            #time.sleep(0.6)
    
    
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, saving model and closing env.")
        model.save("dqn_cane_new_02")
        env.close()
    '''

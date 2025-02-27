import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from gymnasium import spaces

# ball not a cane that just moves on a simple action space


class BlindUserEnv(gym.Env):
    def __init__(self):
        super(BlindUserEnv, self).__init__()


        # Start PyBullet in GUI mode so we can see the simulation
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # I am trying to get a free cam, why won't it work ?
        #p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        #p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_ROTATION, 1)
        #p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_ZOOM, 1)
        
        # Apparently pybullet does not really give free camera options and the viewing options is a little limited
        


        # set a free cam
        # p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

        # Create 3D Plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create a sphere representing the visually impaired user
        self.user_start_pos = [0, 0, 0.1]  # (x, y, z)
        #self.user_id = p.loadURDF("sphere_small.urdf", basePosition=[0, 0, 1])


        # attempt at the humanoid 
        # It ain't working 

        self.user_id = p.loadURDF("small_sphere.urdf", basePosition=[0, 0, 1])
        p.resetBasePositionAndOrientation(humanoid_id, [0, 0, 1], p.getQuaternionFromEuler([math.radians(90), 0, 0]))
        
        # Trying to enable mouse control: 
        # Enable mouse control
        #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,cameraTargetPosition=[0, 0, 0])
        
        

        # Define Action Space (Move Forward, Backward, Left, Right, Stop)
        self.action_space = spaces.Discrete(5)

        # Define Observation Space (User Position in 3D Space)
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0]), 
            high=np.array([10, 10, 1]), 
            dtype=np.float32
        )

    def step(self, action):
        """Apply an action and move the user."""
        pos, _ = p.getBasePositionAndOrientation(self.user_id)

        # Define movement step size
        step_size = 0.1

        if action == 0:  # Move Forward (+Y)
            new_pos = [pos[0], pos[1] + step_size, pos[2]]
        elif action == 1:  # Move Backward (-Y)
            new_pos = [pos[0], pos[1] - step_size, pos[2]]
        elif action == 2:  # Move Left (-X)
            new_pos = [pos[0] - step_size, pos[1], pos[2]]
        elif action == 3:  # Move Right (+X)
            new_pos = [pos[0] + step_size, pos[1], pos[2]]
        elif action == 4:  # Stop (do nothing)
            new_pos = pos  # No movement, stay in the current position

        # Apply movement
        p.resetBasePositionAndOrientation(self.user_id, new_pos, [0, 0, 0, 1])

        # Get updated position
        new_pos, _ = p.getBasePositionAndOrientation(self.user_id)

        # Simple Reward (Encourage Forward Movement)
        reward = new_pos[1]  # More reward for moving forward

        # Keep the simulation running indefinitely
        done = False

        return np.array(new_pos, dtype=np.float32), reward, done, {}

    def reset(self):
        """Reset the environment to its initial state."""
        p.resetBasePositionAndOrientation(self.user_id, self.user_start_pos, [0, 0, 0, 1])
        return np.array(self.user_start_pos, dtype=np.float32)

    def render(self, mode="human"):
        """Already running in GUI mode, so we don't need to reconnect."""
        pass

    def close(self):
        """Close the simulation."""
        p.disconnect()

# Run the simulation continuously with a delay
if __name__ == "__main__":
    env = BlindUserEnv()
    env.reset()

    while True:  # Infinite loop to keep running
        action = env.action_space.sample()  # Take random actions
        env.step(action)
        time.sleep(0.2)  # Slow down the movement (increase value to slow down more)

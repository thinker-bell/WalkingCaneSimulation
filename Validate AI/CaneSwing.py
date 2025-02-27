'''
This swings the cane with x (10) degrees with every step movement
this does not really work well with simulation how to user would use the cane while walking
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
        
        # Fixed baseline: 45° tilt about the X-axis.
        self.baseline_roll = math.radians(45)
        self.baseline_pitch = 0
        
        # We'll automatically update the yaw (swing) based on movement.
        self.current_swing_deg = 0  # In degrees
        
        # Compute vertical offset so that when tilted, the bottom nearly touches the plane.
        vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))
        self.cane_start_pos = [0, 0, vertical_offset]
        
        # Compute the initial orientation (roll=45, pitch=0, yaw=0).
        initial_orientation = p.getQuaternionFromEuler([self.baseline_roll, self.baseline_pitch, math.radians(self.current_swing_deg)])
        
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
        # 4: Stop
        self.action_space = spaces.Discrete(5)
        
        # Observation space: just the cane’s center position.
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0]),
            high=np.array([10, 10, 10]),
            dtype=np.float32
        )
        
        # For automatic swing updates, store the last position.
        self.last_pos = np.array(self.cane_start_pos, dtype=np.float32)
        
    def step(self, action):
        # Get current position.
        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        step_size = 0.1
        
        # Compute new position based on action.
        if action == 0:      # Forward (+Y)
            new_pos = pos + np.array([0, step_size, 0])
        elif action == 1:    # Backward (-Y)
            new_pos = pos + np.array([0, -step_size, 0])
        elif action == 2:    # Left (-X)
            new_pos = pos + np.array([-step_size, 0, 0])
            self.current_swing_deg += 5  # Increase swing to the left.
        elif action == 3:    # Right (+X)
            new_pos = pos + np.array([step_size, 0, 0])
            self.current_swing_deg -= 5  # Increase swing to the right.
        elif action == 4:    # Stop
            new_pos = pos
        
        # If the action is forward, backward, or stop, let the swing angle decay toward 0.
        if action in [0, 1, 4]:
            if self.current_swing_deg > 0:
                self.current_swing_deg = max(0, self.current_swing_deg - 2)
            elif self.current_swing_deg < 0:
                self.current_swing_deg = min(0, self.current_swing_deg + 2)
        
        # Clamp the swing angle to a reasonable range, e.g. [-50, 50] degrees.
        self.current_swing_deg = max(-50, min(50, self.current_swing_deg))
        
        # Compute new orientation: baseline is fixed roll and pitch; yaw is the current swing.
        new_orientation = p.getQuaternionFromEuler([self.baseline_roll,
                                                      self.baseline_pitch,
                                                      math.radians(self.current_swing_deg)])
        # Update the cane's state.
        p.resetBasePositionAndOrientation(self.cane_id, new_pos.tolist(), new_orientation)
        
        # Save current position.
        self.last_pos = new_pos
        
        # For this example, reward could be forward progress (y-axis displacement).
        reward = new_pos[1]
        done = False
        
        return new_pos, reward, done, {}
    
    def reset(self):
        self.current_swing_deg = 0
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos,
                                          p.getQuaternionFromEuler([self.baseline_roll, self.baseline_pitch, 0]))
        self.last_pos = np.array(self.cane_start_pos, dtype=np.float32)
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
            # For testing, sample a random action.
            action = env.action_space.sample()
            env.step(action)
            time.sleep(0.3)
    except KeyboardInterrupt:
        env.close()

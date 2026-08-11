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
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# --- NEW: needed for the custom Double DQN train() override ---
import torch as th
import torch.nn.functional as F

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)


class CaneCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.global_successes = 0

        for info in infos:
            if "episode" in info:
                ep = info["episode"]

                if "episode" in info:
                    self.logger.record("custom/ep_collisions", info["episode"].get("collisions", 0))
                self.logger.record("custom/ep_reward", ep["r"])
                self.logger.record("custom/ep_length", ep["l"])
                self.logger.record("custom/total_successes", info.get("total_successes", 0))
                if info.get("goal_reached", False):
                    self.global_successes += 1

                    self.logger.record("custom/global_successes", self.global_successes)

        return True


# =========================================================================
# NEW: Double DQN
#
# SB3's stock DQN.train() computes the TD target as:
#     next_q_values = q_net_target(next_obs).max(dim=1)
# i.e. the SAME target network both SELECTS and EVALUATES the best next
# action. That's vanilla DQN, and it's the source of the overestimation
# bias discussed in the paper's Section 5.3 / Limitations.
#
# Double DQN decouples selection and evaluation:
#     next_action = argmax_a  q_net(next_obs)        <- ONLINE network selects
#     next_q      = q_net_target(next_obs)[next_action]  <- TARGET network evaluates
#
# Everything else (replay buffer, target network sync schedule, epsilon
# schedule, architecture, optimizer) is identical to vanilla DQN, so this
# is a minimal, apples-to-apples change relative to the DQN baseline.
# =========================================================================
class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # --- Double DQN target computation ---
                # 1. Online network selects the greedy next action.
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

                # 2. Target network evaluates that chosen action.
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(next_q_values_target, dim=1, index=next_actions)

                # 3. Standard Bellman backup.
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Current Q-estimates for the actions actually taken.
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class CaneEnv(gym.Env):
    MAX_TIMESTEPS = 200  # 30 seconds worth of steps

    def __init__(self, gui=False):
        super(CaneEnv, self).__init__()

        if p.isConnected():
            p.disconnect()

        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.episode_collisions = 0
        self.episode_success = 0
        self.total_successes = 0

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()

        self.plane_id = p.loadURDF("plane.urdf")

        # Cane properties.
        self.cane_radius = 0.025
        self.cane_height = 2.0
        self.cane_mass = 1.0

        self.baseline_roll = math.radians(45)
        self.baseline_pitch = 0

        self.current_swing_deg = 0

        vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))
        self.cane_start_pos = [0, 0, vertical_offset + 0.75]

        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, math.radians(90)]
        )
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                   radius=self.cane_radius,
                                                   height=self.cane_height)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                            radius=self.cane_radius,
                                            length=self.cane_height,
                                            rgbaColor=[1, 0.75, 0.8, 1])

        com_height = self.cane_height - (self.cane_height / 8)
        inertial_pos = [0, 0, com_height / 2]

        base_pos = [0, 0, self.cane_height / 2 + 0.1]

        self.cane_id = p.createMultiBody(baseMass=self.cane_mass,
                                          baseCollisionShapeIndex=collision_shape,
                                          baseVisualShapeIndex=visual_shape,
                                          basePosition=base_pos,
                                          baseOrientation=initial_orientation,
                                          baseInertialFramePosition=inertial_pos)

        self.lidar_start_pos = [0, 0, self.cane_height / 8]
        self.cumulative_reward = 0.0

        self.last_safe_pos = inertial_pos
        self.last_safe_orientation = initial_orientation
        self.collision_count = 0
        self.safe_steps_count = 0

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

        self.dt = 1.0 / 240.0

        low_obs = np.array([0.0] * 20 + [-np.pi, 0.0, -np.pi], dtype=np.float32)
        high_obs = np.array([10.0] * 20 + [np.pi, 20.0, np.pi], dtype=np.float32)

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.obstacle_ids = []
        num_obstacles = 60
        min_dist = 1.5

        positions = []
        bounds = (-10, 10)

        while len(positions) < num_obstacles:
            x = random.uniform(*bounds)
            y = random.uniform(*bounds)
            z = 0.5

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

    def get_observation_with_swing(self, T=-2, K=10, N=10):
        if not hasattr(self, 'swing_observation_buffer') or self.swing_observation_buffer.maxlen != N:
            self.swing_observation_buffer = deque(maxlen=N)

        collision = False
        pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)

        angle = 0
        for step in range(K):
            angle += T

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

            self.swing_observation_buffer.append([primary_lidar, secondary_lidar])

            contacts = p.getContactPoints(bodyA=self.cane_id)
            for contact in contacts:
                if contact[8] < 0.01:
                    collision = True
                    T = -T
                    p.resetBasePositionAndOrientation(self.cane_id, pos, cane_orientation)
                    break
                p.changeVisualShape(self.cane_id, -1, rgbaColor=[1, 0, 0, 1])
            p.changeVisualShape(self.cane_id, -1, rgbaColor=[0, 1, 0, 1])

        cane_x, cane_y, cane_z = pos
        goal_x, goal_y = self.goal_location[:2]

        dx = goal_x - cane_x
        dy = goal_y - cane_y

        distance_to_goal = math.hypot(dx, dy)
        goal_angle = math.atan2(dy, dx)

        angle_to_goal = goal_angle - cane_yaw
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi

        position_info = [cane_yaw, distance_to_goal, angle_to_goal]

        flattened_readings = np.array(self.swing_observation_buffer).flatten()
        full_obs = np.concatenate((flattened_readings, position_info))

        return full_obs, collision, angle_to_goal

    def get_lidar_data(self):
        try:
            if hasattr(self, 'beam_id'):
                p.removeUserDebugItem(self.beam_id)
            if hasattr(self, 'beam_id_secondary'):
                p.removeUserDebugItem(self.beam_id_secondary)
        except:
            pass

        cane_pos, cane_orientation = p.getBasePositionAndOrientation(self.cane_id)
        cane_roll, cane_pitch, cane_yaw = p.getEulerFromQuaternion(cane_orientation)

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
            -math.sin(math.radians(45))
        ]

        step_size = 0.3
        num_steps = 1.2
        beam_end = [
            lidar_pos[0] + num_steps * step_size * beam_direction[0],
            lidar_pos[1] + num_steps * step_size * beam_direction[1],
            lidar_pos[2] + num_steps * step_size * beam_direction[2]
        ]

        self.beam_id = p.addUserDebugLine(lidar_pos, beam_end, [1, 0, 0], 2, 0.1)

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
            0
        ]

        num_steps_secondary = 6
        beam_end_secondary = [
            secondary_lidar_pos[0] + num_steps_secondary * step_size * secondary_beam_direction[0],
            secondary_lidar_pos[1] + num_steps_secondary * step_size * secondary_beam_direction[1],
            secondary_lidar_pos[2] + num_steps_secondary * step_size * secondary_beam_direction[2]
        ]

        self.beam_id_secondary = p.addUserDebugLine(secondary_lidar_pos, beam_end_secondary, [1, 0, 0], 2, 0.1)

        result_primary = p.rayTest(lidar_pos, beam_end)
        result_secondary = p.rayTest(secondary_lidar_pos, beam_end_secondary)

        if result_primary[0][0] == -1:
            lidar1_value = 0.0
        else:
            hit_position = result_primary[0][3]
            lidar1_value = math.dist(lidar_pos, hit_position)

        if result_secondary[0][0] == -1 or result_secondary[0][0] == self.cane_id:
            lidar2_value = 0.0
        else:
            hit_position = result_secondary[0][3]
            lidar2_value = math.dist(secondary_lidar_pos, hit_position)

        return lidar1_value, lidar2_value

    def step(self, action):

        if isinstance(action, np.ndarray):
            action = int(action.item())

        pos, orientation = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        self.last_cane_position = pos.copy()
        self.last_cane_orientation = orientation

        step_sizes = {
            0: 0.3,
            1: 0.6,
            2: 1.0
        }

        rotation_angles = {
            4: math.radians(30),
            5: math.radians(-30),
            6: math.radians(60),
            7: math.radians(-60),
            8: math.radians(90),
            9: math.radians(-90),
            10: math.radians(180)
        }

        new_yaw = yaw
        new_pos = pos.copy()

        if action in step_sizes:
            s = step_sizes[action]
            new_pos = pos + np.array([
                -s * math.sin(yaw),
                s * math.cos(yaw),
                0
            ])

        elif action == 3:
            new_pos = pos

        elif action in rotation_angles:
            new_yaw = yaw + rotation_angles[action]

        else:
            raise ValueError(f"Invalid action: {action}")

        new_orientation = p.getQuaternionFromEuler([roll, pitch, new_yaw])

        p.resetBasePositionAndOrientation(
            self.cane_id,
            new_pos.tolist(),
            new_orientation
        )

        p.stepSimulation()

        contacts = p.getContactPoints(bodyA=self.cane_id)
        collision_detected = len(contacts) > 0

        self.last_collision = collision_detected

        if collision_detected:
            self.episode_collisions += 1
            self.collision_count += 1
        else:
            self.collision_count = 0

        if collision_detected:
            if self.collision_count >= 4:
                p.resetBasePositionAndOrientation(
                    self.cane_id,
                    self.last_safe_pos,
                    self.last_safe_orientation
                )
                self.collision_count = 0
            else:
                backoff = 0.3
                escape_pos = pos + np.array([
                    backoff * math.sin(yaw),
                    -backoff * math.cos(yaw),
                    0
                ])

                p.resetBasePositionAndOrientation(
                    self.cane_id,
                    escape_pos.tolist(),
                    orientation
                )

        else:
            self.last_safe_pos = new_pos.copy()
            self.last_safe_orientation = new_orientation

        observation, _, angle_to_goal = self.get_observation_with_swing()

        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        pos = np.array(pos)

        distance_to_goal = np.linalg.norm(pos - self.goal_location)

        terminated = False
        truncated = False
        reward = 0.0
        goal_location = False

        if distance_to_goal < 1:

            reward = 200
            self.total_successes += 1
            self.cumulative_reward += reward
            self.episode_success = 1
            terminated = True
            goal_location = True

            info = {
                "goal_reached": True,
                "collision_detected": collision_detected,
                "steps_taken": self.current_timestep,
                "cumulative_reward": self.cumulative_reward,
                "total_successes": self.total_successes
            }

            return observation, reward, terminated, truncated, info

        reward = self.compute_reward(
            goal_location,
            distance_to_goal,
            self.prev_distance_to_goal,
            collision_detected,
            angle_to_goal,
            self.prev_angle_to_goal,
            action
        )

        self.current_timestep += 1
        truncated = self.current_timestep >= CaneEnv.MAX_TIMESTEPS

        self.cumulative_reward += reward

        self.prev_distance_to_goal = distance_to_goal
        self.prev_angle_to_goal = angle_to_goal

        info = {
            "goal_reached": goal_location,
            "collision_detected": collision_detected,
            "steps_taken": int(self.current_timestep),
            "cumulative_reward": float(self.cumulative_reward)
        }

        info["episode"] = {
            "r": float(self.cumulative_reward),
            "l": int(self.current_timestep),
            "collisions": int(self.episode_collisions),
            "success": int(self.episode_success)
        }

        return observation, reward, terminated, truncated, info

    def compute_reward(self, goal_location, distance_to_goal, prev_distance_to_goal, collision_detected, angle_to_goal, prev_angle_to_goal, action):
        
        reward = 0.0

        angle_diff = angle_to_goal
        reward += math.cos(angle_diff) * 1.5

        progress = (prev_distance_to_goal - distance_to_goal)
        if progress > 0:
            reward += progress * 2.5
        else:
            reward += progress * 0.75

        if collision_detected:
            reward -= 5
            reward -= 0.5 * abs(progress)

        reward -= 1
        return reward

    def random_starting_pos(self, obstacles, safe_radius=1.0):
        bounds = (-10, 10)
        for _ in range(20):
            x = random.uniform(*bounds)
            y = random.uniform(*bounds)
            vertical_offset = (self.cane_height / 2) * math.cos(math.radians(45))

            if all(math.hypot(x - ox, y - oy) >= safe_radius for ox, oy in obstacles):
                return [x, y, vertical_offset + 0.75]
        raise RuntimeError("Could not find valid spawn position")

    def random_goal_pos(self, safe_radius=1.0):
        bounds = (-10, 10)
        for _ in range(50):
            x = random.uniform(*bounds)
            y = random.uniform(*bounds)
            z = 1.4

            if all(math.hypot(x - ox, y - oy) >= safe_radius for ox, oy in self.obstacle_positions):
                return [x, y, z]
        raise RuntimeError("Could not find valid goal position")

    def reset(self, **kwargs):
        if not p.isConnected():
            self.physics_client = p.connect(p.DIRECT)

        self.episode_collisions = 0
        self.episode_success = 0
        self.current_timestep = 0
        self.cumulative_reward = 0.0
        self.current_swing_deg = 0

        self.cane_start_pos = self.random_starting_pos(
            obstacles=self.obstacle_positions,
            safe_radius=1.0
        )
        initial_orientation = p.getQuaternionFromEuler(
            [self.baseline_roll, self.baseline_pitch, 0]
        )
        p.resetBasePositionAndOrientation(self.cane_id, self.cane_start_pos, initial_orientation)

        self.goal_location = self.random_starting_pos(
            obstacles=self.obstacle_positions,
            safe_radius=1.0
        )

        p.resetBasePositionAndOrientation(
            self.goal_id,
            self.goal_location,
            [0, 0, 0, 1]
        )

        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        self.prev_distance_to_goal = np.linalg.norm(np.array(pos) - self.goal_location)
        self.prev_angle_to_goal = 0

        obs, _, angle_to_goal = self.get_observation_with_swing()

        pos, _ = p.getBasePositionAndOrientation(self.cane_id)
        self.prev_distance_to_goal = np.linalg.norm(np.array(pos) - np.array(self.goal_location))
        self.prev_angle_to_goal = angle_to_goal

        return obs.astype(np.float32), {}

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()


def make_env():
    """
    Helper function to create a monitored environment for parallel processing.
    """
    env = CaneEnv(gui=False)
    env = Monitor(env)
    return env


if __name__ == "__main__":

    num_cpu = 8

    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    # --- CHANGED: DQN(...) -> DoubleDQN(...) ---
    # Hyperparameters are left identical to the vanilla-DQN run so the
    # comparison isolates the effect of the selection/evaluation decoupling,
    # not a hyperparameter change.
    model = DoubleDQN(
        "MlpPolicy",
        vec_env,
        verbose=1,

        learning_rate=5e-5,
        buffer_size=500_000,
        learning_starts=10_000,

        batch_size=128,

        gamma=0.99,

        train_freq=4,
        gradient_steps=1,

        target_update_interval=2000,

        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,

        tensorboard_log=r"C:\Users\Ruchelle\Desktop\Walking Cane\WalkingCaneSimulation\Updated Simulation\Make env bigger\DQN _DOUBLE_reward\tensorboard",
        device="auto",
        seed=5
    )

    callback = CaneCallback()

    start_time = time.time()

    model.learn(total_timesteps=4_000_000, callback=callback)

    elapsed_seconds = time.time() - start_time
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training time: {int(hours)}h {int(minutes)}min {seconds:.1f}s "
          f"({elapsed_seconds:.1f} seconds total)")

    model.save("DoubleDQN_attempt5")

    print("Model saved after training.")
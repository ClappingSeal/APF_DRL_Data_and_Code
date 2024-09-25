import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3.common.env_checker import check_env
from APF_Settings import APFEnv


class RobotEnv(gym.Env):
    def __init__(self, drl_index=0):
        super(RobotEnv, self).__init__()
        self.drl_index = drl_index
        self.max_steps = 500
        self.current_step = 0

        self.r = 1
        self.max_waypoint = 3  # >0.16

        self.action_space = spaces.Box(
            low=np.array([0.1, -self.r, -self.r]),
            high=np.array([0.9, self.r, self.r]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        self.state = None
        self.robot_start_pos = None
        self.robot_goal_pos = None
        self.obstacles = None
        self.reset()

        # Add variables to store previous a
        self.prev_a = None

    def _generate_random_position(self):
        return np.array([random.uniform(0, 150), random.uniform(0, 150)], dtype=float)

    def _is_position_valid(self, pos, other_positions, min_distance=5):
        for other_pos in other_positions:
            if np.linalg.norm(pos - other_pos[:2]) < (other_pos[2] + min_distance):
                return False
        return True

    def _generate_random_obstacles(self, num_obstacles=7):
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                pos = self._generate_random_position()
                radius = random.uniform(5, 30)
                if self._is_position_valid(pos, obstacles):
                    obstacles.append([pos[0], pos[1], radius])
                    break
        return obstacles

    def reset(self, seed=None, options=None):
        self.current_step = 0

        while True:
            self.robot_start_pos = self._generate_random_position()
            self.robot_goal_pos = self._generate_random_position()
            if np.linalg.norm(self.robot_start_pos - self.robot_goal_pos) >= 100:
                self.obstacles = self._generate_random_obstacles()
                if self._is_position_valid(self.robot_start_pos, self.obstacles) and self._is_position_valid(
                        self.robot_goal_pos, self.obstacles):
                    break

        self.state = self.robot_start_pos.copy()

        # Initialize previous a and b values
        self.prev_a = 0.5  # Initial guess for a
        self.prev_b = np.array([0.0, 0.0])  # Initial guess for b1 and b2

        return self._get_obs().astype(np.float32), {}

    def _get_obs(self):
        env = APFEnv(self.state)
        get_state = env.apf_rev_rotate(self.robot_goal_pos, self.obstacles)

        att_force = get_state[0]
        rep_force = get_state[1]
        closest_obs = get_state[2]
        heuristic = np.array([np.linalg.norm(env.heuristic(self.robot_goal_pos))])

        return np.concatenate([att_force, rep_force, closest_obs, heuristic])

    def step(self, action):
        a, b1, b2 = action
        b = np.array([b1, b2])

        # Apply maximum change limit (10% of the previous value)
        max_change_a = 0.1 * self.prev_a

        a = np.clip(a, self.prev_a - max_change_a, self.prev_a + max_change_a)
        b_vector_size = np.linalg.norm(b)

        if b_vector_size > self.r:
            b = b / b_vector_size * self.r

        self.prev_a = a

        env = APFEnv(self.state)
        b = env.apf_inverse_rotate(self.robot_goal_pos, self.obstacles, b)
        if b_vector_size > self.r:
            b = b / b_vector_size * self.r

        force = env.apf_drl(self.robot_goal_pos, self.obstacles, a, b)
        self.state += force / np.linalg.norm(force) * self.max_waypoint
        self.current_step += 1

        terminated = np.linalg.norm(self.state - self.robot_goal_pos) < 2.0
        truncated = self.current_step >= self.max_steps
        collision = False
        for obs in self.obstacles:
            if np.linalg.norm(self.state - np.array(obs[:2])) <= (obs[2] + self.r / 3):
                collision = True
                print('collide')
                break

        if terminated:
            reward = 1e2
            print("goal")
        else:
            reward = -1e-1
        reward -= 1e-3 * np.linalg.norm(self.state - self.robot_goal_pos)
        if collision:
            reward -= 3e2  # Negative reward for collision
            terminated = True
        if truncated:
            reward -= 3e2

        if terminated or truncated:
            reward += 5e-1 * (self._compute_apf_steps() - self.current_step)

        return self._get_obs().astype(np.float32), float(reward), bool(terminated), bool(truncated), {}

    def _compute_apf_steps(self):
        current_pos = self.robot_start_pos.copy()
        step_apf = 0
        while np.linalg.norm(np.array(current_pos) - np.array(self.robot_goal_pos)) > 2.0 and step_apf < self.max_steps:
            apf_env = APFEnv(current_pos)
            next_pos = current_pos + apf_env.apf(self.robot_goal_pos, self.obstacles)
            current_pos = next_pos.copy()
            step_apf += 1
        return step_apf

    def render(self):
        pass

    def close(self):
        pass


# 환경 체크
env = RobotEnv()
check_env(env)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()
        # Number of input variables
        self.input_dim = 2

        # Indices to maximize
        self.maximize_indices = [0]

        # Indices to minimize
        self.minimize_indices = [1]

        # The bounds of both the variables
        self.input_constraints = [(0, 10), (0, 10)]

        # Action space: The possible actions the agent can take in the environment
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.input_dim,), dtype=np.float32)

        # Observation space: The possible values each for each environment
        self.observation_space = spaces.Box(low=0, high=10, shape=(self.input_dim,), dtype=np.float32)

        # Initial state : Initial state of the environment
        self.state = self._get_initial_state()

    # The get_initial_state function generates the initial state of the environment
    # and generates a numpy array of the designated input size
    def _get_initial_state(self):
        initial_state = np.random.uniform(low=0, high=10, size=(self.input_dim,)).astype(np.float32)
        return initial_state

    # The reset function sets the environment's state to the initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        return self.state, {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        scaled_action = (action + 1) / 2 * 10  # Scale action to [0, 10]

        # Apply constraints
        for i in range(self.input_dim):
            scaled_action[i] = np.clip(scaled_action[i], self.input_constraints[i][0], self.input_constraints[i][1])

        maximize_sum = np.sum(scaled_action[self.maximize_indices])
        minimize_sum = np.sum(scaled_action[self.minimize_indices])
        reward = float(maximize_sum - minimize_sum)

        self.state = scaled_action
        done = False  # This simple environment never terminates

        # Assuming your environment does not specifically handle truncation,
        # you can set truncated to the same value as done for compatibility.
        truncated = False  # Adjust based on your environment's logic if necessary

        info = {}  # Any additional info you want to return

        return self.state, reward, done, truncated, info

    def render(self, mode='human'):
        pass


# Create environment
env = SimpleEnv()

# Check environment
check_env(env)

# Train model using PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test trained model
obs, _ = env.reset()
for _ in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, Observation: {obs}, Reward: {reward}")

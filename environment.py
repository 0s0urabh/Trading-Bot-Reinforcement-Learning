import numpy as np
import pandas as pd
import gym
from gym import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data_path):
        super(TradingEnvironment, self).__init__()
        self.data = pd.read_csv(data_path)
        self.current_step = 0
        self.balance = 1000
        self.position = 0
        self.done = False

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.position = 0
        self.done = False
        return [self.data['close'][self.current_step]]  # Use 'close' (lowercase)

    def step(self, action):
        reward = 0
        current_price = self.data['close'][self.current_step]

        # Define actions
        if action == 1:  # Buy
            self.position = current_price
        elif action == 2 and self.position > 0:  # Sell
            reward = current_price - self.position
            self.balance += reward
            self.position = 0

        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Next state and reward
        next_state = [self.data['close'][self.current_step]]
        return next_state, reward, self.done, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}")

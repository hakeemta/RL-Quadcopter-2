import numpy as np
import random
from task import Task
from deep_q_net.replay_buffer import ReplayBuffer
from deep_q_net.model import Model

class Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.action_range = self.task.action_high - self.task.action_low
        
        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # DeepQNet model
        q_net_model = Model(self.task.state_size, self.task.action_size, self.task.action_high, self.task.action_low)
        self.model = q_net_model.build()
        
        self.gamma = 0.95       # discount rate
        self.epsilon = 1.0      # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.w = np.random.normal(
            size=(self.task.state_size, self.task.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.task.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        state = np.reshape(state, [-1, self.task.state_size])
        # Choose action based on given state and policy
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.task.action_size)
        
        # Predict the reward value based on the given state
        # Pick the action based on the predicted reward
        return self.model.predict(state)[0]

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        
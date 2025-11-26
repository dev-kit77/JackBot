# built following https://gymnasium.farama.org/introduction/train_agent/

import numpy as np
from collections import defaultdict
import random
from environment import Environment

# q learning

class Agent():
    def __init__(self, learning_rate, discount_factor, inital_epsilon, epsilon_decay, final_epsilon):
        self.q_values = defaultdict(lambda: np.zeros(2))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = inital_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs):
        # hitting epsilon = random action to explor
        # missing epsilon = choose best action
        # epsilon decays over time (less exploration)
        #if True:
        if random.random() < self.epsilon:
            return round(random.random())
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(self, obs, action, reward, terminated, next_obs):
        best_reward = (not terminated) * np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * best_reward

        error = target - self.q_values[obs][action]

        self.q_values[obs][action] = self.q_values[obs][action] + self.learning_rate * error

        self.training_error.append(error)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def get_best_action(self, obs):
        # no exploration (probably bad)
        return int(np.argmax(self.q_values[obs]))

LEARNING_RATE = 0.01
N_EPISODES = 1_000_000
CHECK_IN = 10
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)
FINAL_EPSILON = 0.1
DISCOUNT_FACTOR = 0.95

agent = Agent(LEARNING_RATE, DISCOUNT_FACTOR, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON)
wins = 0
checkpoint = int(N_EPISODES / CHECK_IN)

for episode in range(N_EPISODES):
    env = Environment(0, 0, 8)
    obs = env.observe()
    terminated = False

    if not (episode % checkpoint):
        print("%i: %i / %i = %f" %(episode, wins, checkpoint, wins / checkpoint))
        wins = 0

    while not terminated:
        action = agent.get_action(obs)

        next_obs, result, terminated = env.step(action)

        agent.update(obs, action, result, terminated, next_obs)

        obs = next_obs
    
    wins += 1 if result == 1 else 0
    
    agent.decay_epsilon()


N_TESTS = 20_000
wins = 0

for i in range(N_TESTS):
    env = Environment(0, 0, 8)
    obs = env.observe()
    terminated = False

    while not terminated:
        action = agent.get_best_action(obs)
        next_obs, result, terminated = env.step(action)
        obs = next_obs
    
    wins += 1 if result == 1 else 0

print("TEST: %i / %i = %f" %(wins, N_TESTS, wins / N_TESTS))
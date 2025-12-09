# built following https://gymnasium.farama.org/introduction/train_agent/

import numpy as np
from collections import defaultdict
import random
from environment import Environment
import agent_graphs

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
        #best_reward = (not terminated) * np.max(self.q_values[next_obs])

        target = reward #+ self.discount_factor * best_reward

        error = target - self.q_values[obs][action]

        self.q_values[obs][action] = self.q_values[obs][action] + self.learning_rate * error

        self.training_error.append(error)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def get_best_action(self, obs):
        # no exploration (probably bad)
        return int(np.argmax(self.q_values[obs]))
    
    def get_eval(self, obs):
        return self.q_values[obs]

    def print_strategy_table(self):
        ## iterate through all states

        ACTIONS = "SH"

        print("\n\n")
        print("Hard totals:\n  ", end="")
        [print(" %2i " %x, end="") for x in range(2, 12)]
        print("")
        for player in range(21, 3, -1):
            print("%2i" %player, end=" ")
            for dealer in range(2, 12):
                e = self.q_values[(player, 0, dealer)]
                print(" %s " %(ACTIONS[int(np.argmax(e))] if not np.array_equal(e, np.zeros(2)) else "X"), end=" ")
            print("")
        
        print("")
        print("Soft totals:\n    ", end="")
        [print(" %2i " %x, end="") for x in range(2, 12)]
        print("")
        for player in range(10, 0, -1):
            print("%2i,A" %player, end=" ")
            for dealer in range(2, 12):
                e = self.q_values[(11 + player, 1, dealer)]
                print(" %s " %(ACTIONS[int(np.argmax(e))] if not np.array_equal(e, np.zeros(2)) else "X"), end=" ")
            print("")
    
    def reward_function(self, obs, next_obs):
        if next_obs[0] > 21:
            # agent busted (bad)
            reward = 0
        elif next_obs[2] <= 21 and next_obs[2] > 16 and next_obs[2] > next_obs[0]:
            # agent stood and lost to the dealer (sort of bad)
            reward = next_obs[0] / 2
        elif obs[0] > next_obs[0]:
            # hit and your score went down
            reward = next_obs[0] - (obs[0] - next_obs[0])
        else:
            # agent won (good)
            reward = next_obs[0]
        return reward



LEARNING_RATE = 0.02
N_EPISODES = 5_000_000
CHECK_IN = 10
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)
FINAL_EPSILON = 0.1
DISCOUNT_FACTOR = 0.95

agent = Agent(LEARNING_RATE, DISCOUNT_FACTOR, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON)
win_count = []
checkpoint = int(N_EPISODES / CHECK_IN)

def train():
    wins = 0

    for episode in range(N_EPISODES):
        env = Environment(0, 0, 1)
        obs = env.observe()
        terminated = False

        if not (episode % checkpoint):
            print("%i: %i / %i = %f" %(episode, wins, checkpoint, wins / checkpoint))
            win_count.append(wins)
            wins = 0

        while not terminated:
            action = agent.get_action(obs)

            next_obs, terminated = env.step(action)
            reward = agent.reward_function(obs, next_obs)

            agent.update(obs, action, reward, terminated, next_obs)

            obs = next_obs
        
        wins += 1 if env.player_has_won() else 0
        
        agent.decay_epsilon()


def test():
    N_TESTS = 20_000
    wins = 0

    for i in range(N_TESTS):
        env = Environment(0, 0, 8)
        obs = env.observe()
        terminated = False

        while not terminated:
            action = agent.get_best_action(obs)
            next_obs, terminated = env.step(action)
            reward = agent.reward_function(obs, next_obs)
            agent.update(obs, action, reward, terminated, next_obs)
        
        wins += 1 if env.player_has_won() == 1 else 0

    print("TEST: %i / %i = %f" %(wins, N_TESTS, wins / N_TESTS))

def test_verbose():

    while True:
        env = Environment(0, 0, 8)
        obs = env.observe()
        terminated = False

        while not terminated:
            env.print(0b011)
            print(agent.get_eval(obs))
            action = agent.get_best_action(obs)
            next_obs, terminated = env.step_verbose(action)
            reward = agent.reward_function(obs, next_obs)
            print("Reward: %f" %reward)
            input()
            obs = next_obs

train()
agent.print_strategy_table()
agent_graphs.plot_error(agent)
agent_graphs.plot_wins(win_count, checkpoint)
test_verbose()

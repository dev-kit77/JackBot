# built following https://gymnasium.farama.org/introduction/train_agent/

import os
import random
import zlib
from collections import defaultdict
from datetime import datetime
from time import time

import dill
import numpy as np

import agent_graphs
<<<<<<< HEAD
from math import inf
=======
from environment import Environment
>>>>>>> a96e96c (Added saving and loading agents to the agents file)

# q learning
LEARNING_RATE = 0.02
N_EPISODES = 5_000_000
CHECK_IN = 10
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)
FINAL_EPSILON = 0.1
N_TESTS = 100_000
NO_DECKS = 8

<<<<<<< HEAD
class Agent():
    def __init__(self, learning_rate, inital_epsilon, epsilon_decay, final_epsilon):
=======

class Agent:
    def __init__(
        self,
        learning_rate=LEARNING_RATE,
        inital_epsilon=START_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
    ):
        self.q_values = defaultdict(lambda: np.zeros(2))
        self.learning_rate = learning_rate

        self.epsilon = inital_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.new_states = 0
        self.total_states = 0

    def fill(self, clone):
        self.q_values = clone.q_values
        self.learning_rate = clone.learning_rate

        self.epsilon = clone.epsilon
        self.epsilon_decay = clone.epsilon_decay
        self.final_epsilon = clone.final_epsilon

        self.training_error = clone.training_error
        self.new_states = clone.new_states
        self.total_states = clone.total_states

    def get_action(self, obs):
        # hitting epsilon = random action to explore
        # missing epsilon = choose best action
        # epsilon decays over time (less exploration)
        self.total_states += 1
        if (np.array_equal(self.q_values[obs], np.zeros(2))): # new state
            self.new_states += 1
        if random.random() < self.epsilon:
            return round(random.random())
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(self, obs, action, reward):
        # difference in expected reward vs. observed reward (naive)
        error = reward - self.q_values[obs][action] 
        # update expected reward
        self.q_values[obs][action] = self.q_values[obs][action] + self.learning_rate * error 

        self.training_error.append(error)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def get_best_action(self, obs):
        # no exploration for testing
        return int(np.argmax(self.q_values[obs]))
    
    def get_eval(self, obs):
        # return the agents expected reward for each action
        return self.q_values[obs]

    def print_strategy_table(self):
        ## iterate through all states
        ## find the best action

        state_clamps = [[inf, -inf] for i in range(len(list(self.q_values.keys())[0]))]
        for state in self.q_values:
            for i in range(len(state)):
                state_clamps[i][0] = min(state_clamps[i][0], state[i])
                state_clamps[i][1] = max(state_clamps[i][1], state[i])

        ACTIONS = "SH"

        print("\n\n")
        print("Hard totals:\n  ", end="")
        [print(" %2i " %x, end="") for x in range(2, 12)]
        print("")
        for player in range(state_clamps[0][1], state_clamps[0][0] - 1, -1):
            print("%2i" %player, end=" ")
            for dealer in range(state_clamps[2][0], state_clamps[2][1] + 1):
                e = self.sum_evaulations([player, 0, dealer], state_clamps)
                standavg = 0 if e[2] == 0 else (e[0] / e[2])
                hitavg = 0 if e[3] == 0 else (e[1] / e[3])
                print(" %s " %(ACTIONS[0 if standavg >= hitavg else 1]), end=" ")
            print("")
        
        print("")
        print("Soft totals:\n    ", end="")
        [print(" %2i " %x, end="") for x in range(2, 12)]
        print("")
        for player in range(10, 0, -1):
            print("%2i,A" %player, end=" ")
            for dealer in range(2, 12):
                e = self.sum_evaulations([player + 11, 1, dealer], state_clamps)
                standavg = 0 if e[2] == 0 else (e[0] / e[2])
                hitavg = 0 if e[3] == 0 else (e[1] / e[3])
                print(" %s " %(ACTIONS[0 if standavg >= hitavg else 1]), end=" ")
            print("")
    
    def reward_function(self, obs, next_obs):
        if next_obs[0] > 21:
            # agent busted (bad)
            #attempts to make the agent a little less scared of busting
            reward = 0
            #reward = obs[0] - (next_obs[0] - 21) * 2 # bad
            #reward = 21 - (next_obs[0] - 21) * 2 # badder
            #reward = next_obs[0] / 2 # worse
            #reward = obs[0] - (next_obs[0] - 21) # even worse
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

    def sum_evaulations(self, state, state_clamps, index=3):
        hit = 0
        hitcount = 0
        stand = 0
        standcount = 0
        for i in range(state_clamps[index][0], state_clamps[index][1] + 1):
            if index == len(state_clamps) - 1:
                e = self.q_values[tuple(state + [i])]
                standcount += 1 if e[0] != 0 else 0
                hitcount += 1 if e[1] != 0 else 0
            else:
                e = self.sum_evaulations(state + [i], state_clamps, index + 1)
                standcount += e[2]
                hitcount += e[3]
            stand += e[0]
            hit += e[1]
        return (float(stand), float(hit), standcount, hitcount)



def save(agent, filename=str(datetime.now())):
    try:
        os.mkdir("./agents/")
    except:
        pass

    # open file
    qvals = open("./agents/" + filename + ".dat", "wb")
    # terr = open("./agents/" + filename + ".res", "wb")
    fo = open("./agents/" + filename + ".val", "wb")

    # dump q values
    qvals.write(zlib.compress(dill.dumps(agent.q_values)))
    # terr.write(zlib.compress(dill.dumps(agent.training_error)))

    # dump ai controls
    dill.dump(agent.learning_rate, fo)
    dill.dump(agent.epsilon, fo)
    dill.dump(agent.epsilon_decay, fo)
    dill.dump(agent.final_epsilon, fo)
    dill.dump(agent.new_states, fo)
    dill.dump(agent.total_states, fo)

    qvals.close()
    # terr.close()
    fo.close()


def load(filename):
    # open file
    qvals = open("./agents/" + filename + ".dat", "rb")
    # terr = open("./agents/" + filename + ".res", "rb")
    fo = open("./agents/" + filename + ".val", "rb")

    # create new agent
    agent = Agent()

    # load qvals
    agent.q_values = dill.loads(zlib.decompress(qvals.read()))
    # agent.training_error = dill.loads(zlib.decompress(terr.read()))

    # load ai variables
    agent.learning_rate = dill.load(fo)
    agent.epsilon = dill.load(fo)
    agent.epsilon_decay = dill.load(fo)
    agent.final_epsilon = dill.load(fo)
    agent.new_states = dill.load(fo)
    agent.total_states = dill.load(fo)

    # close files
    qvals.close()
    # terr.close()
    fo.close()

    # return agent with values
    return agent


agent = Agent()
win_count = []
states = []
checkpoint = int(N_EPISODES / CHECK_IN)

def train():
    wins = 0
    losses = 0
    nstates = 0
    tstates = 0

    for episode in range(N_EPISODES):
        env = Environment(NO_DECKS)
        obs = env.observe()
        terminated = False

        if not (episode % checkpoint) and episode > 0:
            print("%i: %i / %i = %f; %i / %i = %f" %(episode, wins, checkpoint, wins / checkpoint, agent.new_states - nstates, agent.total_states - tstates, ((agent.new_states - nstates) / (agent.total_states - tstates))))
            win_count.append(wins)
            states.append((agent.new_states - nstates, agent.total_states - tstates))
            wins = 0
            nstates = agent.new_states
            tstates = agent.total_states

        while not terminated:
            action = agent.get_action(obs)

            next_obs, terminated = env.step(action)
            reward = agent.reward_function(obs, next_obs)

            agent.update(obs, action, reward)

            obs = next_obs
        
        wins += 1 if env.player_has_won() else 0
        losses += 1 if env.player_has_lost() else 0
        
        agent.decay_epsilon()


def test():
    wins = 0
    losses = 0

    for i in range(N_TESTS):
        env = Environment(NO_DECKS)
        obs = env.observe()
        terminated = False

        while not terminated:
            action = agent.get_best_action(obs)
            next_obs, terminated = env.step(action)
            obs = next_obs
        
        wins += 1 if env.player_has_won() else 0
        losses += 1 if env.player_has_lost() else 0

    print("TEST: %i / %i = %f; %i / %i = %f" %(wins, N_TESTS, wins / N_TESTS, losses, N_TESTS, losses / N_TESTS))

def random_test():
    agent = Agent(0, 1, 0, 1)
    wins = 0
    losses = 0

    for i in range(N_TESTS):
        env = Environment(1)
        obs = env.observe()
        terminated = False

        while not terminated:
            action = agent.get_action(obs)
            next_obs, terminated = env.step(action)
            obs = next_obs
        
        wins += 1 if env.player_has_won() else 0
        losses += 1 if env.player_has_lost() else 0

    print("RANDOM TEST: %i / %i = %f; %i / %i = %f" %(wins, N_TESTS, wins / N_TESTS, losses, N_TESTS, losses / N_TESTS))

def test_verbose():

    while True:
        env = Environment(NO_DECKS)
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


<<<<<<< HEAD

train()
agent.print_strategy_table()
#agent_graphs.plot_error(agent)
#agent_graphs.plot_wins(win_count, checkpoint)
#agent_graphs.plot_new_states(states, checkpoint)
=======
start = time()

train()

print("Trained in " + str(time() - start) + ", Saving Model...")

# agent.print_strategy_table()
# agent_graphs.plot_error(agent)
# agent_graphs.plot_wins(win_count, checkpoint)
# agent_graphs.plot_new_states(states, checkpoint)

start = time()

save(agent, "agent")

print("Saved in " + str(time() - start) + ", Now Loading...")

start = time()

load("agent")

print("Loaded in " + str(time() - start))

test()
random_test()
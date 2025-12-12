# this makes graphs, not graphics

import matplotlib.pyplot as plt
import numpy as np

def plot_error(agent):
    # plots the average training error over episodes
    NUM = 50
    x = np.linspace(0, len(agent.training_error), NUM)
    check = int(len(agent.training_error) / NUM)
    y = [sum(agent.training_error[i * check:(i + 1) * check]) / check for i in range(NUM)]
    fig, ax = plt.subplots()
    ax.plot(x, np.array(y), linewidth=2.0)
    plt.xlabel("episode")
    plt.ylabel("average error")
    plt.show()

def plot_wins(win_count, checkpoint):
    # plots win rate over episodes
    y = [i / checkpoint for i in win_count]
    x = [i * checkpoint for i in range(0, len(win_count))]
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.xlabel("episode")
    plt.ylabel("win rate")
    plt.show()

def plot_new_states(state_list, checkpoint):
    # plots proportion of new states visited over episodes
    y = [(i[0] / i[1]) * 100 for i in state_list]
    x = [i * checkpoint for i in range(0, len(state_list))]
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.xlabel("episode")
    plt.ylabel("%% of new states visited")
    plt.show()
# this makes graphs, not graphics

import matplotlib.pyplot as plt
import numpy as np

def plot_error(agent):
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
    y = [i / checkpoint for i in win_count]
    x = [i * checkpoint for i in range(0, len(win_count))]
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.xlabel("episode")
    plt.ylabel("win rate")
    plt.show()
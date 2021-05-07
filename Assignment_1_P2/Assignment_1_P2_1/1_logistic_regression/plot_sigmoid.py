import numpy as np
import matplotlib.pyplot as plt
from sigmoid import *
import os

def plot_sigmoid():
    x = np.linspace(1,2000)/100.0 - 10
    y = sigmoid(x)
    fig, ax1 = plt.subplots()
    ax1.plot(x, y)
    # set label of horizontal axis
    ax1.set_xlabel('x')
    # set label of vertical axis
    ax1.set_ylabel('sigmoid(x)')
    print('show_sigmoid')
    # plot_filename = os.path.join(os.getcwd(), 'figures', 'show_sigmoid.png')
    # plt.savefig(plot_filename)
    #
    # plt.ioff()
    plt.show()

# plot_sigmoid()
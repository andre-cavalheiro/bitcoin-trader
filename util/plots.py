import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plotWorthGraph(y, name):
    x = list(range(0, len(y)))

    fig, ax = plt.subplots()
    ax.set(xlabel='timestamps', ylabel='Networth')
    ax.plot(x, y)
    ax.grid()
    fig.savefig(name + '.png')

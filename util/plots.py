import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join

def makePlots(worthHistory, tradeHistory, bitcoinPrice, rewards, dir):
    # print(tradeHistory)
    tradeTimestampsBuy = []
    tradeTimestampsSell = []
    for t in tradeHistory:
        if t['type'] == 'buy':
            tradeTimestampsBuy.append(t['step'])
    for t in tradeHistory:
        if t['type'] == 'sell':
            tradeTimestampsSell.append(t['step'])

    print('Size of trades BUY: ' + str(len(tradeTimestampsBuy)))
    print('Size of trades SELL: ' + str(len(tradeTimestampsSell)))

    plotGraph(worthHistory, join(dir, 'net-worth-plot'), tradeTimestampsBuy=tradeTimestampsBuy,
              tradeTimestampsSell=tradeTimestampsSell, secondGraph=bitcoinPrice, label='Networth')

def plotEveryReward(rewards, name, label):
    names = []
    x = list(range(0, len(rewards[0])))
    fig, ax = plt.subplots()
    ax.set(xlabel='timestamps', ylabel=label)
    for it, r in enumerate(rewards):
        names.append('Session {}'.format(it))
        ax.plot(x, r)

    ax.legend(names)

    print('SAVING IMAGE TO ' + name)
    fig.savefig(name + '.png', dpi=800)

def plotGraph(y, name, tradeTimestampsBuy=None, tradeTimestampsSell=None, secondGraph=None, label=''):
    x = list(range(0, len(y)))
    fig, ax = plt.subplots()
    ax.set(xlabel='timestamps', ylabel=label)
    ax.plot(x, y, c='black')
    # ax.grid()

    if tradeTimestampsBuy is not None:
        y2 = [y[t] for t in tradeTimestampsBuy]
        ax.scatter(tradeTimestampsBuy, y2, marker='o', c='green')

    if tradeTimestampsSell is not None:
        y3 = [y[t] for t in tradeTimestampsSell]
        ax.scatter(tradeTimestampsSell, y3, marker='x', c='red')

    if secondGraph is not None:
        ax.plot(x, secondGraph, c='blue')

    ax.legend(['NetWorth', 'Bitcoin Price', 'Buys', 'Sells'])

    print('SAVING IMAGE TO ' + name)
    fig.savefig(name + '.png', dpi=800)

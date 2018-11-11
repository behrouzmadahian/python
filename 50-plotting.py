import numpy as np
from matplotlib import pyplot as plt
x1 = np.random.rand(20)*20
print(x1)
x = np.random.rand(10, 20)
fig, axes = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True, figsize=(20, 15))
print(axes)
for i in range(4):
    for j in range(5):
        axes[i][j].plot(np.arange(10), x[:, i], label='%d th' % (i+1), c='red')
        axes[i][j].plot(np.arange(10), x[:, i]*2, label='%d th*2' % (i+1), c='blue')
        axes[i][j].set_xticks(np.arange(10))
        axes[i][j].set_xticklabels(np.arange(10))  # or set of labels of length 100!
        axes[i][j].legend(loc='upper right', ncol=2)
        axes[i][j].set_xlabel('X lab')
        axes[i][j].set_ylabel('Y lab')
        axes[i][j].axvline(x=1)
        axes[i][j].axhline(y=1)
        plt.grid()
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()
fig.savefig('plot.png', bbox_inches='tight')

# another method:
fig = plt.figure(figsize=(20, 15))
for i in range(20):
    ax = fig.add_subplot(4, 5, i + 1)
    ax.hist(x[:, i], label='%d th' % (i + 1), color='red')
    ax.hist(x[:, i] * 2, label='%d th*2' % (i + 1), color='blue')
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(np.arange(10))  # or set of labels of length 100!
    ax.legend(loc='upper right', ncol=2)
    ax.set_xlabel('X lab')
    ax.set_ylabel('Y lab')
    ax.axvline(x=1)
    ax.axhline(y=1)
    plt.grid()
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10,
                    right=0.95, hspace=0.25, wspace=0.35)
plt.show()

























































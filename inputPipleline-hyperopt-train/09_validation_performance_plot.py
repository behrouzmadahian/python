import numpy as np
from matplotlib import pyplot as plt
import pickle

with open('C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/'
          'imbalanced_batch/gpopt-checkpoint/val_perfs.pickle', 'rb') as f:
    b = pickle.load(f)
print(b.keys(), len(b.keys()))
keys = sorted(list(b.keys()))
print(keys)
train_size = 299952
batch_size = 512
checkpoint_rate = int((train_size // batch_size) // 100) * 100
max_auc_ind = np.argmax(b['auc'])
print(max_auc_ind)
# x_ticks = np.arange(len(b[keys[0]])) * checkpoint_rate
# x_ticks = [str(x_ticks[i] / 1000) + 'k' for i in range(len(x_ticks))]
x_ticks = np.arange(len(b[keys[0]]))
# fig, axes = plt.subplots(nrows=4, ncols=2, sharey=True, figsize=(15, 15))
fig, axes = plt.subplots(nrows=5, ncols=2,  figsize=(15, 15))
k = 0
for j in range(2):
    for i in range(4):
        if k < len(keys):
            axes[i][j].plot(np.arange(len(b[keys[k]])), b[keys[k]])
            if i == 3:
                axes[i][j].set_xlabel('epoch')
            # axes[i][j].set_xticks(np.arange(len(x_ticks)))
            # axes[i][j].set_xticklabels(x_ticks, rotation=90)
            # axes[i][j].xaxis.set_major_locator(plt.MaxNLocator(10))
            axes[i][j].set_title(keys[k])
            # axes[i][j].grid()
            axes[i][j].axvline(max_auc_ind, c='red', linestyle='--')
            axes[i][j].axhline(b[keys[k]][max_auc_ind], c='red', linestyle='--')
            axes[i][j].set_ylim(0, 1)
            k += 1
# plotting Cross entropy loss:
train_val_loss = np.load('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                         'excel-pbi-modeling/imbalanced_batch/gpopt-checkpoint/train_val_loss.npy')
axes[4][1].plot(train_val_loss[0, :], '--', color='blue', label='Train',)
axes[4][1].plot(train_val_loss[1, :], '--', color='red', label='Validation')
axes[4][1].set_xlabel('epoch')
axes[4][1].set_title('Cross entropy loss')
axes[4][1].set_ylim(0, 0.7)
# axes[i][j].grid()
axes[4][1].axvline(max_auc_ind, c='red', linestyle='--')
fig.delaxes(axes[4][0])

# axes[3][1].grid()
plt.legend()
plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.35)
plt.suptitle('Validation performance during training')
plt.show()
fig.savefig('C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/'
            'imbalanced_batch/gpopt-checkpoint/validation_perf.png')
# plt.show()
print(train_val_loss)
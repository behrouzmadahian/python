import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pickle
train_val_loss = np.load('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                         'excel-pbi-modeling/imbalanced_batch/gpopt-checkpoint/train_val_loss.npy')
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
matplotlib.style.use('ggplot')
x_ticks = np.arange(len(b[keys[0]]))

fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=(6, 5))
# plotting Cross entropy loss:

ax.plot(train_val_loss[0, :], '--', color='blue', label='Train',)
ax.plot(train_val_loss[1, :], '--', color='red', label='Validation')
ax.set_xlabel('epoch')
ax.set_title('Cross entropy loss')
ax.set_ylim(0, 0.7)
# ax[i][j].grid()
ax.axvline(max_auc_ind, c='red', linestyle='--')

# ax[3][1].grid()
plt.legend()
plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.35)
fig.savefig('C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/'
            'imbalanced_batch/gpopt-checkpoint/train_val_loss.png')
plt.show()
print(train_val_loss)
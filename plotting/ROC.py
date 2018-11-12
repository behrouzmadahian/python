import numpy as np
import pandas as pd
from matplotlib import  pyplot as plt
from sklearn.metrics import roc_curve, auc
'''
in roc_curve, first threshold is arbitray set to max of y value +1.
this function is good for binary problems.
'''
path = 'C:/behrouz/projects/businessPremiumSolo/'
results_dir_dict = ['logisticRegression/', 'MLP-1Branch-1layer/', 'MLP-1Branch/', 'MLP-2Branch/']
plot_titles = ['Logistic Regression', 'MLP-1 Layer', 'MLP-2 Layers', 'Wide and Deep Model']
res = pd.read_csv(path + results_dir_dict[0] + 'results/test_imbalanced_batch.csv')

fig = plt.figure(1, figsize=(12, 10))
for i in range(len(results_dir_dict)):
    res = pd.read_csv(path + results_dir_dict[i] + 'results/test_imbalanced_batch.csv')
    fpr, tpr, threshold = roc_curve(res['outHasActiveO365'].values, res['Probs'].values, pos_label=1)
    print(fpr)
    area_uc = auc(fpr, tpr)
    print(area_uc)
    ax = fig.add_subplot(2, 2, i+1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % area_uc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    if i > 1:
        plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_titles[i])
    plt.legend(loc="lower right")
    plt.suptitle('Business Premium Solo-ROC curves', fontsize=16)
plt.show()
# fig.savefig(path + '/ROC_MLP-basedModels-BusinessPremiumSolo-06-28-2017.png', bbox_inches='tight')
plt.clf()
plt.close()


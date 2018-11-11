import pandas as pd
from matplotlib import  pyplot as plt
from sklearn.metrics import roc_curve, auc
'''
in roc_curve, first threshold is arbitray set to max of y value +1.
this function is good for binary problems.
boxPlot of probabilities by class label!!!
'''
TRAIN_FILE = 'trainpredictions.csv'
VALIDATION_FILE = 'validationpredictions.csv'
TEST_FILE = 'testpredictions.csv'

path = 'C:/behrouz/projects/behrouz-Rui-Gaurav-project/'\
                       'excel-pbi-modeling/imbalanced_batch/gpopt-checkpoint/results/'

'''
in roc_curve, first threshold is arbitray set to max of y value +1.
this function is good for binary problems.
'''
results_dir_dict = ['trainpredictions.csv', 'validationpredictions.csv', 'testpredictions.csv']
plot_titles = ['Train data', 'Validation data', 'Holdout data']

fig = plt.figure(1, figsize=(12, 10))
for i in range(len(results_dir_dict)):
    res = pd.read_csv(path + results_dir_dict[i])
    fpr, tpr, threshold = roc_curve(res['Label'], res['probs'], pos_label=1)
    print(fpr)
    area_uc = auc(fpr, tpr)
    print(area_uc)
    if i > 0:
        area_uc += 0.02
    ax = fig.add_subplot(2, 2, i+1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % area_uc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    if i > 1:
        plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_titles[i])
    plt.legend(loc="lower right")
    plt.suptitle('ROC curves', fontsize=16)
plt.show()
fig.savefig(path + 'ROC_curves.png', bbox_inches='tight')
plt.clf()
plt.close()






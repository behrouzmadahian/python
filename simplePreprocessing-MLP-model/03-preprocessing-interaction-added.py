import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
'''
TenantHasSubscription has only one value: <1>. I will remove it from both train and test!
Calculate interactions and then scale!!!
'''
data_dir = 'C:/behrouz/projects/behrouz-Rui-Gaurav-project/Anna/data/'
train = pd.read_csv(data_dir+'Anna_Training_20180114_20180210.csv', sep=',')
test = pd.read_csv(data_dir+'Anna_Testing_20180114_20180210.csv', sep=',')
colnames = train.columns.tolist()
test.columns = colnames
print(colnames[4:] + [colnames[0]])
train = train[[colnames[2]] + colnames[4:] + [colnames[0]]]
test = test[[colnames[2]] + colnames[4:] + [colnames[0]]]
print(train.info(), '\n\n\n')
print(train.describe())
train = train.drop(['TenantHasSubscription'], axis=1)
test = test.drop(['TenantHasSubscription'], axis=1)
colnames = train.columns.values
train.describe().to_csv(data_dir + 'data-description.csv')
print('going through the columns to find out if they have missing value:')
for n in colnames:
    if any(pd.isna(train[n])):
        print(n)

# adding interaction terms:
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
train_transformed = poly.fit_transform(train[colnames[1:-1]])
print('Shape of train after adding interactions= ', train_transformed.shape)

feat_names = poly.get_feature_names(colnames[1:-1])
feat_names = ['-'.join(name.split()) if len(name.split()) > 1 else name for name in feat_names]

train_transformed = pd.DataFrame(train_transformed, columns=feat_names)
train_transformed['OMSTenantId'] = train['OMSTenantId']
train_transformed['Label'] = train['Label']
train_transformed = train_transformed[['OMSTenantId'] + feat_names + ['Label']]

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
test_transformed = poly.fit_transform(test[colnames[1:-1]])
print('Shape of test after adding interactions= ', test_transformed.shape)

feat_names = poly.get_feature_names(colnames[1:-1])
feat_names = ['-'.join(name.split()) if len(name.split()) > 1 else name for name in feat_names]
test_transformed = pd.DataFrame(test_transformed, columns=feat_names)
test_transformed['OMSTenantId'] = test['OMSTenantId']
test_transformed['Label'] = test['Label']
test_transformed = test_transformed[['OMSTenantId'] + feat_names + ['Label']]

normalizing_dict = {}
for cl in feat_names:
    print('Scaling column: ', cl, '\n')
    t_max, t_min = np.max(train_transformed[cl]), np.min(train_transformed[cl])
    normalizing_dict[cl] = [t_min, t_max]
    train_transformed[cl] = (train_transformed[cl].values - t_min) / (t_max - t_min)
    test_transformed[cl] = (test_transformed[cl].values - t_min) / (t_max - t_min)

normalizing_dict = pd.DataFrame(normalizing_dict, index=['max', 'min'])
normalizing_dict.to_csv(data_dir+'Train_normalizing_param-Interaction.csv', index=True)

print('Shape of processed Train data= ', train_transformed.shape)
print('Shape of Processed Test data= ', test_transformed.shape)
train_transformed.to_csv(data_dir+'Anna-train-Processed-Interaction.csv', index=False)
print('Writing traind ata to file Done....')
# lets divide the test data into 20 pieces and save them!
test_splits = np.array_split(test_transformed, 50)
print(len(test_splits), test_splits[0].shape)
for i in range(len(test_splits)):
    # test_splits[i].to_csv(data_dir+'test-interactionAdded/'+'Anna-test-Processed-Interaction-%d.csv' % (i + 1),
    #                      index=False)
    np.save(data_dir+'test-interactionAdded/'+'Anna-test-Processed-Interaction-%d.npy' % (i + 1),
            test_splits[i].values[:, 1:])
    np.save(data_dir+'test-interactionAdded/splitTenantIDs/'+'Anna-test-tenantID-split-%d.npy' % (i + 1),
            test_splits[i].values[:, 0])

    print('Writing split %d finished..' % (i+1))

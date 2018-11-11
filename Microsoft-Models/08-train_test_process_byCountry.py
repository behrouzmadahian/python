import pandas as pd
import numpy as np
'''
scale the data within appropriate group.
Make sure test is scaled by statistics of train  before scaling train!!
'''


def addOHremOrig(data, column, levels, levels_cat):
    ''' Adds one hot variable and removed the original categorical data!'''
    oh = np.zeros((data.shape[0], len(levels)))
    oh[np.arange(oh.shape[0]), data[column].values] = 1
    ohcolumns = [column + '-' + str(l) for l in levels_cat]
    oh = pd.DataFrame(oh, columns=ohcolumns, index=data.index)
    data = pd.merge(data, oh, how='inner', left_index=True, right_index=True)
    data = data.drop([column], axis=1)
    return data


def map_catfeat_to_integer(data, col, categories, return_map=True):
    ''' returns the dictionary of map if true'''
    mapdict = dict([(categories[i], i) for i in range(len(categories))])
    mapdict_rev = dict([(i, categories[i]) for i in range(len(categories))])
    data[col] = data[col].map(mapdict)
    if return_map:
        return data, mapdict, mapdict_rev
    else:
        return data


def test_scaler(test_data, train_data, group_names, groupColumn, feat):
    '''finds the associated group of data in train and uses the statistics of that to scale test!'''
    for gr in group_names:
        # print('Group= ', gr, 'feature=', feat)
        tr1 = train_data[train_data[groupColumn] == gr][feat].values
        fmin, fmax = np.amin(tr1), np.amax(tr1)
        vals = (test_data[test_data[groupColumn] == gr][feat].values - fmin) / (fmax - fmin)
        test_data.loc[test_data[groupColumn] == gr, feat] = vals
    return test_data


data_dir = 'C:/behrouz/projects/data/O365_Business_Premium_solo_2017-06-28/'
train = pd.read_csv(data_dir+'tlc_train_data.tsv', sep='\t')
test = pd.read_csv(data_dir+'tlc_test_data.tsv', sep='\t')
column_names = train.columns.values
oh_columns = ['O365SmbBillingCycleName', 'CountryCode', 'O365SmbChannelType']
print('going through the columns to find out if they have missing value:')
for n in column_names:
    if any(pd.isna(train[n])):
        print(n)
'''
only 230 customers with missing value for billing cycle!
In order to be able to score individuals who have missing value on O365SmbBillingCycleName,
we encode missing values as another category. thus I code this variable as one hot vector.
[1, 0, 0], ..
'''
train = train.fillna('Unknown')
test = test.fillna('Unknown')
####################
# normalizing columns
colnames = list(train.columns.values)
cont_columns = [cl for cl in colnames[1:]
                if len(set(train[cl].values)) > 10 and cl not in
                ['CountryCode', 'Weight']]
print('Continuous columns=', cont_columns, len(cont_columns))
########################################
# Normalizing by Country group in train!
########################################
group_names = train['CountryCode'].unique()
print('All country codes to normalize data in each group: \n', group_names)
for cl in cont_columns:
    test = test_scaler(test, train, group_names, 'CountryCode', cl)
    train.loc[:, cl] = train[['CountryCode', cl]].groupby('CountryCode').apply(
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
########################################
# Mapping categorical features to integer for Country, O365SmbChannelType, O365SmbBillingCycleName column!
for cl in oh_columns:
    levels_cat = sorted(list(set(train[cl].values)))
    train, mapdict, mapdict_rev = map_catfeat_to_integer(train, cl, levels_cat, return_map=True)
    levels_int = sorted(list(set(train[cl].values)))
    train = addOHremOrig(train, cl, levels_int, levels_cat)

    test = map_catfeat_to_integer(test, cl, levels_cat, return_map=False)
    test = addOHremOrig(test, cl, levels_int, levels_cat)
    mapdict = pd.DataFrame(mapdict, index=[1]).T
    mapdict = pd.DataFrame({cl: list(mapdict.index), 'code': mapdict.iloc[:, 0]})
    mapdict.to_csv(data_dir + cl+'_code_dict.csv', index=False)


# for small business of size 1, total seats is equal to 1 for everybody -> remove it from data
train = train.drop(['TotalO365PaidSeats'], axis=1)
test = test.drop(['TotalO365PaidSeats'], axis=1)
train.to_csv(data_dir + 'train_Processed_1branch_byCountry.csv', index=False)
test.to_csv(data_dir + 'test_Processed_1branch_byCountry.csv', index=False)


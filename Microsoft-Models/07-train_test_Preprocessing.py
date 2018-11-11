import pandas as pd
import numpy as np


def addOHremOrig(data, column, levels, levels_cat):
    ''' Adds one hot variable and removes the original categorical data'''
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
###################
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

#########
# normalizing columns
colnames = list(train.columns.values)
cont_columns = [cl for cl in colnames[1:]
                if len(set(train[cl].values)) > 10 and cl not in
                ['CountryCode', 'Weight']]
print('Continuous columns=', cont_columns, len(cont_columns))

normalizing_dict = {}
for cl in cont_columns:
    t_max, t_min = np.max(train[cl]), np.min(train[cl])
    normalizing_dict[cl] = [t_min, t_max]
    train[cl] = (train[cl].values - t_min) / (t_max - t_min)
    test[cl] = (test[cl].values - t_min) / (t_max - t_min)


normalizing_dict = pd.DataFrame(normalizing_dict, index=['max', 'min'])

normalizing_dict.to_csv(data_dir+'Train_normalizing_param.csv', index=False)

# for small business of size 1, total seats is equal to 1 for everybody -> remove it from data
train = train.drop(['TotalO365PaidSeats'], axis=1)
test = test.drop(['TotalO365PaidSeats'], axis=1)
print('Shape of Final data= ', train.shape)
train.to_csv(data_dir+'train_Processed.csv', index=False)
test.to_csv(data_dir+'test_Processed.csv', index=False)


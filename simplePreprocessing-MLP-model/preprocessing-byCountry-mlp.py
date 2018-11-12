import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
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


def test_scaler(test_data, train_data, group_names, groupColumn, feat):
    '''finds the associated group of data in train and uses the statistics of that to scale test!'''
    for gr in group_names:
        # print('Group= ', gr, 'feature=', feat)
        tr1 = train_data[train_data[groupColumn] == gr][feat].values
        fmin, fmax = np.amin(tr1), np.amax(tr1)
        vals = (test_data[test_data[groupColumn] == gr][feat].values - fmin) / (fmax - fmin)
        test_data.loc[test_data[groupColumn] == gr, feat] = vals
    return test_data


data_dir = 'C:/behrouz/PythonCodes/ML/preprocessing/'
train = pd.read_csv(data_dir+'train.csv', sep=',')
test = pd.read_csv(data_dir+'test.csv', sep=',')
column_names = train.columns.values
oh_columns = ['O365SmbChannelType']
print(train.shape)
print('going through the columns to find out if they have missing value:')
for n in column_names:
    if any(pd.isna(train[n])):
        print(n)
# just wanna have a category for unknown!
train = train.fillna('Unknown')
test = test.fillna('Unknown')
###################
# looking at histograms of train and test:
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
train.hist(ax=ax)
plt.suptitle('Histogram of features')
plt.show()

# looking at TenureFirstPaid variable by values of O365SmbChnnelType
train.pivot_table('TenureFirstPaid', index=['O365SmbChannelType'], columns=['outHasActiveO365'], aggfunc=np.mean).plot(kind='bar')
plt.ylabel('Mean Number of days till paid')
plt.xticks(rotation=45)
plt.subplots_adjust(top=0.92, bottom=0.2, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()

# Mapping categorical features to integer for O365SmbChannelType column!
oh_col_levels = sorted(train[oh_columns[0]].unique())
print(oh_col_levels)
oh_col_map = dict([(l, i) for i, l in enumerate(oh_col_levels)])
oh_col_map_rev = dict([(i, l) for i, l in enumerate(oh_col_levels)])
print(oh_col_map)
train[oh_columns[0]+str('to_int')] = train[oh_columns[0]].map(oh_col_map)
test[oh_columns[0]+str('to_int')] = test[oh_columns[0]].map(oh_col_map)

# converting categorica; variables into one hot:
for cl in oh_columns:
    oh = pd.get_dummies(train[cl], drop_first=False)
    print(oh.shape, oh.head())
    oh = oh.rename(columns=oh_col_map_rev)
    train = pd.merge(train, oh, left_index=True, right_index=True, how='inner')

    oh = pd.get_dummies(test[cl], drop_first=False)
    print(oh.shape, oh.head())
    oh = oh.rename(columns=oh_col_map_rev)
    test = pd.merge(test, oh, left_index=True, right_index=True, how='inner')

print(train.shape)

# min- max scaling the data:
# first column is index, second is the label, will min max scale the rest!!
colnames = list(train.columns.values)
print(colnames)
cont_columns = [cl for cl in colnames[2:]
                if len(set(train[cl].values)) > 10]
print('Continuous columns=', cont_columns, len(cont_columns))


group_column = 'O365SmbChannelType'
print(train.columns.values)
print(train[[group_column]].head())
group_names = oh_col_levels
for cl in cont_columns:
    test = test_scaler(test, train, group_names, group_column, cl)
    train.loc[:, cl] = train[[group_column, cl]].groupby(group_column).apply(
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

train = train.drop([group_column], axis=1)
test = test.drop([group_column], axis=1)
train.to_csv(data_dir+'train_Processed.csv', index=False)
test.to_csv(data_dir+'test_Processed.csv', index=False)
print(train.shape, test.shape)

# simple MLP on this data:


def MLP(x, weights, biases, learning_rate, activation = tf.nn.relu,
        l2Reg = 0.01, curr_optimizer = tf.train.AdamOptimizer):
    '''
    :param x: placeholder tensor for input
    :param weights: dictionary of all the weight tensors in the model
    :param biases: dictionary of all the bias tensors in the model
    :return: returns the output of the model (just the logits before the softmax (linear output)!
    '''
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = activation(layer_2)
    logits = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    # l2 regularization loss:
    l2Loss = 0
    for key in weights.keys():
        l2Loss += tf.nn.l2_loss(weights[key])
    for key in biases.keys():
        l2Loss += tf.nn.l2_loss(biases[key])
    l2Loss *= l2Reg
    # calculating accuracy measure:
    # binary true false vector of length of training data
    correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # cost function and optimization
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    cost_plus_l2Loss = cost + l2Loss
    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(cost_plus_l2Loss)
    return logits, l2Loss, optimizer, accuracy, cost, cost_plus_l2Loss


def model(x1, activation, istrain, h1_size, h2_size, dropoutRate):
    with tf.variable_scope('wide_model'):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
        else:
            bias_init = tf.zeros_initializer()
        h1 = tf.layers.dense(x1,
                             h1_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h1 = tf.layers.dropout(h1, rate=dropoutRate, training=istrain)

        h2 = tf.layers.dense(h1,
                             h2_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h2 = tf.layers.dropout(h2, rate=dropoutRate, training=istrain)
        logits = tf.squeeze(tf.layers.dense(h2, 1,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer()))
    return logits


colnames = train.columns.values
trainx = train[colnames[2:]].values
trainy = train[colnames[1]].values
testx = test[colnames[2:]].values
testy = test[colnames[1]].values
print(trainx.shape, testx.shape)
x = tf.placeholder(tf.float32, shape=[None, trainx.shape[1]])
y = tf.placeholder(tf.float32, [None])
istrain = tf.placeholder(tf.bool)

logits = model(x, tf.nn.relu, istrain, 32, 32, 0.5)
out_prob = tf.nn.sigmoid(logits)
out_pred = tf.cast(tf.greater(out_prob, 0.5), tf.float32)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.inverse_time_decay(0.001,
                                            global_step=global_step,
                                            decay_steps=10000,
                                            decay_rate=0.1,
                                            staircase=True)
#######################
# every time minimize is called, global step will increment by 1!
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_pred = tf.equal(y, out_pred)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    inds = np.arange(train.shape[0])
    np.random.shuffle(inds)
    trainx = trainx[inds]
    trainy = trainy[inds]
    k = 0
    for i in range(10000):
        if k * 128 >= trainx.shape[0]:
            k = 0
            np.random.shuffle(inds)
            trainx = trainx[inds]
            trainy = trainy[inds]

        start_ind = k * 128
        end_ind = min((k + 1) * 128, trainx.shape[0])
        k += 1
        xbatch = trainx[start_ind: end_ind]
        ybatch = trainy[start_ind: end_ind]
        _ = sess.run(train_op, feed_dict={x: xbatch, y: ybatch, istrain: True})

        if (i + 1) % 100 == 0:
            train_loss, train_prob, train_accu = sess.run([loss, out_prob, accuracy],
                                                           feed_dict={x: trainx,
                                                                      y: trainy,
                                                                      istrain: False})
            print('Iteration= ', i+1)
            print('train loss= ', train_loss, 'train_accu= ', train_accu)



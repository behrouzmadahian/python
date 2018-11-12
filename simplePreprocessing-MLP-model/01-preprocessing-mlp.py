import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc


def glorot_normal_weight_initializer(shape, name=None):
    ''' Use for tanh activation'''
    # Glorot et al. 2012
    initial = tf.random_normal(shape, stddev=np.sqrt(3. / (shape[0] + shape[1])))
    # initial = tf.truncated_normal(shape, stddev=np.sqrt(3. / (shape[0] + shape[1])))
    if name:
        return tf.Variable(initial, name=name)

    else:
        return tf.Variable(initial)


def xavier_from_tf_initializer(shape, name=None):
    ''' Use for tanh activation'''
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def relu_weight_initializer(shape, name=None):
    # He et al 2015
    # values whose magnitude is more than 2 standard deviations
    # from the mean are dropped and re-picked
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[0]))
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


def bias_initializer(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


def performance_statistics(y_true, y_pred, y_probs):
    pos_recall = recall_score(y_true, y_pred, pos_label=1)
    pos_precision = precision_score(y_true, y_pred, pos_label=1)
    pos_f1 = f1_score(y_true, y_pred, pos_label=1)
    neg_recall = recall_score(y_true, y_pred, pos_label=0)
    neg_precision = precision_score(y_true, y_pred, pos_label=0)
    neg_f1 = f1_score(y_true, y_pred, pos_label=0)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    auc_ = auc(fpr, tpr)
    # auc_ = roc_auc_score(y_true, y_probs, average=None)
    return {'prec_recal_f1_pos': (round(pos_precision, 3), round(pos_recall, 3), round(pos_f1, 3)),
            'prec_recall_f1_neg': (round(neg_precision, 3), round(neg_recall, 3), round(neg_f1, 3)),
            'AUC': round(auc_, 3)}


def addOHremOrig(data, column, levels_cat):
    ''' Adds one hot variable and removes the original categorical data'''
    oh = np.zeros((data.shape[0], len(levels_cat)))
    oh[np.arange(oh.shape[0]), data[column].values] = 1
    ohcolumns = [column + '-' + str(l) for l in levels_cat]
    oh = pd.DataFrame(oh, columns=ohcolumns, index=data.index)
    data = pd.merge(data, oh, how='inner', left_index=True, right_index=True)
    data = data.drop([column], axis=1)
    return data


data_dir = 'C:/behrouz/PythonCodes/ML/simple-preprocesing-simpleMLP/'
train = pd.read_csv(data_dir+'train.csv', sep=',')
test = pd.read_csv(data_dir+'test.csv', sep=',')
# looking at the data:
print('Looking at data types:')
print(train.info(), '\n\n')
print('Looking at simple statistics of the columns:')
print(train.describe(), '\n\n')
column_names = train.columns.values
oh_columns = ['O365SmbChannelType']
# lets see how many levels:
print('Levels of categorical column= ', train[oh_columns[0]].unique())
print(train.shape)
print('going through the columns to find out if they have missing value:')
for n in column_names:
    if any(pd.isna(train[n])):
        print('columns %s has NA values' % n)
# just wanna have a category for unknown!
# train = train.fillna('Unknown')
# test = test.fillna('Unknown')
# dropping NA values if any!
train = train.dropna(axis=0)
test = test.dropna(axis=0)
#######################################################################################################################
# looking at histograms of train and test:
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
train.hist(ax=ax)
plt.suptitle('Histogram of features')
plt.show()

# looking at TenureFirstPaid variable by values of O365SmbChannelType
train.pivot_table('TenureFirstPaid',
                  index=['O365SmbChannelType'],
                  columns=['outHasActiveO365'],
                  aggfunc=np.mean).plot(kind='bar')
plt.ylabel('Mean Number of days till paid')
plt.xticks(rotation=45)
plt.subplots_adjust(top=0.92, bottom=0.2, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()

# pairwise feature correlations and scatter plots:
import seaborn as sns
fig, ax = plt.subplots(figsize=(15, 15))
correlations = train.corr()

sns.heatmap(correlations,
            mask=np.zeros_like(correlations, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True,
            ax=ax,
            annot=True,
            fmt='.1f')
plt.show()

# pairs plot of features
'''
diagonal : {‘hist’, ‘kde’}
pick between ‘kde’ and ‘hist’ for either Kernel Density Estimation or Histogram plot in the diagonal
'''
axes = pd.plotting.scatter_matrix(train[train.columns.values[:6]],
                                  # color scatter plot points by categories of one column!
                                  c=train['outHasActiveO365'],
                                  figsize=(15, 15),
                                  marker='o',
                                  diagonal='hist',
                                  hist_kwds={'bins': 20},
                                  s=60,
                                  alpha=0.8)
# print(plt.np.triu_indices_from(axes, k=1))
correlations = correlations.as_matrix()
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("correlation= %.2f" % correlations[i, j],
                        (0.3, 0.8),
                        xycoords='axes fraction',
                        ha='center',
                        va='center')
plt.show()

##########################################################################################################
# Mapping categorical features to integer for O365SmbChannelType column!
oh_col_levels = sorted(train[oh_columns[0]].unique())
print(oh_col_levels)
oh_col_map = dict([(l, i) for i, l in enumerate(oh_col_levels)])
oh_col_map_rev = dict([(i, l) for i, l in enumerate(oh_col_levels)])
print(oh_col_map)
train[oh_columns[0]] = train[oh_columns[0]].map(oh_col_map)
test[oh_columns[0]] = test[oh_columns[0]].map(oh_col_map)

# converting categorical; variables into one hot:
for cl in oh_columns:
    oh = pd.get_dummies(train[cl], drop_first=False)
    print(oh.head())
    oh = oh.rename(columns=oh_col_map_rev)
    train = pd.merge(train, oh, left_index=True, right_index=True, how='inner')
    train = train.drop([cl], axis=1)
    # the same for test data:
    oh = pd.get_dummies(test[cl], drop_first=False)
    oh = oh.rename(columns=oh_col_map_rev)
    test = pd.merge(test, oh, left_index=True, right_index=True, how='inner')
    test = test.drop([cl], axis=1)

print(train.shape, test.shape)

# min- max scaling the data:
# first column is index, second is the label, will min max scale the rest!!
colnames = list(train.columns.values)
print(colnames)

cont_columns = [cl for cl in colnames[2:]
                if len(set(train[cl].values)) > 10]
print('Continuous columns=', cont_columns, len(cont_columns))

normalizing_dict = {}
for cl in cont_columns:
    t_max, t_min = np.max(train[cl]), np.min(train[cl])
    normalizing_dict[cl] = [t_min, t_max]
    train[cl] = (train[cl].values - t_min) / (t_max - t_min)
    test[cl] = (test[cl].values - t_min) / (t_max - t_min)

normalizing_dict = pd.DataFrame(normalizing_dict, index=['max', 'min'])
normalizing_dict.to_csv(data_dir+'Train_normalizing_param.csv', index=False)
train.to_csv(data_dir+'train_Processed.csv', index=False)
test.to_csv(data_dir+'test_Processed.csv', index=False)


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
'''
Weight and bias dicts for above model

weights = {'h1':   xavier_from_tf_initializer([trainx.shape[1], 32], name=None),
           'h2':   xavier_from_tf_initializer([32, 32], name=None)
           'out' : xavier_from_tf_initializer([32, 1], name=None)
           }
biases = {'b1' :  bias_initializer([32]),
          'b2' : bias_initializer([32]),
          'out': bias_initializer([1])
           }
'''


def model(x1, activation, istrain, h1_size, h2_size, dropoutRate):
    with tf.variable_scope('model'):
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
correct_pred = tf.equal(y, out_pred)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
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





import numpy as np
import random
import tensorflow as tf
import os
from gridworld_7 import gameEnv
from matplotlib import  pyplot as plt
'''
Problem:
grid world. move blue block to green (reward 1) while avoiding red block (reward -1).
POMDP: Agent can only see a single block  around it at any given direction. 
environment is  9*9 block.
each episode is fixed at 50 steps. there are 4 greens and 2 red squares.
when the agent moves to a green or red square, a new one is randomly placed in the environment
to replace it.
'''

env = gameEnv(partial =  True, size = 9)

# defining conv layer
def conv2d(x, W, b, strides = 1, padding = 'SAME', activation = tf.nn.relu):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = padding )
    x = tf.nn.bias_add(x, b)
    return activation(x)

# #of final conv layer filters before splitting into Advantage and Value streams
h_size = 512
# we want 4 conv layers in Q network and target network

# RQ network
class RQnetwork(object):

    def __init__(self, weights, biases, wNames, bNames, myScope):

        # each frame is originally 84 *84 *3

        self.scalarInput = tf.placeholder(tf.float32, shape = [None, 21168])
        self.imageIn = tf.reshape(self.scalarInput, shape = [-1, 84, 84, 3])
        self.weights = weights
        self.biases = biases

        self.conv1 = conv2d(self.imageIn, self.weights[wNames[0]], self.biases[bNames[0]] ,
                            strides = 4, padding = 'VALID', activation = tf.nn.relu)

        self.conv2 = conv2d(self.conv1, self.weights[wNames[1]], self.biases[bNames[1]],
                            strides = 2, padding ='VALID', activation = tf.nn.relu)

        self.conv3 = conv2d(self.conv2, self.weights[wNames[2]], self.biases[bNames[2]],
                            strides = 1, padding ='VALID', activation = tf.nn.relu)

        self.conv4 = conv2d(self.conv3, self.weights[wNames[3]], self.biases[bNames[3]],
                            strides = 1, padding ='VALID', activation = tf.nn.relu)
        print('Output of last Conv Layer=', self.conv4.get_shape)

        # output of conv4 is (?, 1, 1, 512) need to reshape to (?, timesteps, 512)

        self.trainLength = tf.placeholder(dtype = tf.int32)
        self.batch_size = tf.placeholder( dtype = tf.int32, shape = [])

        # flatten and maintain batch size
        self.convFlat = tf.contrib.layers.flatten(self.conv4)
        self.convFlat = tf.reshape(self.convFlat, shape = [self.batch_size, self.trainLength, h_size])

        cell = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, forget_bias = 1.0,
                                            activation = tf.nn.tanh, state_is_tuple = True)

        self.state_in = cell.zero_state(self.batch_size, tf.float32)

        # dynamic_rnn: 'rnn_h' is a tensor of shape [batch_size, timestep, cell_state_size]
        # 'rnn_c' is a tensor of shape [batch_size, cell_state_size]

        self.rnn_h, self.rnn_c = tf.nn.dynamic_rnn(inputs = self.convFlat, cell = cell,  dtype = tf.float32,
                                                   initial_state = self.state_in, scope = myScope +'_rnn' )

        print('Shape of hidden state output of rnn block=', self.rnn_h.get_shape())

        # uses all the previous H not only the last one!
        # reason: we need advantage and values for each of the frames in all traces.
        self.rnn = tf.reshape(self.rnn_h, shape = [-1, h_size])

        # split the output into separate advantage and value streams
        self.streamA, self.streamV = tf.split(self.rnn, num_or_size_splits = 2, axis = 1)

        self.AW = tf.get_variable(wNames[0][0] + 'AW', shape = [self.streamA.get_shape()[1], env.actions],
                                  initializer = tf.contrib.layers.xavier_initializer())

        self.VW = tf.get_variable(wNames[0][0] + 'VW', shape = [self.streamV.get_shape()[1], 1],
                                  initializer = tf.contrib.layers.xavier_initializer())

        self.Advantage = tf.matmul(self.streamA, self.AW)
        print('Shape of advantage tensor =', self.Advantage.get_shape())

        self.Value = tf.matmul(self.streamV, self.VW)
        print('Shape of value tensor=', self.Value.shape)

        # combine value and advantage to get final Q value

        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis = 1, keep_dims = True))
        print('Shape of output Q tensor=',self.Qout.get_shape())

        self.predict = tf.argmax(self.Qout, axis = 1)
        print('Shape of predicted action tensor=', self.predict.get_shape())

        # Obtain the loss by taking the sum of squares difference between the target and prediction Q values

        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, depth = 4,  dtype = tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis = 1)
        print('Shape of Q(s,a)  tensor=',self.Q.get_shape())

        self.td_error = tf.square(self.targetQ - self.Q)

        # in order to only propagate accurate gradients, we mask the first half of losses for each trace
        # Lample & Chatlot 2016
        # here each trace is 8 consecutive frames.
        # Make sure you understand how batches are fed into the model so that
        # masking makes sense

        self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])
        self.maskB = tf.ones([self.batch_size, self.trainLength // 2])
        self.mask = tf.concat([self.maskA, self.maskB], axis = 1)
        self.mask = tf.reshape(self.mask, [-1])

        self.loss = tf.reduce_mean(self.td_error * self.mask)
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.update_model = self.trainer.minimize(self.loss)


class Experience_buffer(object):
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):

        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0: (1 + len(self.buffer)) - self.buffer_size] = []

        self.buffer.append(experience)

    def sample (self, batch_size, trace_length):
        # need to make sure experience are stored correctly sequentially for each trace in training part!
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []

        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point: point + trace_length])

        sampledTraces = np.array(sampledTraces)
        print('Shape of sampled Traces:', sampledTraces.shape)
        return np.reshape (sampledTraces, [batch_size * trace_length, 5]) # 256 * 5

def processState (states):
    return np.reshape(states, [21168])

def updateTargetGraph(tfvars, tau):
    ''' Target network weights are weighted average of Q network weights and target network weights.
        note: in this implementation, the first half of parameters are those of primary Q network.
        rau ~0.001
    '''
    mid_point = len(tfvars) // 2
    op_holder = []
    for ix, var in enumerate(tfvars[0: mid_point]):

        update = var * tau + (1 - tau) * tfvars[ix  + mid_point]
        op_holder.append(tfvars[ix + mid_point].assign(update))

    return op_holder

def updateTarget(op_holder, sess):

    for op in op_holder:
        sess.run(op)

batch_size = 32  # How many experiences to use for each training step.
batch_size_preTrain = 1
update_freq = 4  # How often to perform a training step.
discount = 0.99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.0  # Final chance of random action
annealing_steps = 6000  # How many steps of training to reduce startE to endE.
num_episodes = 1000  # How many episodes of game environment to train network with.
pre_train_steps = 2000  # How many steps of random actions before training begins.
max_epLength = 50  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path = "./8-rdqn"  # The path to save our model to.
tau = 0.001  # Rate to update target network toward primary network
time_per_step = 1 # length of each step used in gif creation
summary_length = 100 # Number of episodes to periodically save for analysis
trace_length = 8 # How long each experience trace will be when training

q_wnames = ['W1', 'W2', 'W3', 'W4']
q_bnames = ['b1', 'b2', 'b3', 'b4']

t_wnames = ['tW1', 'tW2', 'tW3', 'tW4']
t_bnames = ['tb1', 'tb2', 'tb3', 'tb4']

Qn_weights = {
    # 8 * 8 convolution 3 channel input, 32 output
    'W1': tf.get_variable( 'W1', shape =[8, 8, 3, 32], initializer = tf.contrib.layers.xavier_initializer()),
    'W2': tf.get_variable( 'W2', shape =[4, 4, 32, 64], initializer = tf.contrib.layers.xavier_initializer()),
    'W3': tf.get_variable( 'W3', shape =[3, 3, 64, 64], initializer = tf.contrib.layers.xavier_initializer()),
    'W4': tf.get_variable( 'W4', shape =[7, 7, 64, h_size], initializer = tf.contrib.layers.xavier_initializer())
}
Qn_biases = {
    'b1' : tf.Variable(tf.constant(0.1, shape = [32]), dtype= tf.float32),
    'b2' : tf.Variable(tf.constant(0.1, shape = [64]), dtype= tf.float32),
    'b3' : tf.Variable(tf.constant(0.1, shape = [64]), dtype= tf.float32),
    'b4' : tf.Variable(tf.constant(0.1, shape = [h_size]), dtype= tf.float32)
}

mainQN = RQnetwork(Qn_weights, Qn_biases, q_wnames, q_bnames, 'main')


targetN_weights = {
    'tW1': tf.get_variable( 'tW1', shape =[8, 8, 3, 32], initializer= tf.contrib.layers.xavier_initializer()),
    'tW2': tf.get_variable( 'tW2', shape =[4, 4, 32, 64], initializer= tf.contrib.layers.xavier_initializer()),
    'tW3': tf.get_variable( 'tW3', shape =[3, 3, 64, 64], initializer= tf.contrib.layers.xavier_initializer()),
    'tW4': tf.get_variable( 'tW4', shape =[7, 7, 64, h_size], initializer= tf.contrib.layers.xavier_initializer())
}
targetN_biases = {
    'tb1' : tf.Variable(tf.constant(0.1, shape = [32]), dtype= tf.float32),
    'tb2' : tf.Variable(tf.constant(0.1, shape = [64]), dtype= tf.float32),
    'tb3' : tf.Variable(tf.constant(0.1, shape = [64]), dtype= tf.float32),
    'tb4' : tf.Variable(tf.constant(0.1, shape = [h_size]), dtype= tf.float32)
}

targetQN = RQnetwork(targetN_weights, targetN_biases, t_wnames, t_bnames, 'target')

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = 5)
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

myBuffer = Experience_buffer()

# set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / annealing_steps

# lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:

    if load_model == True:
        print('Restoring model from file..')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    else:
        sess.run(init)

    for i in range(num_episodes):
        episode_buffer = []

        # reset environment and get first new observation

        s = processState(env.reset())
        done = False
        rAll = 0
        j = 0

        # state used in pre training and gathering new experiences..

        lstm_state = (np.zeros([batch_size_preTrain, h_size]), np.zeros([batch_size_preTrain, h_size])) # reset RNN state

        while j < max_epLength:
            j += 1

            # choose action greedily (with chance of random action from Q network)
            # These will be accumulated in the buffer to be randomly sampled in training

            if np.random.rand(1) < e or total_steps < pre_train_steps:

                lstm_state1 = sess.run(mainQN.rnn_c, feed_dict = {mainQN.scalarInput : [s / 255.],
                                                             mainQN.trainLength : 1, mainQN.state_in : lstm_state,
                                                             mainQN.batch_size : batch_size_preTrain})
                a = np.random.randint(0, 4)

            else:
                a, lstm_state1 = sess.run([mainQN.predict, mainQN.rnn_c], feed_dict = {mainQN.scalarInput:[s / 255.],
                                                                                 mainQN.trainLength :1,
                                                                                 mainQN.state_in : lstm_state,
                                                                                 mainQN.batch_size :batch_size_preTrain})
                a = a[0]

            s1, r, done = env.step(a)
            s1 = processState(s1)

            total_steps += 1
            episode_buffer.append(np.reshape([s, a, r, s1, done], [1, 5]))

            if total_steps > pre_train_steps:
                if e >  endE:
                    e -= stepDrop

                if total_steps % update_freq == 0:

                    updateTarget(targetOps, sess)

                    # Reset the RNN state
                    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))

                    trainBatch = myBuffer.sample(batch_size, trace_length)
                    print('Shape of train Batch=', trainBatch.shape)

                    print('Shape of states in the batch=', np.vstack(trainBatch[:, 3]).shape)

                    # perform Double-DQN update to target Q values
                    Q1_action = sess.run([mainQN.predict], feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3]/ 255.0),
                                                               mainQN.trainLength : trace_length,
                                                               mainQN.state_in : state_train,
                                                               mainQN.batch_size: batch_size})

                    Q2 = sess.run(targetQN.Qout, feed_dict= {targetQN.scalarInput: np.vstack(trainBatch[:, 3] / 255.0),
                                                             targetQN.trainLength: trace_length,
                                                             targetQN.state_in: state_train,
                                                             targetQN.batch_size: batch_size})

                    #print('Length of generated index for Max Q from mainQN=', len(Q1))
                    print('TargetQ matrix shape=', Q2.shape)

                    end_multiplier = -(trainBatch[:, 4] - 1) # 1 if not done else 0

                    # hack to get different columns for differnent rows!
                    doubleQ = Q2 [range(batch_size * trace_length), Q1_action]

                    # if end is reached, there's no more reward!
                    targetQ = trainBatch[:, 2] + (discount * doubleQ * end_multiplier)
                    print(targetQ.shape, '====')
                    targetQ = targetQ[0, :]
                    print(targetQ.shape, '====')

                    # Update the network with our target values.
                    sess.run(mainQN.update_model, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0] / 255.0),
                                        mainQN.targetQ: targetQ,
                                        mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length,
                                        mainQN.state_in: state_train, mainQN.batch_size: batch_size})

            rAll += r
            s = s1

            lstm_state = lstm_state1 # this state is used as part of pretraining...

            if done == True:
                break

        # add the episode to the experience buffer
        episode_buffer = np.array(episode_buffer)
        print('Shape of experience for current episode=', episode_buffer.shape)

        myBuffer.add(episode_buffer)
        jList.append(j)
        rList.append(rAll)

        if i % 10 == 0:
            print('Total Steps = ', total_steps, "Episode=", i,' Mean Reward=', np.mean(rList[10:]),'Random action chance=', e)

        # Periodically save the model.
        if i % 1000 == 0 and i != 0:
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print("Saved Model")

    saver.save(sess, path + '/model-' + str(i) + '.cptk')


print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")


# Mean reward over time:
rMat = np.resize(np.array(rList),[len(rList)//100, 100])
rMean = np.average(rMat, axis=1)
plt.plot(rMean)
plt.show()


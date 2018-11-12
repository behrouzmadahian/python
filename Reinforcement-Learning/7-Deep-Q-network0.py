import numpy as np
import random
import tensorflow as tf
from matplotlib import  pyplot as plt
import os

'''
What we do here:
1. a Q network that works on screenshot of the game. 

2. impelment experience replay, that stores the episodes and randomly uses them to train Q network.
we can prevent the neural network from learning only about immediate behavior in the environment,
 and to learn from a more diverse array of past experiences. 

3. a target network to calculate target Q values.
This second neural network is used to produce a target-Q value that can be used to
calculate the loss of behavior during learning. 
we separate out, the network that produces the target Q values, to stablizie the learning process.

Instead of periodically updating everything about the target neural network at  once,
we will update frequently, but slowly. 

Double DQN:
the networsk, over estimate the Q values. to remedy this, we divide the last layer out put of the network
into two piece, one estimates the value and one the Advantage.
V(s): How good a given state is.
A(a): How much better a particular action at state s is.
Q(s,a) = V(s) + A(a)
 
Double DQN have resulted in improved performance, stability and faster learning time.


Deep Q-Network using both Double DQN and Dueling DQN. 
The agent learn to solve a navigation task in a basic grid world.
Goal:
Move the blue block to green while avoiding red. The agent controls the blue square
and can move up, down, left, and right.
move to gree (reward =1), hitting red square (reqard =-1).
The position 3 blocks are randomized every episode.

 
'''
# load environment
from gridworld_7 import  gameEnv
env = gameEnv( partial = False, size = 5) # 5*5 grid

# defining conv layer
def conv2d( x, W, b, strides = 1, padding='SAME', activation =tf.nn.relu):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = padding )
    x = tf.nn.bias_add(x, b)
    return activation(x)

# size of final conv layer before  splitting into Advantage and Value streams
h_size = 512
xavier_init = tf.contrib.layers.xavier_initializer()
# we want 4 conv layers in Q network and target network

# Q network
class Qnetwork(object):

    def __init__(self, weights, biases, wNames, bNames):
        # The network receives a fame from the game, flattened into an array.
        # the network, reshapes the image and pass it through CNNs.
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
                            strides = 1, padding ='VALID', activation =tf.nn.relu)


        # split the output of conv4  into separate advantage and value streams
        self.streamAC, self.streamVC = tf.split(self.conv4, num_or_size_splits = 2, axis = 3)

        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)

        self.AW = tf.get_variable (wNames[0][0] + 'AW', shape = [self.streamA.get_shape()[1], env.actions],
                                   initializer=   tf.contrib.layers.xavier_initializer())

        self.VW = tf.get_variable(wNames[0][0] + 'VW', shape=[self.streamV.get_shape()[1], 1],
                                  initializer=tf.contrib.layers.xavier_initializer())


        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # combine value and advantage to get final Q value

        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis = 1, keep_dims = True))

        self.predict = tf.argmax(self.Qout, 1)

        # loss and train step
        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)

        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype = tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot) , axis = 1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)

        self.updateModel =self.trainer.minimize(self.loss)


# Experience Replay Class:
# allows us to store experiences and sample randomly to train the network
class Experience_buffer(object):
    def __init__(self, buffer_size = 50000):
        self.buffer =[]
        self.buffer_size = buffer_size

    def add (self, experience):
        # if buffer is full, remove some of the old experiences
        if len(self.buffer) + len(experience) >= self.buffer_size:

            self.buffer [0: len(experience) + len(self.buffer) - self.buffer_size] = []

        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape (np.array(random.sample(self.buffer, size)), [size, 5])

# function to flatten the game frames:
def processState (states):
    return np.reshape(states, [21168])

# Function to update the parameters of target network with those of primary Q network.
# this helps to stablizie Q value learning.

def updateTargetGraph(tfvars, tau):
    ''' Target network weights are weighted average of Q network weights and target network weights.
        note: in this implementation, the first half of parameters are those of primary Q network.
    '''
    mid_point = len(tfvars) //2
    op_holder = []
    for ix, var in enumerate(tfvars[0: mid_point]):

        update = var * tau + ( 1 - tau) * tfvars[ix  + mid_point]
        op_holder.append(tfvars[ ix + mid_point].assign(update))

    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


batch_size = 32 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
discount = 0.99 # Discount factor on the target Q-values
startE = 1 # Starting chance of random action
endE = 0.1 # Final chance of random action
annealing_steps = 1000 # How many steps of training to reduce startE to endE.
num_episodes = 1000 # How many episodes of game environment to train network with.
pre_train_steps = 1000 # How many steps of random actions before training begins.
max_epLength = 50 # The max allowed length of our episode.
load_model = False # Whether to load a saved model.
path = "./7-dqn" # The path to save our model to.
tau = 0.001 # Rate to update target network toward primary network

q_wnames = ['W1', 'W2', 'W3', 'W4']
q_bnames = ['b1', 'b2', 'b3', 'b4']

t_wnames = ['tW1', 'tW2', 'tW3', 'tW4']
t_bnames = ['tb1', 'tb2', 'tb3', 'tb4']

Qn_weights = {
    # 8 * 8 convolution 3 channel input, 32 output
    'W1': tf.get_variable( 'W1', shape =[8, 8, 3, 32]  ,initializer= tf.contrib.layers.xavier_initializer()),
    'W2': tf.get_variable( 'W2', shape =[4, 4, 32, 64]  ,initializer= tf.contrib.layers.xavier_initializer()),
    'W3': tf.get_variable( 'W3', shape =[3, 3, 64, 64]  ,initializer= tf.contrib.layers.xavier_initializer()),
    'W4': tf.get_variable( 'W4', shape =[7, 7, 64, h_size]  ,initializer= tf.contrib.layers.xavier_initializer())
}
Qn_biases = {
    'b1' : tf.Variable(tf.constant(0.1, shape = [32]), dtype= tf.float32),
    'b2' : tf.Variable(tf.constant(0.1, shape = [64]), dtype= tf.float32),
    'b3' : tf.Variable(tf.constant(0.1, shape = [64]), dtype= tf.float32),
    'b4' : tf.Variable(tf.constant(0.1, shape = [h_size]), dtype= tf.float32)
}

mainQN = Qnetwork(Qn_weights, Qn_biases, q_wnames, q_bnames)

targetN_weights = {
    'tW1': tf.get_variable( 'tW1', shape =[8, 8, 3, 32]  ,initializer= tf.contrib.layers.xavier_initializer()),
    'tW2': tf.get_variable( 'tW2', shape =[4, 4, 32, 64]  ,initializer= tf.contrib.layers.xavier_initializer()),
    'tW3': tf.get_variable( 'tW3', shape =[3, 3, 64, 64]  ,initializer= tf.contrib.layers.xavier_initializer()),
    'tW4': tf.get_variable( 'tW4', shape =[7, 7, 64, h_size]  ,initializer= tf.contrib.layers.xavier_initializer())
}
targetN_biases = {
    'tb1' : tf.Variable(tf.constant(0.1, shape = [32]), dtype= tf.float32),
    'tb2' : tf.Variable(tf.constant(0.1, shape = [64]), dtype= tf.float32),
    'tb3' : tf.Variable(tf.constant(0.1, shape = [64]), dtype= tf.float32),
    'tb4' : tf.Variable(tf.constant(0.1, shape = [h_size]), dtype= tf.float32)
}

targetQN = Qnetwork(targetN_weights, targetN_biases, t_wnames, t_bnames)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

myBuffer = Experience_buffer()

# Set the rate of decrease in random action
e = startE
stepDrop = (startE - endE) /annealing_steps

# lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

if not os.path.exists(path):
    os.makedirs(path)

# Training the Network

with tf.Session() as sess:
    sess.run(init)

    if load_model == True:
        print('restoring Model from file ..')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(num_episodes):
        print('Episode=', i + 1)
        episodeBuffer = Experience_buffer()

        # reset environment and get new initial observation
        s = env.reset()
        s = processState(s)
        done = False
        rAll = 0
        j = 0

        # the Q network
        # If the agent takes longer than 50 moves to reach either of the blocks, end the trial.
        while j < max_epLength :
            j += 1

            # Choose an action by greedily ( with a chance of random action) from the Q net
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)

            else:
                a = sess.run(mainQN.predict, feed_dict = {mainQN.scalarInput : [s]})

            s1, r, done = env.step(a)

            s1 = processState(s1)
            total_steps += 1

            # save experience to episode buffer

            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, done]), [1, 5]))

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % update_freq == 0:
                    # get a random batch of experiences
                    train_batch = myBuffer.sample(batch_size)

                    # perform double-DQN update to target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict = { mainQN.scalarInput : np.vstack(train_batch[:, 3])})

                    Q2 = sess.run(targetQN.Qout, feed_dict = {targetQN.scalarInput:np.vstack(train_batch[:,3])})

                    end_multiplier = -( train_batch[:, 4] - 1) # if done zero else 1.

                    doubleQ = Q2 [range(batch_size), Q1]

                    targetQ = train_batch[:, 2] + (discount * doubleQ * end_multiplier)

                    # update the network with our target values

                    _ = sess.run(mainQN.updateModel, feed_dict = {
                                        mainQN.scalarInput : np.vstack(train_batch[:, 0]),
                                        mainQN.targetQ : targetQ ,mainQN.actions : train_batch[:, 1]
                                })

                    # update target network toward primary network
                    updateTarget(targetOps, sess)

                rAll += r

                s = s1
                if done == True:
                    break

        myBuffer.add(episodeBuffer.buffer)

        jList.append(j)
        rList.append(rAll)

        # periodically save the model
        if i % 1000 ==0:
            saver.save(sess, path+'/model-' + str(i)+ '.ckpt')
            print('Saved Model')

        if len(rList) % 10 == 0:
            print('Total Steps = ',total_steps,' Mean Reward=' ,np.mean(rList[10:]),'Random action chance=' ,e)

    saver.save(sess, path +'/model'+ str(i) + 'ckpt')

print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")


# Mean reward over time:
rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
plt.show()







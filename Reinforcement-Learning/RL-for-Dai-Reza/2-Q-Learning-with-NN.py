import gym
import numpy as np
import tensorflow as tf
from matplotlib import  pyplot as plt

'''
Tables do not scale well! We can use NNs as function approximators. We learn to map
states to Q-values!

Here we use a 1-layer NN and that takes a one hot vector representing a state, and 
produces a vector of size 4 Q values, one for each action!
loss function: sum squared error loss between current predicted Q value and the target value. 

In this case, our Q-target for the chosen action is the equivalent to the Q-value computed as:
Q(s, a) =  r + decay * max( Q(s', a') )

'''

env = gym.make('FrozenLake-v0')
tf.reset_default_graph()

# building the network

state_in = tf.placeholder(shape = [1, 16], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))

Q_sa_prediction = tf.matmul(state_in, W )
max_Q_sa_prediction = tf.argmax(Q_sa_prediction, axis = 1)

# loss and optimizer

# we need to feed this.
nextQ = tf.placeholder(shape = [1, 4], dtype = tf.float32)

loss = tf.reduce_sum(tf.square(nextQ - Q_sa_prediction ))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1)
train_step = optimizer.minimize(loss)

init  = tf.global_variables_initializer()

# set learning parameters
decay = 0.99 ; e = 0.1 ; num_episodes = 2000

# create lists to contain total rewards and steps per episode

jList = []
rList = []
one_hot_states = np.identity(16)
with tf.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):

        # reset env and get a new state

        s = env.reset()
        rAll = 0
        done = False
        j = 0

        #  the Q network

        while j < 99:
            j += 1

            # choosing an action greedily with noise:
            # Q value of all actions from state s
            # a is vector of one value!

            a, Qsa_all = sess.run([max_Q_sa_prediction, Q_sa_prediction],
                                  feed_dict = {state_in: one_hot_states[s : s + 1]})

            # explore or exploit
            if np.random.rand(1) < e:

                a[0] = env.action_space.sample()

            # get new state and reward from environment

            s1, r, done, _ = env.step(a[0])

            # Obtain the Q values  of the new state s1 by feeding the new state through our network

            Qsa1 = sess.run(Q_sa_prediction, feed_dict= {state_in: one_hot_states[s1 : s1 + 1]})

            # obtain maxQ1 and set our target value for chosen action
            maxQ1 = np.max(Qsa1)
            targetQ = Qsa_all

            # Q(s,a) = r + decay * max( Q(s', a'))

            targetQ[0, a[0]] = r + decay * maxQ1

            # Train our network using target and predicted Q values

            sess.run([train_step], feed_dict = {state_in: one_hot_states[s : s + 1], nextQ : targetQ})

            rAll += r
            s = s1

            if done == True:
                # reduce the chance of random action as we train the model
                e = 1./ (i/50 + 10)
        jList.append(j)
        rList.append(rAll)

print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

plt.plot(jList)
print(jList)
plt.show()
plt.plot(rList)
plt.show()

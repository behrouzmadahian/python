import tensorflow as tf
import numpy as np
'''
N-armed bandit Problem definition: 
The are n slot machines, each with different fixed payout probability.
Goal: discover the machine with the best payout, and maximize 
the returned reward by always choosing it.
Here we explore the 4-armed bandit.
Assume, we know the payout of each machine, and the agent needs to discover it.

Goal: obtain the optimal policy. Here, we use a method called policy gradients where NN learns a policy for 
picking actions by adjusting its weights through gradient descent using feedback from the environment.

Other approach would be Q-value learning. In this approach, instead of learning the optimal action in 
a given state, the agent learns to predict how good a given state or action will be
for the agent to be in.
'''
bandits = [-0.2, 0, 0.2, 1]
num_bandits = len(bandits)
total_episodes = 10000

# holds the total reward of choosing each slot machine
total_reward = np.zeros(num_bandits)

e = 0.1 # chance of taking a random action

# pullBandit: generates a random number from a normal dist with mean zero,
# the higher the bandit number, the more likely a positive reward will be returned. We want our
# agent to learn to always choose the bandit that will give that positive reward.
# here, bandit 4, is set to most often positive  reward

def pullBandit(bandit):
    ''' given the bandit, returns the reward '''

    # get a random number
    result = np.random.rand(1)

    if result < bandit:
        # return a positive reward
        return 1
    else:
        return -1

#  simple Neural agent

# Consists of set of values for each of the bandits, each value is an estimate
# of the value of the return from choosing the bandit.
# we use a policy gradient method to update the agent by moving the value for the
# selected action toward the received reward.

weights = tf.Variable(tf.ones([num_bandits])) # assume all bandits are equal initially
chosen_action = tf.argmax(weights, 0) # axis = 0

# we feed the reward and chosen action to the network
reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)

# we only need this if we want to be able to take random action (Exploration) time to time

action_holder = tf.placeholder(shape = [1], dtype = tf.int32)

responsible_weight = tf.slice(weights, action_holder, [1])

loss = - ( tf.log(responsible_weight) * reward_holder )
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

train_step = optimizer.minimize(loss)

# Training the simple Neural Agent

init = tf.global_variables_initializer()

# launch the Tensorflow graph
with tf.Session() as sess:
    sess.run(init)

    i = 0
    while i < total_episodes:

        # choose either a random action or one from the network
        if np.random.rand(1) < e:

            action = np.random.randint(num_bandits)

        else:

            action = sess.run(chosen_action)

        reward = pullBandit(bandits[action])

        # Update the network
        _, resp_W, W = sess.run([train_step, responsible_weight, weights],
                              feed_dict = {reward_holder: [reward], action_holder:[action]})

        # update the running tally of scores
        total_reward[action] += reward

        if i % 50 == 0:
            print( 'Running reward for the ' + str(num_bandits) + ' bandits: ', str(total_reward))

        i += 1

print('The agent thinks bandit '+ str (np.argmax(W) + 1) + ' is the most promising  ...')

if np.argmax(W) == np.argmax(np.array(bandits)):
    print(' The agent was correct..')
else:
    print('The agent was wrong..')

print('Final weights of each bandit')
print(W)





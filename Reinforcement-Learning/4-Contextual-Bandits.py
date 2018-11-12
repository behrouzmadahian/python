import  numpy as np
import tensorflow as tf

'''
In n- armed bandit, there were no environmental states, and the agent must simply learn to
choose which action is the best to take.
Without any state, the best action at any moment is also the best action always.

Here: (Still not a ful Reinforcement Learning Formulation!)
1. There are states, but they aren't detemined by previous states or actions.
2. We won't be considering delayed rewards.
i.e: Actions at time t, does not give rise to state at time t + 1
state -> action
action -> reward
state -> reward

Problem definition:
there are 3 bandits, each have 4 arms, each arm results in some reward with some probability.

There can be multiple bandit, and the state of the environment tells us which bandit we are dealing with.

GOAL:
The goal of the agent is to learn the best action not just for a single bandit,
but for any number of them.

The agent needs to learn to condition its actions on the state of the environment. 
unless it does this it won't achieve the maximum reward possible over time.

Neural agent:
single layer NN. Takes an state and produces an action.
Using policy Gradient, the Neural agent learns to take actions that maximize its reward.
'''

class Contextual_bandit(object):

    def __init__(self):
        self.state = 0

        # list of bandits. arms 4, 2, and one are most optimal respectively
        self.bandits = np.array([[-0.2, -0.1, 0, 5], [-0.1, 5, -1, -0.25], [5, -5, -5, -5]])

        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):

        #returns a random state for each episode start!

        self.state = np.random.randint(0, self.num_bandits)

        return self.state

    def pullArm(self, action):
        '''Given an action returns the reward'''

        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)

        if result < bandit:
            return 1 # positive reward
        else:
            return -1

# The policy-based Agent:

class Agent(object):
    def __init__(self, lr, s_size, a_size):

        self.state_in = tf.placeholder(shape = [1], dtype = tf.int32)

        state_in_oh = tf.one_hot(self.state_in, depth = s_size)

        # agent network

        W = tf.Variable(tf.constant(1, shape = [s_size, a_size], dtype= tf.float32))
        b = tf.Variable(tf.constant(0, shape = [a_size], dtype= tf.float32))

        outP =  tf.matmul(state_in_oh, W) + b
        outP = tf.nn.sigmoid(outP)

        self.output = tf.reshape(outP, [-1])

        print('Shape of Flattened output=', self.output.get_shape())

        self.chosen_action = tf.argmax(self.output, 0)

        # Training Procedure

        self.reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)

        # we only need this if we want to be able to take random action (Exploration) time to time
        # we will feed self.action_holder either with self.chosen_action (Exploitation) or random action

        self.action_holder = tf.placeholder(shape=[1], dtype =tf.int32)

        self.handle_to_pull_weight = tf.slice(self.output, self.action_holder, [1])

        self.loss = -(tf.log(self.handle_to_pull_weight) * self.reward_holder)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)

        self.train_step = optimizer.minimize(self.loss)


# Training the Neural Agent

cBandit = Contextual_bandit()
myAgent = Agent(lr = 0.01, s_size = cBandit.num_bandits, a_size = cBandit.num_actions)

weights = tf.trainable_variables()[0] # weights that we will evaluate to look into the network, only W, not b
print(weights.get_shape())

total_episodes = 50000
total_reward  = np.zeros((cBandit.num_bandits, cBandit.num_actions))

# chance for exploration
e = 0.1

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    i = 0
    while i < total_episodes:

        # get a random initial state (bandit)

        s = cBandit.getBandit()

        # choose either a random action or one from network
        if np.random.rand(1) < e:

            action = np.random.randint(cBandit.num_actions)

        else:
            action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in: [s]})

        reward = cBandit.pullArm(action)

        _, trained_weights = sess.run([myAgent.train_step, weights],
                               feed_dict = {myAgent.reward_holder:[reward],
                                            myAgent.action_holder: [action],
                                            myAgent.state_in:[s]} )

        # update running tally of scores
        total_reward[s, action] += reward


        if i % 1000 == 0:
            print('Mean reward for each of the ' + str(cBandit.num_bandits) +
                  ' bandits: ' + str(np.mean(total_reward, axis = 1)))

        i += 1

for a in range(cBandit.num_bandits):
    print('The Agent thinks action '+ str(np.argmax(trained_weights[a,:] + 1)) +
          ' for bandit '+ str(a + 1) + ' is the most promissing')

    if np.argmax(trained_weights[a,:]) == np.argmax(cBandit.bandits[a]):
        print('.. and it was right!')

    else:
        print('.. and it was wrong!')

print(cBandit.bandits)



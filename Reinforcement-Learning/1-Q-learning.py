import gym
import numpy as np

'''
Unlike policy gradient methods, which attempt to learn functions which directly map
 an observation to an action, Q-Learning attempts to learn the value of being in a
 given state, and taking a specific action there < Q(s, a) >.

Problem:
Frozen lake from OpenAI gym. Consists of a 4*4 grid of blocks. each one either <4 blocks each having>:
start block, goal block, safe frozen block, or a dangerous block

Objective:
learn the object to navigate from start to goal without moving to a hole.

Actions: up, down, left, right
states: 16

reward:  At every step is 0. except for entering the goal with reward of 1!

In it’s simplest implementation, Q-Learning is a table of values for every state (row)
and action (column) possible in the environment. Within each cell of the table,
we learn a value for how good it is to take a given action within a given state.
our Q table will be 16 * 4

Initialize the table to zeros and update as rewards are observed!
'''
# load environment
env = gym.make('FrozenLake-v0')

# Implementing Q-Table learning algorithm
# Initialize the table:
Q = np.zeros([env.observation_space.n, env.action_space.n])

# set learning parameters:
lr = 0.8
decay = 0.95
num_episodes = 2000

# Create a list to contain total rewards  per episodes
rList = []

for i in range(num_episodes):

    print('Episode= ', i)

    # reset env and get a new initial observation
    s = env.reset()
    rAll = 0
    f = False
    j = 0

    # The Q table Learning algorithm

    while j < 99 :

        j += 1

        # choose an action by greedily(with noise) picking from Q table
        # reduce randomness as we progress ( i increases)

        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1./ (i + 1)))

        # get new state and reward from env:
        s1, r, done, _ = env.step(a)

        # update Q table with new knowledge < keep a moving average of Q(s, a) around and update!

        Q[s, a] = (1 - lr) * Q[s, a] + lr * (r + decay * np.max(Q[s1, :]))

        rAll += r
        s = s1

        # if goal block or hole is reached!
        if done == True:
            break
    rList.append(rAll)

print ("Average Score over time: " +  str(sum(rList)/num_episodes) )

print('Final Q- table values')
print(Q)

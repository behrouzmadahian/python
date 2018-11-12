import gym

'''
OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. 
It makes no assumptions about the structure of your agent, and is compatible with any
 numerical library such as tensorflow.
'''
# Running an environment:
# some example environments:
# 'CartPole-v0','MsPacman-v0', 'Hopper-v1', 'MountainCar-v0'

env = gym.make('CartPole-v0')
env.reset()

for episode in range(20):

    observation = env.reset()

    for t in range(1000):

        env.render()
        #print(observation)
        print('=')
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward)

        if done:
            print('Episode finished after {} timesteps'.format(t + 1))
            break

# env.step():
# returns 4 values:
# 1. observation(object) eg. pixel data from a camera, joint angles of a robot, state of a board game
# 2. reward(float): amount of reward achieved by the action.
# 3. done(bool): whether it's time to reset the environment again.  done being true means episode is terminated.
# 4. info(dict): diagnostic info useful for debugging

# env.reset():
# resets the environment and returns an initial state.

# spaces:
# Every environment comes with first-class Space objects that describe the valid actions and observations

# CartPole: valid actions are 0, 1 for left or right
print('Action Space for CartPole env:')
print(env.action_space)

# box space represents and n-dimensional box, so valid observations will be an array of 4 numbers.
# (x, x', theta, theta')
print('observation Space for CartPole env')
print(env.observation_space)

# box bounds:
print('Observations Upper bounds=',env.observation_space.high)
print('Observations Lower bounds',env.observation_space.low)


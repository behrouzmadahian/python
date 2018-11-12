import  tensorflow as tf
import numpy as np
from matplotlib import  pyplot as plt
import gym

'''
In Cart pole example, we learned how to design a policy gradient reinforcement agent that could solve CartPole task.
Here, we re-examine this problem but this time introduce the concept of a model of the environment
that the agent can use to improve it's performance.

Schematic:
Real Environment -> Model Network
Model Network -> Policy Network
Policy network -> Real Environment

Model network:
A neural network aimed at learning the dynamics of the real environment.
e.g: in Cartpole Example, we would like the Model network to predict the next position
of the Cart given the previous position and an action.

Why this is helpful?
By learning an accurate model, we can train our agent using the model
rather than requiring to use the real environment every time.

Here, since we have the real environment in simulation to take sample from, this might seem less useful.

However, it can have huge advantage when attempting to learn policies for acting in physical world!
Note: physical environments take time to navigate and take samples from
and the physical rules of the world prevent things like easy environment resets from being feasible.

With such a model, an agent can ‘imagine’ what it might be like to move around the real environment,
and we can train a policy on this imagined environment in addition to the real one.

we are going to be using a neural network that will learn the transition dynamics between a 
previous (observation and action), and the expected (new observation, reward, and done state).

Training:
Our training procedure will involve switching between training our model using 
the real environment, and training our agent’s policy using the model environment.

By using this approach we will be able to learn a policy that allows our agent
to solve the CartPole task without actually ever training the policy on the real environment

'''
import  tensorflow as tf
import numpy as np
import pickle
from matplotlib import  pyplot as plt
import math
import gym
env = gym.make('CartPole-v0')

# hyperparameters
policy_Hsize = 16 # number of hidden layer neurons
learning_rate = 1e-2
discount = 0.99 # discount factor for reward

model_bs = 3 # Batch size when learning from model
real_bs = 3 # Batch size when learning from real environment

# Policy Network
policy_input_D = 4 # input dimensionality

policy_inp = tf.placeholder(tf.float32, [None, policy_input_D] , name = "input_x")
input_y = tf.placeholder(tf.float32, [None, 1], name = "input_y")
advantages = tf.placeholder(tf.float32, name = "reward_signal")
W1Grad = tf.placeholder(tf.float32, name = "W1_grad1")
W2Grad = tf.placeholder(tf.float32,name = "W2_grad2")
policyGrads = [W1Grad, W2Grad]


W1 = tf.get_variable("W1", shape=[policy_input_D, policy_Hsize], initializer = tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[policy_Hsize, 1], initializer = tf.contrib.layers.xavier_initializer())

layer1 = tf.nn.relu(tf.matmul(policy_inp, W1))
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()

policy_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

#loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))

loglik = input_y * tf.log(probability) + (1 - input_y) * tf.log(1- probability)
loss = -tf.reduce_mean(loglik * advantages)

newGrads = tf.gradients(loss, tvars)
updateGrads = policy_optimizer.apply_gradients(zip(policyGrads, tvars))


# Model Network
model_Hsize = 256 # model layer size

input_data = tf.placeholder(tf.float32, [None, 5])
# with tf.variable_scope('rnnlm'):
#     softmax_w = tf.get_variable("softmax_w", [mH, 50])
#     softmax_b = tf.get_variable("softmax_b", [50])

previous_observation = tf.placeholder(tf.float32, [None, 5] , name="previous_observation")
true_state = tf.placeholder(tf.float32, [None, 4], name = "true_state")
true_reward = tf.placeholder(tf.float32, [None, 1],name = "true_reward")
true_done = tf.placeholder(tf.float32, [None, 1], name = "true_done")

W1M = tf.get_variable("W1M", shape = [5, model_Hsize], initializer = tf.contrib.layers.xavier_initializer())
B1M = tf.Variable(tf.zeros([model_Hsize]), name="B1M")
W2M = tf.get_variable("W2M", shape = [model_Hsize, model_Hsize], initializer = tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(tf.zeros([model_Hsize]),name="B2M")

wO = tf.get_variable("wO", shape=[model_Hsize, 4], initializer = tf.contrib.layers.xavier_initializer())
wR = tf.get_variable("wR", shape=[model_Hsize, 1], initializer = tf.contrib.layers.xavier_initializer())
wD = tf.get_variable("wD", shape=[model_Hsize, 1], initializer = tf.contrib.layers.xavier_initializer())

bO = tf.Variable(tf.zeros([4]), name = "bO")
bR = tf.Variable(tf.zeros([1]), name = "bR")
bD = tf.Variable(tf.ones([1]), name = "bD")

layer1M = tf.nn.relu(tf.matmul(previous_observation, W1M) + B1M)

layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)

predicted_state = tf.matmul(layer2M, wO, name = "predicted_observation") + bO
#predicted_reward = tf.clip_by_value(tf.matmul(layer2M, wR, name = "predicted_reward") + bR,  -1, 1)
predicted_reward = tf.matmul(layer2M, wR, name = "predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name = "predicted_done") + bD)

predicted_observation = tf.concat([predicted_state, predicted_reward, predicted_done], 1)

state_loss = tf.square(true_state - predicted_state)

reward_loss = tf.square(true_reward - predicted_reward)

#done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1 - predicted_done, 1 - true_done)
#done_loss = - tf.log(done_loss)

done_loss = true_done * tf.log(predicted_done) + (1 - true_done) * tf.log(1 - predicted_done)
done_loss = - done_loss
model_loss = tf.reduce_mean(state_loss + done_loss + reward_loss)

model_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
updateModel = model_optimizer.minimize(model_loss)

# helper Functions
def resetGradBuffer(gradBuffer):
    ''' sets the gradient buffer to zero values'''
    for ix, grad in enumerate(gradBuffer):

        gradBuffer[ix] = grad * 0

    return gradBuffer


def discount_rewards (r):

    ''' Takes 1d Array of rewards and computes dicounted reward'''

    discounted_r = np.zeros_like(r)
    running_add = 0

    for t in reversed(range(0, r.size)):

        # rewards in the future are discounted
        # we like rewards sooner rather than later!

        # discounted_reward[t] = reward at time t + disounted futures reward going forward
        # discounted_reward[t] = r[t] + decay *r[t + 1] + decay^2 * r[t + 2] +  .. + decay^n * r[t + n]

        running_add = discount * running_add  + r[t]
        discounted_r[t] = running_add

    return discounted_r

def stepModel(sess, xs, action):

    '''uses our model to produce a new state when given a previous state and action'''

    toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])

    myPredict = sess.run(predicted_observation, feed_dict = {previous_observation: toFeed})
    reward = myPredict[:, 4]
    observation = myPredict[:, :4]

    observation[:, 0 ] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2 ] = np.clip(observation[:, 2], -0.4, 0.4)

    doneP = np.clip(myPredict[:, 5], 0, 1)

    if doneP > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False

    return observation, reward, done

xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
batch_size = real_bs

drawFromModel = False  # When set to True, will use model for policy_inp
trainTheModel = True  # Whether to train the model
trainThePolicy = False  # Whether to train the policy
switch_point = 1 # switch every 100 episodes

init = tf.global_variables_initializer()

nit = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    rendering = False # dont show the CartPole
    sess.run(init)
    observation = env.reset()
    x = observation
    gradBuffer = sess.run(tvars)
    gradBuffer = resetGradBuffer(gradBuffer)

    while episode_number <= 5000:

        # Start displaying environment once performance is acceptably high.
        if episode_number > 4000:
            env.render()
            rendering =True

        x = np.reshape(observation, [1, 4])

        tfprob = sess.run(probability, feed_dict = {policy_inp: x})
        action = 1 if np.random.uniform() < tfprob else 0

        # record various intermediates (needed later for backprop)
        xs.append(x)

        ys.append(action)

        # step the  model or real environment and get new measurements
        if drawFromModel == False:
            observation, reward, done, info = env.step(action)


        else:
            observation, reward, done = stepModel(sess, xs, action)

        reward_sum += reward

        ds.append(done * 1)
        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:

            if drawFromModel == False:
                real_episodes += 1
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs, drs, ys, ds = [], [], [], []  # reset array memory

            if trainTheModel == True:
                #print('Traininbg Model. EPISODE=', episode_number )

                actions = np.array( epy[:-1])
                state_prevs = epx[:-1, :]
                state_prevs = np.hstack([state_prevs, actions])
                state_nexts = epx[1:, :]
                rewards = np.array(epr[1:, :])
                dones = np.array(epd[1:, :])
                state_nextsAll = np.hstack([state_nexts, rewards, dones])

                feed_dict = {previous_observation: state_prevs, true_state: state_nexts, true_done: dones,
                             true_reward: rewards}
                loss, pState, _ = sess.run([model_loss, predicted_state, updateModel], feed_dict)

            if trainThePolicy == True:
                #print('Traininbg Policy. EPISODE=', episode_number )

                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = sess.run(newGrads, feed_dict={policy_inp: epx, input_y: epy, advantages: discounted_epr})

                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

            if episode_number % batch_size == 0:  # every batch size episodes, do update!
                if trainThePolicy == True:

                    sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                reward_sum = 0

            # Once the model has been trained on 100 episodes, we start alternating between training the policy
            # from the model and training the model from the real environment.
            if  (episode_number > 1000) :
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy

                    if drawFromModel == False:
                        print(' Performance on Real Environment. Episode %d. Reward %f. action: %f. mean reward %f.' % (
                            episode_number, reward_sum / real_bs, action, running_reward / real_bs))

            if drawFromModel == True:
                observation = np.random.uniform(-0.1, 0.1, [4])  # Generate reasonable starting point
                batch_size = model_bs
            else:
                observation = env.reset()
                batch_size = real_bs

print(real_episodes)



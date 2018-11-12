import tensorflow as tf
import numpy as np
import gym
from matplotlib import  pyplot as plt

'''
problem: CartPole Balancing: keep the pole upright as long as possible.

Oservations: The agent needs to know where the pole currently is and the angle at which it is balancing.
to accomplish this our NN will take an observation and use it when produces the probability of an action.

Delayed reward :
 Keeping the pole in the air as long as possible means moving in ways that will be advantageous for both
the present and the future. To accomplish this we will adjust the reward value for each observation-action
pair using a function that weighs actions over time.

Forming policy Gradient:
we now need to update our agent with more than one experience at a time < this is the strategy adopted here >.
To accomplish this, we will collect experiences in a buffer, and then 
occasionally use them to update the agent all at once. These sequences
of experience are sometimes referred to as rollouts, or experience traces.
We can’t just apply these rollouts by themselves however, we will need to 
ensure that the rewards are properly adjusted by a discount factor.

We use this modified reward as an estimation of advantage in our loss equation.

States:  (x, x', theta, theta') 
Actions: (left, right)

'''
env = gym.make('CartPole-v0')

discount_factor = 0.99

def discount_rewards (r):
    ''' Takes 1d Array of rewards and computes dicounted reward'''

    discounted_r = np.zeros_like(r)
    running_add = 0

    for t in reversed(range(0, r.size)):

        # rewards in the future are discounted
        # we like rewards sooner rather than later!

        # discounted_reward[t] = reward at time t + disounted futures reward going forward
        # discounted_reward[t] = r[t] + decay *r[t + 1] + decay^2 * r[t + 2] +  .. + decay^n * r[t + n]

        running_add = discount_factor * running_add  + r[t]
        discounted_r[t] = running_add

    return discounted_r

class Agent():
    def __init__(self, lr, s_size, a_size, h_size):

        # setting up Neural agent. agent takes in a state and generates an action
        # state here is 4 dimentional - features pertaining to the cartPole.

        self.state_in = tf.placeholder(shape =[None, s_size], dtype= tf.float32)

        W1 = tf.Variable(tf.random_normal(shape = (s_size, h_size)))
        W2 =tf.Variable(tf.random_normal(shape = (h_size, a_size)))

        hidden = tf.matmul(self.state_in, W1)
        hidden = tf.nn.relu(hidden)

        output = tf.matmul(hidden, W2)

        self.output = tf.nn.softmax(output)
        self.chosen_action = tf.argmax(self.output, 1)

        # training procedure:

        self.reward_holder = tf.placeholder(shape=[None], dtype = tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype = tf.int32)

        # hack to get correct indices of flattened output

        self.indices = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indices)

        self.loss = - tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        print('Shape of weight matrices=', tvars[0].get_shape(), tvars[1].get_shape())

        # if the agent fails fast in one episode, we might not get enough samples.
        # Thats why we define the gradient_holder, so that we can sum the gradients
        # over multiple episodes if this happens.
        # and we perform the update  Manually.
        # we will feed this gradients in feed_dict to gradient_holders
        self.gradient_holders = []
        for idx, var in enumerate(tvars):

            placeholder = tf.placeholder(dtype = tf.float32, name = str(idx) + '_holder', shape = var.get_shape())
            self.gradient_holders.append(placeholder)


        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate = lr)

        # apply gradients (in self.gradient_holders) to tvars

        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


# Training the agent
myAgent = Agent(lr = 5e-3, s_size = 4, a_size = 2, h_size = 8)
total_episodes = 10000

# max number of observations
max_ep = 999

# update the weights after this many episodes
update_frequency = 5

init = tf.global_variables_initializer()

# launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())

    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:

        # get initial state
        s = env.reset()

        running_reward = 0
        ep_history = []

        for j in range(max_ep):
            env.render() # shwing the dynamic learning

            # Probabistically take an action given network output: (1, 2)
            a_dist = sess.run(myAgent.output, feed_dict = {myAgent.state_in: [s]})[0]

            a = np.random.choice(a_dist, p = a_dist)
            a = np.argmax(a_dist == a)

            s1, r, done, info = env.step(a)
            ep_history.append([s, a, r, s1])

            s = s1
            running_reward += r

            if done :

                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])

                #print(ep_history[:10, 0])
                state_hist = np.vstack(ep_history[:, 0])

                feed_dict = {myAgent.reward_holder : ep_history[:, 2], myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: state_hist}

                grads = sess.run(myAgent.gradients , feed_dict = feed_dict)

                for idx, grad in enumerate(grads):

                    gradBuffer[idx] += grad

                # update the network
                if i % update_frequency == 0 and i != 0:

                    feed_dict = dict(zip(myAgent.gradient_holders, gradBuffer))

                    _ = sess.run(myAgent.update_batch, feed_dict = feed_dict)

                    # put the gradients to zero
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

        if i % 100 == 0:
            print('Episode=',i)
            print(np.mean(total_reward[-100:]))

        i += 1

plt.plot(total_reward)
plt.ylim(0, 400)
plt.show()
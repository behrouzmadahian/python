"""
Implementation of DeepMind's Deep Q-Learning
Double DQN
Dueling DQN : Q(s, a) = V(s) + (A(s, a) - Mean_a(A(s, a)))
# MINUS mean advantage from state s over all actions:
    The expected value of advantage MUST be zero:
    A = Q - V; V = E(Q) -> E(A) = 0
Huber loss for gradient explosion avoidance

In dueling architecture, the agent can learn which states are valuable or not, without having to learn
the effect of each action to each state  This is particularly useful in states where its actions
do not affect the environment in any relevant way.

target_Qt = rt + lambda * Q(s_t+1) ; Q(s_t+1) obtained from double DQN strategy
if terminal state is reached then target_Qt = rt
"""
import os
import random
import gym
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
# list-like container with fast appends and pops on either end
from collections import deque


class ProcessFrame(object):
    """ Resizes and converts RGB Atari frames to gray scale"""
    def __init__(self, frame_height=84, frame_width=84):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(self.processed,
                                                [self.frame_height, self.frame_width],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def process(self, session, frame):
        """
        :param session: tensorflow session object
        :param frame: a (210, 160, 3) frame in RGB
        :return: a processed (84, 84, 1) frame in gray scale
        """
        return session.run(self.processed, feed_dict={self.frame: frame})


class DQN(object):
    """Implements a Deep Q Network"""
    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001,
                 frame_height=84, frame_width=84, agent_history_length=4):
        """
        :param n_actions: int, number of possible actions
        :param hidden: int, number of filters in the final conv layer
        :param learning_rate:
        :param frame_height:
        :param frame_width:
        :param agent_history_length: int, Number of frames stacked together to create a state
        """
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.input = tf.placeholder(shape=[None, self.frame_height, self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)
        # normalizing input
        self.input_scaled = self.input / 255.

        # Convolutional layers:
        self.conv1 = tf.layers.conv2d(inputs=self.input_scaled, filters=32, kernel_size=[8, 8], strides=4,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                      padding='valid', activation=tf.nn.relu, use_bias=False, name='conv1')
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                      padding='valid', activation=tf.nn.relu, use_bias=False, name='conv2')
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                      padding='valid', activation=tf.nn.relu, use_bias=False, name='conv3')
        self.conv4 = tf.layers.conv2d(inputs=self.conv3, filters=self.hidden, kernel_size=[7, 7], strides=1,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                      padding='valid', activation=tf.nn.relu, use_bias=False, name='conv4')

        # Conv4 is [None, 1, 1, 1024]
        # splitting into value and advantage stream:
        self.value_stream, self.advantage_stream = tf.split(self.conv4, num_or_size_splits=2, axis=3)
        self.value_stream = tf.layers.flatten(self.value_stream)  # keeps batch dimension
        self.advantage_stream = tf.layers.flatten(self.advantage_stream)

        self.advantage = tf.layers.dense(inputs=self.advantage_stream, units=self.n_actions,
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                         name='advantage')
        self.value = tf.layers.dense(inputs=self.value_stream, units=1,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                     name='value')
        # Combining value and advantage into Q values as described above:
        self.q_values = self.value + tf.subtract(self.advantage,
                                                 tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.best_action = tf.argmax(self.q_values, axis=1)

        # Parameter updates:
        # target_Q according to Bellman equation:
        # target_Q(s_t, a_t) = r + gamma * max_a(Q(s_(t+1), a_(t+1)) calculated in function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # the placeholder for action that was performed is needed since we switch between exploration  and exploitation
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.action_oh = tf.one_hot(self.action, depth=self.n_actions, dtype=tf.float32)
        # Q value of the action that was performed:
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, self.action_oh), axis=1)
        # Parameter updates:
        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)


# epsilon greedy exploration exploitation strategy
# can also adopt exploration functions
class ActionGetter(object):
    """
    According to the papaer: Asynchronous methods for RL, I implement 3 epsilons annealing at different rates
    and at each iteration sample from them.
    determines an action according to an epsilon greedy strategy with annealing epsilon
    Modify the annealing and exploration as desired..
    """
    def __init__(self, n_actions, eps_initials=[1, 1, 1], eps_final=[0.1, 0.01, 0.5], eps_probs=[0.4, 0.3, 0.3],
                 eps_annealing_frames=1000000, eps_evaluation=0.0):
        """
        :param n_actions: int, number of possible actions
        :param eps_initials: float, list of initial exploration probabilies
        :param eps_final : final exploration probability after eps_annealing_frames frames
        :param eps_evaluation: float, exploration probability during evaluation
        :param eps_annealing_frame: int, number of frame over which the exploration probability is annealed
               from eps_initial to eps_final
        """
        self.n_actions = n_actions
        self.eps_initials = np.array(eps_initials)
        self.eps_final = np.array(eps_final)
        self.eps_probs = np.array(eps_probs)
        self.eps_annealing_frames = eps_annealing_frames
        self.eps_evaluation = eps_evaluation

        self.slopes = np.array([-(self.eps_initials[i] - self.eps_final[i]) / self.eps_annealing_frames
                                for i in range(len(self.eps_initials))])

    def get_action(self, session, frame_number, state, main_dqn, evaluation=False):
        """
        :param session: tensorlfow session object
        :param frame_number: int, number of current frame
        :param state: A (84, 84, 4) sequence of frames of an atari game
        :param main_dqn:  A DQN object
        :param evaluation: if true: agent being evaluated
        :return: an integer between 0 and n_actions -1 determining the action the agent performs next
        """
        if evaluation:
            cur_e = self.eps_evaluation
        else:
            if frame_number <= self.eps_annealing_frames:
                eps = self.eps_initials + self.slopes * frame_number
            else:
                eps = self.eps_final
            cur_e = np.random.choice(eps, size=1, replace=False, p=self.eps_probs)[0]
        if np.random.rand(1) < cur_e:
            return np.random.randint(0, self.n_actions)
        else:
            return session.run(main_dqn.best_action, feed_dict={main_dqn.input: [state]})[0]


"""
We want to store last 10^6 transitions in the form of [st, at, rt, terminal, s_t+1]
each state st, has 4 frames! (84, 84, 4) -> memory intensive!
st to s_t+1: remove first frame of st and add a frame to the end!
thus only save the frames, 
make sure you get the same consecutive frame for each example
make sure you don't get frames from different episodes
-> look at terminal flag, ..
"""


class ReplayMemory(object):
    """
    Replay memory that stores the last 1 million transitions and returns mini batches
    """
    def __init__(self, size=1000000, frame_height=84, frame_width=84, agent_history_length=4, batch_size=32):
        """
        :param size: int, number of the stored transitions
        :param frame_height: int,
        :param frame_width: int,
        :param agent_history_length: int, number of consecutive frame stacked together to create a state
        :param batch_size: int, number of transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # pre allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # pre allocate memory for the states and new states in a mini batch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width),
                               dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width),
                                   dtype=np.uint8)
        self.indices = np.empty(self.batch_size, np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        :param action: int, between 0, n_action -1
        :param frame: a (84, 84, 1) frame
        :param reward: float, reward of performing the action
        :param terminal: bool, stating if episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Frame has wrong dimension')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count == 0:
            raise ValueError('The replay memory is empty!')
        if index < (self.agent_history_length - 1):
            raise ValueError('Index must greater or equal to 3')
        return self.frames[index - self.agent_history_length: index, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                # self.current holds location of most recent frame added! any frame after it is not
                # from the same episode!
                if index >= self.current and (index - self.agent_history_length) <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length: index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """Returns a minibatch of self.batch_size transitions"""
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a mini batch')
        self._get_valid_indices()
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx)
            self.new_states[i] = self._get_state(idx + 1)
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], \
               np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]


# Target network and parameter update:

def train_op(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    """
     performs back propagation and retuns the loss
    :param session: tensorflow session
    :param replay_memory: a replay memory object
    :param main_dqn: a DQN object
    :param target_dqn: a DQN object
    :param batch_size: int
    :param gamma: float, discount factor
    :return: the loss for minibatch
    """
    # draw a minibatch from the replay memory
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
    # Double DQN:
    # the main network estimates which action is the best for the next state for every transition in minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input: new_states})

    # the target network estimates the q values in the next state
    qvals = session.run(target_dqn.q_values, feed_dict={target_dqn.input: new_states})
    double_q = qvals[range(batch_size), arg_q_max]

    # Multiplication with (1- terminal_flags) makes sure that if the game is over, targetQ = reward
    target_q = rewards + gamma * double_q * (1 - terminal_flags)

    # Gradient descent step to update parameters of the main network
    loss, _ = session.run([main_dqn.loss, main_dqn.update], feed_dict={main_dqn.input: states,
                                                                       main_dqn.target_q: target_q,
                                                                       main_dqn.action: actions})
    return loss


class TargetNetworkUpdater(object):
    """Copies the parameters of the main DQN to the target DQN"""
    def __init__(self, main_dqn_vars, target_dqn_vars):
        """
        :param main_dqn_vars: a list of tensorflow variables belonging to main DQN network
        :param target_dqn_vars:  a list of tensorflow variables belonging to target DQN network
        """
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops

    def update_networks(self, sess):
        """Assigns the values of the parameters of main networks to target networks"""
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)


# this function creates a gif from a sequence of frames passed to it
def generate_gif(frame_number, frames_for_gif, reward, path):
    """
    :param frame_number: int, determining the index of the current frame
    :param frames_for_gif: a sequence of (210, 160, 3) frames of an atari game in RGB
    :param reward: int, total reward of the episode that outputted as a gif
    :param path: string, path to save the gif
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1./30)


"""
If agent loses a life, there's no penalty at the current environment.
It helps the agent tremendously avoiding losing a life if you consider loss of life as end of episode.
We define terminal_life_lost flag and when a life is lost, we save the terminal flag of experience buffer
to this. this helps the agent improve its performance!
< I am not convinced why this helps!!! >
When we add terminal state to true in experience buffer for the current state,
we only make sure this state (4 frames) can appear only
in exactly this order. e.g. last frame can not appear in the middle of any any state!!
This is due to the fact that, the last frame of current state will become terminal thus there will be no next state!!
Thus, in sampling batches, this four frame can only be the next state and not the state!! 

We could do that by resetting the environment each time a life is lost but to costly to do so- 
especially we do not want to reset the environment when the first life is lost

During evaluation, at the beginning of each episode, action 1 <FIRE> is repeated for a random number of steps
between 1 and 10. this ensures that the agent starts in a different situation each time and thus 
cannot simply learn a fixed sequence of actions.
"""


class Atari(object):
    """Wrapper for the environment provided by gym"""
    def __init__(self, env_name, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(env_name)
        self.frame_processor = ProcessFrame()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames on top of each other to
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        # Set to true so that the agent starts with a 'FIRE' action when evaluating
        terminal_life_lost = True
        if evaluation:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1)  # Action 'Fire'
        processed_frame = self.frame_processor.process(sess, frame)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)
        return terminal_life_lost

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']
        processed_new_frame = self.frame_processor.process(sess, new_frame)
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2)
        self.state = new_state

        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame





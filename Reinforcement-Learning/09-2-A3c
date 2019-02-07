import threading
import time
import os
import numpy as np
import scipy.signal
import tensorflow as tf
from helper import *
from vizdoom import *

'''
Asynchronous Advantage Actor Critic Method (A3C):
    1. Uses Generalized Advatage Estimation (GAE).
     
While training is taking place, statistics on agent performance are available from Tensorboard:
tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'

Threading vs Multiprocessing:

The threading module uses threads, the multiprocessing module uses processes. The difference is that threads
run in the same memory space, while processes have separate memory. This makes it a bit harder to share 
objects between processes with multiprocessing. 
Since threads use the same memory, precautions have to be taken or two threads will write to the 
same memory at the same time. This is what the global interpreter lock is for.
 
sequence_length of RNN(step_size):
in getting experiences step size is one, since we only have one experience.
when we perform the train function of worker, the step size becomes the # of elements in the buffer(here 30)!
So sequence length is correctly defined
'''


# helper functions:
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# process Doom screen image to produce cropped and resized image
def process_frame(frame):
    """
    :param frame: a frame image of the game
    :return: flattened image as an array
    """
    s = frame[10:-10, 20:-30]
    s = scipy.misc.imresize(s, [84, 84])
    s = np.reshape(s, [np.prod(s.shape)])/255.
    return s


# discounting function used to calculate discounted returns
def discount(x, gamma):
    """
    :param x: sequence of values to be discounted
    :param gamma: discount factor
    :return: sequence of discounted sum of returns from position t to the end
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# function to initialize weights for policy and value output layers
# I would just use Xavier initializer..
def initializer(activation):
    """
    :param activation: activation function
    :return: initializers for kernel and bias
    """
    if activation == tf.nn.relu:
        bias_init = tf.constant_initializer(0.1)
        kernel_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
    else:
        bias_init = tf.zeros_initializer()
        kernel_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
    return kernel_init, bias_init


activation = tf.nn.elu


# Actor-Critic Network:
class ACNetwork(object):
    def __init__(self, s_size, a_size, scope, optimizer):
        """
        builds mode graph and instructions for calculating loss and back propagation
        :param s_size: length of flattened frame
        :param a_size: size od action space
        :param scope: score of the network: worker_i or global
        :param optimizer: optimizer used for back propagation
        """
        self.kernel_initializer, self.bias_initializer = initializer(activation)
        with tf.variable_scope(scope):
            # Input and visual encoding layers:
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.image_in = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            print('Shape of Image=', self.image_in.get_shape())

            self.conv1 = tf.layers.conv2d(inputs=self.image_in, filters=16, kernel_size=[8, 8],
                                          strides=4, padding='VALID', activation=activation,
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=32, kernel_size=[4, 4],
                                          strides=2, padding='VALID', activation=activation,
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer)
            # Flattens the input while maintaining the batch_size.
            hidden = tf.layers.dense(tf.contrib.layers.flatten(self.conv2),
                                     256,
                                     activation=activation,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer
                                     )
            # Output of dense layer: [N, 256]
            # transpose to input to rnn: (1, N, 256); < each batch consists of consecutive frames..>
            rnn_in = tf.expand_dims(hidden, [0])
            print('shape of rnn_in=', rnn_in.get_shape())
            # Recurrent layer for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            print('Size of LSTM hidden state=', lstm_cell.state_size.h)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, shape=[1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, shape=[1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            # we have one sequence so the sequence length to dynamic rnn has to be list of one element [step_size]
            self.step_size = tf.shape(self.image_in)[:1]
            print('STEP size=', self.step_size.get_shape())

            # sequence_length:
            # Used to copy-through state and zero-out outputs when past a batch element's sequence length.
            # So it's more for correctness than performance.
            # dynamic_rnn output: [batch_size, max_time, cell.output_size] if time_major=False
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=self.state_in,
                                                         sequence_length=self.step_size, time_major=False)
            print('Shape of LSTM output=', lstm_outputs.get_shape())
            lstm_c, lstm_h = lstm_state
            print(lstm_c.get_shape(), lstm_h.get_shape())
            print(lstm_c[:1, :].get_shape())
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])  # seems redundant indexing on axis =0

            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            print('Shape of flattened output of RNN=', rnn_out.get_shape())

            # Output Layers for policy and value estimations
            self.policy = tf.layers.dense(rnn_out,
                                          a_size,
                                          activation=tf.nn.softmax,
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer
                                          )
            self.value = tf.layers.dense(rnn_out,
                                         1,
                                         activation=None,
                                         kernel_initializer=self.kernel_initializer,
                                         bias_initializer=self.bias_initializer
                                         )

            # only the worker network needs ops for loss functions and gradient updating
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, depth=a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, axis=[1])

                # Loss functions.  arbitrary constant multipliers...
                # feel free to tune!
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = tf.reduce_sum(self.policy * tf.log(self.policy))  # used as regularizer!
                self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss + self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                # used to write var norms to summary files
                self.var_norms = tf.global_norm(local_vars)
                # the clipped gradients are set to:
                # grad * clip_norm / MAX(global_norm, clip_norm)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, clip_norm=40.0)
                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))


SUMMARY_PATH = "./9-summaryFolder/train_"


# Worker Agent
class Worker(object):
    def __init__(self, game, name, s_size, a_size, optimizer, model_path, global_episodes):
        """
        :param game: environment for the game
        :param name: scope name
        :param s_size: number of pixels of each frame
        :param a_size: size of the action space
        :param optimizer: optimizer used for back propagation
        :param model_path: path to save the model
        :param global_episodes: counter for global number of episodes!
        """
        self.name = "worker_" + str(name)
        self.model_path = model_path
        self.optimizer = optimizer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)  # run at the end of each episode for worker_0
        self.episode_rewards = []
        self.episode_lengths = [] 
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(SUMMARY_PATH + str(self.name))

        # Create the local copy of the network
        self.local_agent = ACNetwork(s_size, a_size, self.name, optimizer)

        # the tensorflow op to copy global parameters to local network
        # this ops is run at the initialization of worker and at the end of each experience update!
        # check with paper to see how it is done properly
        self.update_local_ops = update_target_graph('global', self.name)

        # The Below code is related to setting up the Doom environment
        game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        # One hot list of actions!
        self.actions = np.identity(a_size, dtype=bool).tolist()
        print('SELF.ACTIONS=', self.actions)
        # End Doom set-up
        self.env = game

    def train(self, rollout, sess, gamma, lmbda, bootstrap_value):
        """
        One step of update and loss calculation
        reset of RNN state:
            Each call to train updates rnn state and the self.batch_rnn_state
            after the call to train (when it becomes long enough),
            we empty our curr episode buffer but keep the rnn state
            and continue to gather experiences till end of episode is reached.
            then we call the train again, empty episode buffer and zero
            out self.batch_rnn_state at the start of new episode.

        :param rollout: [observations, as, rs, next_observations, done, values]
                        observations: list of flattened images
                        as: list of actions
                        rs: list of rewards
                        observations_next: list of next flattened frames
                        values: list of values corresponding to STATES in the observations
                        NOTE:   state s is fed in, environment returns a reward,
                                network estimates value of state: V(s_t)
        :param sess: tensorflow session
        :param gamma: discounting factor as in A3c
        :param lmbda: another discount factor used to estimate Advantage in Generalized Advantage Estimation
        :param bootstrap_value: value of last state in the experience, estimated by network.
                                assume buffer has 30 observation, last observation is:
                                [st, at, rt, s_t+1, done, v_t]
                                We do not know the value of the v_t+1
                                We need that: remember- original Advantage estimate:
                                    Grad (Policy) = Grad(logP * (rt+ gamma *V(s_t+1) - V(st))
                                Thus we run the model for s_t+1 and get the V(s_t+1)
        :return:
             for current worker:
             average value loss, average policy loss, average entropy loss,
             gradients' global norm, variables' global norm
        """
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]
        # Here, we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted rewards.
        # The advantage function uses "Generalized Advantage Estimation"
        # bootstrap value is used when the episode is not finished but experience is long enough
        # and we want to do the update but we don't know The value of the final state in the experience
        # note: discunted rewards has bootstrap_value as last element -> we remove it!
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        # residual advantages: delta = rt + gamma*V(s_t+1) - V(st)
        resid_advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        adv_dicount = gamma * lmbda
        advantages_gae = discount(resid_advantages, adv_dicount)
        # self.batch_rnn_state will get initialized before train is called in work()
        feed_dict = {self.local_agent.target_v: discounted_rewards,  # used to update value function
                     self.local_agent.inputs: np.vstack(observations),
                     self.local_agent.actions: actions,
                     self.local_agent.advantages: advantages_gae,  # used to update policy
                     self.local_agent.state_in: (self.batch_rnn_state[0], self.batch_rnn_state[1])
                     }
        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, step_size, _ = sess.run([self.local_agent.value_loss,
                                                                                self.local_agent.policy_loss,
                                                                                self.local_agent.entropy,
                                                                                self.local_agent.grad_norms,
                                                                                self.local_agent.var_norms,
                                                                                self.local_agent.state_out,
                                                                                self.local_agent.step_size,
                                                                                self.local_agent.apply_grads],
                                                                               feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n, step_size

    def work(self, max_episode_length, gamma, lmbda, sess, coord, saver):
        """
        The work process continues forever.-> we can implement iterations, ..
        :param max_episode_length: maximum episode length
        :param gamma: discount factor for calculating residual advantages
        :param lmbda discount factor for Generalized Advantage Estmation
        :param sess: tensorflow session
        :param coord: tensorflow coordinator for threads
        :param saver: save for saving parameters
        :return:
        """
        episode_count = sess.run(self.global_episodes)
        # total_steps = 0
        print("Starting worker " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            # stop the thread if exception happens
            with coord.stop_on_exception():
                while not coord.should_stop():
                    # update parameters of local network with that of master network
                    # initialize the parameters of local network from that of master network
                    sess.run(self.update_local_ops)
                    episode_buffer = []
                    episode_values = []
                    episode_frames = []
                    episode_reward = 0
                    episode_step_count = 0
                    # done = False
                    self.env.new_episode()
                    s = self.env.get_state().screen_buffer
                    # print('Shape of screen buffer from environment=', s.shape)
                    episode_frames.append(s)
                    s = process_frame(s)
                    rnn_state = self.local_agent.state_init
                    self.batch_rnn_state = rnn_state
                    while not self.env.is_episode_finished():
                        # Take an action using probabilities from policy network output.
                        a_dist, v, rnn_state = sess.run([self.local_agent.policy, self.local_agent.value,
                                                         self.local_agent.state_out],
                                                        feed_dict={self.local_agent.inputs: [s],
                                                                   self.local_agent.state_in: rnn_state})
                        a = np.random.choice(a_dist[0], p=a_dist[0])
                        a = np.argmax(a_dist == a)
                        r = self.env.make_action(self.actions[a]) / 100.0
                        done = self.env.is_episode_finished()
                        if not done:
                            s1 = self.env.get_state().screen_buffer
                            episode_frames.append(s1)  # are saved to file
                            s1 = process_frame(s1)
                        else:
                            s1 = s
                        episode_buffer.append([s, a, r, s1, done, v[0, 0]])
                        # episode values and rewards, used for printing...
                        episode_values.append(v[0, 0])
                        episode_reward += r
                        s = s1
                        # total_steps += 1
                        episode_step_count += 1

                        # max episodes is reached we break
                        if episode_step_count == max_episode_length - 1:
                            break
                        # If the episode hasn't ended, but the experience buffer is full, then we
                        # make an update step using that experience rollout.
                        if len(episode_buffer) == 30 and not done:
                            # Since we don't know what the true final return is, we "bootstrap" from our current
                            # value estimation.
                            v1, step_size = sess.run([self.local_agent.value, self.local_agent.step_size],
                                                     feed_dict={self.local_agent.inputs: [s],
                                                                self.local_agent.state_in: rnn_state})
                            v1 = v1[0, 0]
                            print('STEP size in one experience =', step_size)
                            try:
                                state_rnn_batch1 = sess.run(self.batch_rnn_state)
                            except:
                                state_rnn_batch1 = self.batch_rnn_state
                            print('Shape of Batch State=', state_rnn_batch1[1].shape, '=======================')

                            # Training
                            v_l, p_l, e_l, g_n, v_n, step_size = self.train(episode_buffer, sess, gamma, lmbda, v1)
                            print('STEP size in training =', step_size)
                            # empty the episode buffer. the new experiences in current episode will be added
                            # and self.batch_rnn_state encapsulate the memory of previous experiences in the episode
                            episode_buffer = []
                            # update the freshly updated parameters of global net to local worker
                            sess.run(self.update_local_ops)
                        if done:
                            break
                    # Update the network using the episode buffer at the end of the episode.
                    if len(episode_buffer) != 0:
                        v_l, p_l, e_l, g_n, v_n, step_size = self.train(episode_buffer, sess, gamma, lmbda, 0.0)
                        sess.run(self.update_local_ops)
                        # total_steps += 1
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_step_count)
                    self.episode_mean_values.append(np.mean(episode_values))

                    # Periodically save gifs of episodes, model parameters, and summary statistics.
                    if episode_count % 5 == 0 and episode_count != 0:
                        if self.name == 'worker_0' and episode_count % 25 == 0:
                            time_per_step = 0.05
                            images = np.array(episode_frames)
                            make_gif(images, './9-summaryFolder/frames/image' + str(episode_count) + '.gif',
                                     duration=len(images) * time_per_step, true_image=True, salience=False)

                        if episode_count % 250 == 0 and self.name == 'worker_0':
                            saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                            print("Saved Model")

                        mean_reward = np.mean(self.episode_rewards[-5:])
                        mean_length = np.mean(self.episode_lengths[-5:])
                        mean_value = np.mean(self.episode_mean_values[-5:])
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                        summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        self.summary_writer.add_summary(summary, episode_count)

                        self.summary_writer.flush()
                    # if self.name == 'worker_0':
                    #     sess.run(self.increment)
                    sess.run(self.increment)
                    episode_count += 1
                    total_episodes = sess.run(self.global_episodes)
                    assert (total_episodes <= MAX_STEPS), "MAX total steps for training reached- optimization finished"


"""
We can finish the optimization at the end of MAX_Updates
across all workers, or Updates for ONe worker as well!
During training, only weights from worker_0 is saved,
which is OK since all parameters are loaded from master network
 
"""
max_episode_length = 300
gamma = .99  # discount rate for advantage estimation and reward discounting
lmbda = 0.6
s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
a_size = 3  # Agent can move Left, Right, or Fire
load_model = False
model_path = './9-summaryFolder/model'
# Max number of episodes across all workers
# if global_step equals this, we raise exception to stop ALL threads
MAX_STEPS = 1000

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(model_path + '/Target_model/'):
    os.makedirs(model_path + '/Target_model/')
# Create a directory to save episode playback gifs to
if not os.path.exists('./9-summaryFolder/frames'):
    os.makedirs('./9-summaryFolder/frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = ACNetwork(s_size, a_size, 'global', None)  # Generate global network
    # num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    num_workers = 2  # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(DoomGame(), i, s_size, a_size, optimizer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    coord = tf.train.Coordinator()   # read on it!
    if load_model:
        # restores values of target network
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path + '/Target_model/')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate thread.

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, lmbda, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        time.sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
    # after all threads finish, I will save the parameters of target network to file!!
    target_network_W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    saver = tf.train.Saver(target_network_W)
    saver.save(sess, model_path + '/Target_model/' + 'target_network.ckpt')


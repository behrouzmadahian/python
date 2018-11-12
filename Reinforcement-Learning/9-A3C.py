import numpy as np
import tensorflow as tf
import threading
import scipy.signal
from helper import *
from vizdoom import *
import tensorflow.contrib.slim as slim
import time

'''
Asynchronous Advantage Actor Critic Method (A3C)

While training is taking place, statistics on agent performance are available from Tensorboard:

tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'

Threading vs Multiprocessing:

The threading module uses threads, the multiprocessing module uses processes. The difference is that threads
 run in the same memory space, while processes have separate memory. This makes it a bit harder to share 
 objects between processes with multiprocessing. 
 Since threads use the same memory, precautions have to be taken or two threads will write to the 
 same memory at the same time. This is what the global interpreter lock is for.
 
 sequence_length of RNN(step_size):
 in getting experiences stepsize is one, since we only have one experience.
 when w perform the train function of worker, the step size becomes the # of elements in the buffer( here 30)!
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

    s = frame[10:-10, 20:-30]
    s = scipy.misc.imresize(s, [84, 84])
    s = np.reshape(s, [np.prod(s.shape)])/255.

    return s

# discounting function used to calculate discounted returns
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis = 0)[::-1]

# function to initialize weights for policy and value output layers
# I would just use Xavier initializer..

def normalized_columns_initializer(std = 1.0):

    def _initializer(shape, dtype = None, partition_info = None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt( np.square(out).sum(axis = 0, keepdims =True) )
        return tf.constant(out)

    return _initializer

# Actor-Critic Network:
class AC_Network(object):
    def __init__(self, s_size, a_size, scope, optimizer):

        with tf.variable_scope(scope):

            # Input and visual encoding layers:
            self.inputs = tf.placeholder(shape = [None, s_size], dtype = tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape = [-1, 84, 84, 1])
            print('Shape of Image=', self.imageIn.get_shape())

            self.conv1 = slim.conv2d(inputs = self.imageIn, num_outputs = 16, kernel_size = [8, 8],
                                     stride = [4, 4], padding = 'VALID', activation_fn = tf.nn.elu)

            self.conv2 = slim.conv2d(inputs = self.conv1, num_outputs = 32, kernel_size = [4, 4],
                                     stride = [2, 2], padding = 'VALID', activation_fn = tf.nn.elu)

            hidden = slim.fully_connected(tf.contrib.layers.flatten(self.conv2),
                                          num_outputs = 256, activation_fn = tf.nn.elu)

            # Recurrent layer for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple = True)

            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)

            print('Size of LSTM hidden state=', lstm_cell.state_size.h)

            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, shape = [1, lstm_cell.state_size.c ])
            h_in = tf.placeholder(tf.float32, shape = [1, lstm_cell.state_size.h ])

            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0]) # (1, N, 256), each batch consists of consecutive frames..
            print('shape of rnn_in=', rnn_in.get_shape())

            step_size = tf.shape(self.imageIn)[:1]
            print('STEP size=', step_size.get_shape())
            self.step_size = step_size

            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            # sequence_length:
            # Used to copy-through state and zero-out outputs when past a batch element's sequence length.
            #  So it's more for correctness than performance.

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state = state_in,
                                                         sequence_length = step_size, time_major = False)

            print('Shape of LSTM output=', lstm_outputs.get_shape())
            lstm_c, lstm_h = lstm_state
            print(lstm_c.get_shape(), lstm_h.get_shape())
            print(lstm_c[:1, :].get_shape())

            self.state_out = (lstm_c[:1, :], lstm_h[:1, :]) # seems redundant indexing on axis =0

            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            print('Shape of flattened output of RNN=', rnn_out.get_shape())

            # Output Layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size, activation_fn = tf.nn.softmax,
                                               weights_initializer = normalized_columns_initializer(0.01),
                                               biases_initializer=None)

            self.value = slim.fully_connected(rnn_out, 1, activation_fn =  None,
                                               weights_initializer =  normalized_columns_initializer(1.0),
                                               biases_initializer = None)

            # only the worker network needs ops for loss functions and gradient updating
            if scope != 'global':
                self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, depth = a_size, dtype = tf.float32)
                self.target_v = tf.placeholder(shape = [None], dtype = tf.float32)
                self.advantages = tf.placeholder(shape = [None], dtype = tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, axis = [1])

                # Loss functions.  arbitary multipliers...
                # feel free to tune!

                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))

                self.entropy = tf.reduce_sum(self.policy * tf.log(self.policy)) # used as regularizer!

                self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)

                self.loss = 0.5 * self.value_loss + self.policy_loss + self.entropy * 0.01

                # Get gradients from local network using local losses

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)

                self.var_norms = tf.global_norm(local_vars)

                # the clipped gradients are set to:
                # grad * clip_norm / MAX(global_norm, clip_norm)

                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, clip_norm = 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))

# Worker Agent
class Worker():
    def __init__(self, game, name, s_size, a_size, optimizer, model_path, global_episodes):

        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.optimizer = optimizer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = [] 
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("./9-summaryFolder/train_" + str(self.number))

        # Create the local copy of the network
        self.local_AC = AC_Network(s_size, a_size, self.name, optimizer)

        # the tensorflow op to copy global paramters to local network
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

        self.actions = np.identity(a_size, dtype = bool).tolist()
        print('SELF.ACTIONS=', self.actions)
        # End Doom set-up
        self.env = game

    def train (self, rollout, sess, gamma, bootstrap_value):

        '''
        Each call to train updates the self.batch_rnn_state
        after the call to train (when it becomes long enough),
         we empty our curr episode buffer but keep the rnn state and continue to gather
        experiences till end of episode is reached. then we call the train again, empty episode buffer and zero
        out  self.batch_rnn_state at the start of new episode.
        '''

        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted rewards.
        # The advantage function uses "Generalized Advantage Estimation"

        # boot strap value is used when the episode is not finished but experience is long enough
        # and we want to do the update but we dont know what the final reward is!

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]

        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.inputs: np.vstack(observations),
                         self.local_AC.actions: actions,
                         self.local_AC.advantages: advantages,
                         self.local_AC.state_in: (self.batch_rnn_state[0], self.batch_rnn_state[1])
                     }

        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, step_size, _ = sess.run([self.local_AC.value_loss,
                                                                         self.local_AC.policy_loss,
                                                                         self.local_AC.entropy,
                                                                         self.local_AC.grad_norms,
                                                                         self.local_AC.var_norms,
                                                                         self.local_AC.state_out,
                                                                         self.local_AC.step_size,
                                                                         self.local_AC.apply_grads],
                                                                        feed_dict = feed_dict)

        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n, step_size

    def work(self, max_episode_length, gamma, sess, coord, saver):

        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():

            while not coord.should_stop():

                sess.run(self.update_local_ops) # update local network with target network
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

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                while self.env.is_episode_finished() == False:

                    # Take an action using probabilities from policy network output.

                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict = {self.local_AC.inputs: [s],
                                   self.local_AC.state_in: rnn_state})

                    a = np.random.choice(a_dist[0], p = a_dist[0])
                    a = np.argmax(a_dist == a)

                    r = self.env.make_action(self.actions[a]) / 100.0
                    done = self.env.is_episode_finished()

                    if done == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)

                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, done, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.

                    if len(episode_buffer) == 30 and done != True and episode_step_count != max_episode_length - 1:

                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.

                        v1, step_size = sess.run([self.local_AC.value,self.local_AC.step_size],
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in: rnn_state})
                        v1 = v1[0, 0]

                        print('STEP size in one experience =', step_size)
                        try:
                            state_rnn_batch1 = sess.run(self.batch_rnn_state)
                        except:
                            state_rnn_batch1 = self.batch_rnn_state

                        print('Shape of Batch State=', state_rnn_batch1[1].shape, '=======================')

                        v_l, p_l, e_l, g_n, v_n, step_size = self.train(episode_buffer, sess, gamma, v1)
                        print('STEP size in training =', step_size)

                        # empty the episode buffer. the new experiences in current episode will be added
                        # and self.batch_rnn_state encapsulate the memory of previous experiences in the episode
                        episode_buffer = []

                        # update the freshly updated parameters of global net to local worker
                        sess.run(self.update_local_ops)

                    if done == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n, step_size = self.train(episode_buffer, sess, gamma, 0.0)

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
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 3 # Agent can move Left, Right, or Fire
load_model = False
model_path = './9-summaryFolder/model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists('./9-summaryFolder/frames'):
    os.makedirs('./9-summaryFolder/frames')

with tf.device("/cpu:0"):

    global_episodes = tf.Variable(0, dtype = tf.int32, name = 'global_episodes', trainable = False)
    optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)
    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
    #num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    num_workers = 2  # Set workers to number of available CPU threads

    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(DoomGame(), i, s_size, a_size, optimizer, model_path, global_episodes))

    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator() #  read on it!
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate thread.

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        time.sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

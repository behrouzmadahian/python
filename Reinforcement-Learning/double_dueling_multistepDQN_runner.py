"""
We can finish the optimization at the end of MAX_Updates
across all workers, or Updates for ONe worker as well!
During training, only weights from worker_0 is saved,
which is OK since all parameters are loaded from master network
Global episodes only updated using worker_0
this is important, since when we want to write summaries, there's no reason to use all workers!!
and thus we keep track of episodes in worker_0
Summaries for all workers are written independently to their own folder/

"""
import threading
import time
import os
import numpy as np
import scipy.signal
import tensorflow as tf
from helper import *
from vizdoom import *
from double_dueling_multistep_DQN import *

max_episode_length = 300
gamma = 0.99  # discount rate for advantage estimation and reward discounting
s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
a_size = 3  # Agent can move Left, Right, or Fire
load_model = False
model_path = './9-summaryFolder/model'
# Max number of episodes for worker_0
# if global_step equals this, we raise exception to stop ALL threads
MAX_STEPS = 100000
N_SAVE_MODEL = 250
GLOBAL_TARGET_UPDATE_FREQ = 500
# k-step DQN
STEPS = 30
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(model_path + '/Target_model/'):
    os.makedirs(model_path + '/Target_model/')
# Create a directory to save episode playback gifs to
if not os.path.exists('./9-summaryFolder/frames'):
    os.makedirs('./9-summaryFolder/frames')
lock = threading.Lock()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = DQN(s_size, a_size, 'global', None, 256)  # Generate global network
    master_target_network = DQN(s_size, a_size, 'global_target', 256)
    # num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    num_workers = 4  # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        # Worker Agent
        workers.append(Worker(DoomGame(), i, s_size, a_size, STEPS, optimizer, model_path, global_episodes, lock))
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
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord,
                                          saver, master_target_network,
                                          GLOBAL_TARGET_UPDATE_FREQ, N_SAVE_MODEL, MAX_STEPS)
        t = threading.Thread(target=worker_work)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
    # after all threads finish, I will save the parameters of target network to file!!
    target_network_W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    saver = tf.train.Saver(target_network_W)
    saver.save(sess, model_path + '/Target_model/' + 'target_network.ckpt')

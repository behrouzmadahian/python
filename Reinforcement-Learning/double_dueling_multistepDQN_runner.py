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
from config_double_dueling_multistep_DQN import DQNConfig
dqn_config = DQNConfig()
tf.reset_default_graph()


if not os.path.exists(dqn_config.model_path):
    os.makedirs(dqn_config.model_path)
if not os.path.exists(dqn_config.model_path + '/Target_model/'):
    os.makedirs(dqn_config.model_path + '/Target_model/')
# Create a directory to save episode playback gifs to
if not os.path.exists('./9-summaryFolder/frames'):
    os.makedirs('./9-summaryFolder/frames')
lock = threading.Lock()

with tf.device("/cpu:0"):

    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    # Generate global network
    master_network = DQN(dqn_config.s_size, dqn_config.a_size, 'global', None, dqn_config.activation, 256)
    master_target_network = DQN(dqn_config.s_size, dqn_config.a_size, 'global_target', None, dqn_config.activation, 256)
    # num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    num_workers = 4  # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        # Worker Agent
        workers.append(Worker(DoomGame(), i, dqn_config.s_size,
                              dqn_config.a_size, dqn_config.STEPS, dqn_config.optimizer,
                              dqn_config.model_path, global_episodes, lock, dqn_config.activation,
                              dqn_config.summary_path))
    saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    coord = tf.train.Coordinator()   # read on it!
    if dqn_config.load_model:
        # restores values of target network
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(dqn_config.model_path + '/Target_model/')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate thread.

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(dqn_config.max_episode_length, dqn_config.gamma, sess, coord,
                                          saver, master_target_network, dqn_config.GLOBAL_TARGET_UPDATE_FREQ,
                                          dqn_config.N_SAVE_MODEL, dqn_config.max_episodes)
        t = threading.Thread(target=worker_work)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
    # after all threads finish, I will save the parameters of target network to file!!
    target_network_W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    saver = tf.train.Saver(target_network_W)
    saver.save(sess, dqn_config.model_path + '/Target_model/' + 'target_network.ckpt')

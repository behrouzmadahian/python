import tensorflow as tf


class DQNConfig(object):
    def __init__(self):
        # Control parameters
        self.max_episode_length = 300
        self.gamma = 0.99  # discount rate for advantage estimation and reward discounting
        self.s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
        self.a_size = 3  # Agent can move Left, Right, or Fire
        self.load_model = False
        self.model_path = './9-summaryFolder/model'
        # Max number of episodes for worker_0
        # if global_step equals this, we raise exception to stop ALL threads
        self.max_episodes = 100000
        self.N_SAVE_MODEL = 250
        self.GLOBAL_TARGET_UPDATE_FREQ = 500
        # k-step DQN
        self.STEPS = 30
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.summary_path = "./9-summaryFolder/train_"
        self.activation = tf.nn.relu



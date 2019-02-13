class DQNConfig(object):
    def __init__(self):
        # Control parameters
        self.MAX_EPISODE_LENGTH = 18000        # Equivalent of 5 minutes of gameplay at 60 frames per second
        self.EVAL_FREQUENCY = 200000           # Number of frames the agent sees between evaluations
        self.EVAL_STEPS = 10000                # Number of frames for one evaluation
        self.NETW_UPDATE_FREQ = 10000          # Number of chosen actions between updating the target network.
        # According to Mnih et al. 2015 this is measured in the number of
        # parameter updates (every four actions), however, in the
        # DeepMind code, it is clearly measured in the number
        # of actions the agent chooses
        self.DISCOUNT_FACTOR = 0.99            # gamma in the Bellman equation
        self.REPLAY_MEMORY_START_SIZE = 50000  # Number of completely random actions,
        # before the agent starts learning
        self.MAX_FRAMES = 30000000             # Total number of frames the agent sees
        self.MEMORY_SIZE = 1000000             # Number of transitions stored in the replay memory
        self.NO_OP_STEPS = 10                  # Number of 'NOOP' or 'FIRE' actions at the beginning of an
        # evaluation episode
        self.UPDATE_FREQ = 4                   # Every four actions a gradient descend step is performed

        self.LEARNING_RATE = 1e-5              # Set to 0.00025 in Pong for quicker results.
                                               # Hessel et al. 2017 used 0.0000625
        self.BS = 32                           # Batch size
        # Number of filters in the final convolutional layer. The output
        # has the shape (1,1,1024) which is split into two streams. Both
        # the advantage stream and value stream have the shape
        # (1,1,512). This is slightly different from the original
        # implementation but tests I did with the environment Pong
        # have shown that this way the score increases more quickly
        self.HIDDEN = 1024
        self.TRAIN = True
        self.TEST = False
        # Gifs and checkpoints will be saved here
        self.PATH = "/Users/behrouzmadahian/Desktop/python/Reinforcement-Learning/dqn_atari/output/"
        # logdir for tensorboard
        self.SUMMARIES = "/Users/behrouzmadahian/Desktop/python/Reinforcement-Learning/dqn_atari/summaries"
        self.RUNID = 'run_1'


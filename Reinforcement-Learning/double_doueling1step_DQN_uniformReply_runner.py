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
from DuelingDouble_1step_DQN_uniform_replay import *
from config_DuelingDouble_1step_DQN_uniform_replay import DQNConfig
ENV_NAME = 'BreakoutDeterministic-v4'

dqn_config = DQNConfig()
tf.reset_default_graph()

if not os.path.exists(dqn_config.PATH):
    os.makedirs(dqn_config.PATH)
if not os .path.exists(os.path.join(dqn_config.SUMMARIES, dqn_config.RUNID)):
    os.makedirs(os.path.join(dqn_config.SUMMARIES, dqn_config.RUNID))

SUMM_WRITER = tf.summary.FileWriter(os.path.join(dqn_config.SUMMARIES, dqn_config.RUNID))
atari = Atari(ENV_NAME, dqn_config.NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                atari.env.unwrapped.get_action_meanings()))

# main DQN and target DQN networks:
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, dqn_config.HIDDEN, dqn_config.LEARNING_RATE)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, dqn_config.HIDDEN)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

# Setting up tensorboard summaries for the loss, the average reward,
# the evaluation score and the network parameters to observe the learning process:

LAYER_IDS = ["conv1", "conv2", "conv3", "conv4", "denseAdvantage",
             "denseAdvantageBias", "denseValue", "denseValueBias"]

# Scalar summaries for tensorboard: loss, average reward and evaluation score
with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
    EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
    EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

# Histogramm summaries for tensorboard: parameters
with tf.name_scope('Parameters'):
    ALL_PARAM_SUMMARIES = []
    for i, Id in enumerate(LAYER_IDS):
        with tf.name_scope('mainDQN/'):
            MAIN_DQN_KERNEL = tf.summary.histogram(Id, tf.reshape(MAIN_DQN_VARS[i], shape=[-1]))
        ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)


def train():
    """Contains the training and evaluation loops"""
    my_replay_memory = ReplayMemory(size=dqn_config.MEMORY_SIZE, batch_size=dqn_config.BS)
    target_network_updater = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = ActionGetter(atari.env.action_space.n)
    with tf.Session() as sess:
        sess.run(init)
        frame_number = 0
        rewards = []
        loss_list = []
        while frame_number < dqn_config.MAX_FRAMES:
            ########################
            ####### Training #######
            ########################
            epoch_frame = 0
            while epoch_frame < dqn_config.EVAL_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                for _ in range(dqn_config.MAX_EPISODE_LENGTH):
                    action = action_getter.get_action(sess, frame_number, atari.state, MAIN_DQN, evaluation=False)
                    processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward
                    #  Store transition in the replay memory
                    my_replay_memory.add_experience(action=action,
                                                    frame=processed_new_frame[:, :, 0],
                                                    reward=reward,
                                                    terminal=terminal_life_lost)
                    if frame_number % dqn_config.UPDATE_FREQ == 0 and frame_number > dqn_config.REPLAY_MEMORY_START_SIZE:
                        loss = train_op(sess, my_replay_memory, MAIN_DQN,
                                        TARGET_DQN, dqn_config.BS, gamma=dqn_config.DISCOUNT_FACTOR)
                        loss_list.append(loss)
                    if frame_number % dqn_config.NETW_UPDATE_FREQ == 0 and frame_number > dqn_config.REPLAY_MEMORY_START_SIZE:
                        target_network_updater.update_networks(sess)

                    if terminal:
                        terminal = False
                        break

                rewards.append(episode_reward_sum)
                # Output the progress:
                if len(rewards) % 10 == 0:
                    # Scalar summaries for tensorboard
                    if frame_number > dqn_config.REPLAY_MEMORY_START_SIZE:
                        summ = sess.run(PERFORMANCE_SUMMARIES,
                                        feed_dict={LOSS_PH: np.mean(loss_list),
                                                   REWARD_PH: np.mean(rewards[-100:])})
                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []
                    # Histogramm summaries for tensorboard
                    summ_param = sess.run(PARAM_SUMMARIES)
                    SUMM_WRITER.add_summary(summ_param, frame_number)
                    print('reward_seq_len', 'frame_number', 'AVG_last_100_rewards')
                    print(len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open(dqn_config.PATH + 'rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number,
                              np.mean(rewards[-100:]), file=reward_file)
            ########################
            ###### Evaluation ######
            ########################
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0
            for _ in range(dqn_config.EVAL_STEPS):
                if terminal:
                    terminal_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    terminal = False

                # Fire (action 1), when a life was lost or the game just started,
                # so that the agent does not stand around doing nothing. When playing
                # with other environments, you might want to change this...
                # frame number has no effect in evaluation mode!
                # at reset, terminal_life_lost is set to TRUE and next line makes sure FIRE action happens @ reset!
                action = 1 if terminal_life_lost else action_getter.get_action(sess, frame_number,
                                                                               atari.state,
                                                                               MAIN_DQN,
                                                                               evaluation=True)
                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif:
                    frames_for_gif.append(new_frame)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False  # Save only the first game of the evaluation as a gif

            print("Evaluation score:\n", np.mean(eval_rewards))
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], dqn_config.PATH)
            except IndexError:
                print("No evaluation game finished")
            # Save the network parameters
            saver.save(sess, dqn_config.PATH+'/my_model', global_step=frame_number)

            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH: np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)


def test():
    gif_path = "GIF/"
    if not os.path.exists(os.path.join(dqn_config.PATH, gif_path)):
        os.makedirs(os.path.join(dqn_config.PATH, gif_path))

    if ENV_NAME == 'BreakoutDeterministic-v4':
        trained_path = dqn_config.PATH + '/my_model'
        save_file = "my_model-15845555.meta"  # Get this name from check point file!

    elif ENV_NAME == 'PongDeterministic-v4':
        trained_path = "trained/pong/"
        save_file = "my_model-3217770.meta"  # Get this name from check point file!

    action_getter = ActionGetter(atari.env.action_space.n)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(trained_path + save_file)
        saver.restore(sess, tf.train.latest_checkpoint(trained_path))
        frames_for_gif = []
        terminal_live_lost = atari.reset(sess, evaluation=True)
        episode_reward_sum = 0
        while True:
            atari.env.render()
            action = 1 if terminal_live_lost else action_getter.get_action(sess, 0, atari.state,
                                                                           MAIN_DQN,
                                                                           evaluation=True)
            processed_new_frame, reward, terminal, terminal_live_lost, new_frame = atari.step(sess, action)
            episode_reward_sum += reward
            frames_for_gif.append(new_frame)
            if terminal:
                break

        atari.env.close()
        print("The total reward is {}".format(episode_reward_sum))
        print("Creating gif...")
        generate_gif(0, frames_for_gif, episode_reward_sum, gif_path)
        print("Gif created, check the folder {}".format(gif_path))


if dqn_config.TRAIN:
    train()

if dqn_config.TEST:
    test()

r"""Train and Eval DDPG on Continuous Cartpole environment.

To run:

```bash
tensorboard --logdir $HOME/rl-logs/cartpole/ --port 2223 &

python DDPG_Cartpole.py \
  --root_dir=$HOME/rl-logs/cartpole/ \
  --num_iterations=100000 \
  --alsologtostderr
```
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from rllib.agents import ddpg_agent_ex

from rllib.environments.continuous_cartpole import ContinuousCartPoleEnv
from rllib.environments.wrappers import GymEnvSeedWrapper
from rllib.utils import trainer as rllib_trainer

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.utils import common


def create_env(discount=1.0,
               max_episode_steps=200,
               gym_env_wrappers=(),
               env_wrappers=(),
               spec_dtype_map=None):
    gym_env = ContinuousCartPoleEnv()
    return suite_gym.wrap_env(
        gym_env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map)

flags.DEFINE_string('root_dir', '~/rl-logs/cartpole_ddpg_1ml',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 1000000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
        root_dir,
        num_iterations=100000,
        seed=142,
        actor_fc_layers=(100, 100),
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=(100, 100),
        n_step_update=5,
        # Params for collect
        initial_collect_steps=1000,
        collect_steps_per_iteration=1,
        replay_buffer_capacity=100000,
        sigma=0.1,
        # Params for target update
        target_update_tau=0.01,
        target_update_period=5,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=1024,
        actor_learning_rate=0.5e-4,
        critic_learning_rate=0.5e-4,
        dqda_clipping=None,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        use_tf_functions=True,
        prioritized_replay=False,
        rank_based=False,
        remove_boundaries=True,
        # Params for eval
        num_eval_episodes=10,
        eval_interval=1000,
        # Params for checkpoints, summaries, and logging
        log_interval=1000,
        summary_interval=1000,
        summaries_flush_secs=10,
        debug_summaries=False,
        summarize_grads_and_vars=True,
        eval_metrics_callback=None,
        dir_suffix="-ddpg-5"):
    """A simple train and eval for DDPG."""
    tf.random.set_seed(seed)
    np.random.seed(seed + 1)
    seed_for_env = seed + 2

    root_dir = os.path.expanduser(root_dir)
    train_dir = root_dir + '/train' + dir_suffix
    eval_dir = root_dir + '/eval' + dir_suffix

    global_step = tf.Variable(0, name="global_step", dtype=tf.int64, trainable=False)

    # Need to set the seed in the enviroment, otherwise it uses a non-deterministic generator
    env_set_seed = GymEnvSeedWrapper(seed_for_env)
    tf_env = tf_py_environment.TFPyEnvironment(
        create_env(gym_env_wrappers=(env_set_seed,)))
    eval_tf_env = tf_py_environment.TFPyEnvironment(
        create_env(gym_env_wrappers=(env_set_seed,)))
    actor_net = actor_network.ActorNetwork(
        tf_env.time_step_spec().observation,
        tf_env.action_spec(),
        fc_layer_params=actor_fc_layers)

    critic_net = critic_network.CriticNetwork(
        (tf_env.time_step_spec().observation, tf_env.action_spec()),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        output_activation_fn=None)

    tf_agent = ddpg_agent_ex.DdpgAgentEx(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        n_step_update=n_step_update,
        sigma=sigma,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)

    trainer = rllib_trainer.Trainer(
        train_dir,
        eval_dir,
        tf_agent,
        tf_env,
        eval_tf_env,
        global_step,
        batch_size=batch_size,
        initial_collect_steps=initial_collect_steps,
        collect_steps_per_iteration=collect_steps_per_iteration,
        train_steps_per_iteration=train_steps_per_iteration,
        remove_boundaries=remove_boundaries,
        replay_buffer_capacity=replay_buffer_capacity,
        prioritized_replay=prioritized_replay,
        rank_based=rank_based,
        summaries_flush_secs=summaries_flush_secs,
        summary_interval=summary_interval,
        eval_interval=eval_interval,
        log_interval=log_interval,
        num_eval_episodes=num_eval_episodes,
        use_tf_functions=use_tf_functions)
    result = trainer.train(num_iterations)

    return result


def main(_):
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)
    return 0

if __name__ == '__main__':
    app.run(main)

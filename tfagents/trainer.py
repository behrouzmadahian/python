import gin
import time
import tensorflow as tf
from absl import logging
import shutil

from tf_agents.agents import tf_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from rllib.replay_buffers import tf_prioritized_replay_buffer
from rllib.utils.simple_metrics import SimpleMetrics


@gin.configurable(module="rllib")
class Trainer:
    def __init__(self,
                 train_dir: str,
                 eval_dir: str,
                 agent: tf_agent.TFAgent,
                 tf_env: tf_environment.TFEnvironment,
                 eval_tf_env: tf_environment.TFEnvironment,
                 train_step_counter: tf.Variable,
                 batch_size: int = 32,
                 initial_collect_steps: int = 1000,
                 collect_steps_per_iteration: int = 1,
                 train_steps_per_iteration: int = 1,
                 remove_boundaries: bool = True,
                 replay_buffer_capacity: int = 100000,
                 prioritized_replay: bool = True,
                 rank_based: bool = True,
                 summaries_flush_secs: int = 10,
                 summary_interval: int = 1000,
                 eval_interval: int = 1000,
                 log_interval: int = 1000,
                 num_eval_episodes: int = 10,
                 use_tf_functions: bool = True,
                 print_actions: bool = False,
                 profile: bool = False,
                 checkpoint: bool = False,
                 checkpoint_interval: int = 1000):
        self._agent = agent
        self._batch_size = batch_size
        self._n_step_update = agent.n_step_update
        self._use_tf_functions = use_tf_functions
        self._print_actions = print_actions
        self._remove_boundaries = remove_boundaries
        self._train_step_counter = train_step_counter
        self._num_eval_episodes = num_eval_episodes
        self._prioritized_replay = prioritized_replay
        self._summary_interval = summary_interval
        self._eval_interval = eval_interval
        self._log_interval = log_interval
        self._train_steps_per_iteration = train_steps_per_iteration
        self._train_dir = train_dir
        self._profile = profile

        agent.initialize()

        if not checkpoint:
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(eval_dir, ignore_errors=True)
            time.sleep(5)  # Give tensorboard a chance to clear previous output
        self._simple_metrics = SimpleMetrics(train_dir)

        self._train_summary_writer = tf.summary.create_file_writer(
            train_dir, flush_millis=summaries_flush_secs * 1000)
        self._train_summary_writer.set_as_default()

        self._eval_summary_writer = tf.summary.create_file_writer(
            eval_dir, flush_millis=summaries_flush_secs * 1000)
        self._eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
        ]

        self._tf_env = tf_env
        self._eval_tf_env = eval_tf_env

        self._train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]
        self._eval_policy = agent.policy
        # self._eval_policy = agent.eval_policy
        self._collect_policy = agent.collect_policy

        if prioritized_replay:
            self._replay_buffer = tf_prioritized_replay_buffer.TFPrioritizedReplayBuffer(
                data_spec=agent.collect_data_spec,
                batch_size=self._tf_env.batch_size,
                global_step=self._train_step_counter,
                max_length=replay_buffer_capacity,
                rank_based=rank_based,
                rank_segments=32,
                rank_sort_period=50)
        else:
            self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                agent.collect_data_spec,
                batch_size=self._tf_env.batch_size,
                max_length=replay_buffer_capacity)

        self._checkpoint = checkpoint
        self._checkpoint_interval = checkpoint_interval
        if checkpoint:
            self._train_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir,
                agent=agent,
                global_step=train_step_counter,
                metrics=metric_utils.MetricsGroup(self._train_metrics, 'train_metrics'))
            self._policy_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir + '/policy',
                policy=self._eval_policy,
                global_step=train_step_counter)
            self._rb_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir + '/replay_buffer',
                max_to_keep=1,
                replay_buffer=self._replay_buffer)
            self._train_checkpointer.initialize_or_restore()
            self._policy_checkpointer.initialize_or_restore()
            self._rb_checkpointer.initialize_or_restore()

        self._initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            self._tf_env,
            self._collect_policy,
            observers=[self._replay_buffer.add_batch] + self._train_metrics,
            num_steps=initial_collect_steps)

        self._collect_driver = dynamic_step_driver.DynamicStepDriver(
            self._tf_env,
            self._collect_policy,
            observers=[self._replay_buffer.add_batch] + self._train_metrics,
            num_steps=collect_steps_per_iteration)

        # Dataset generates trajectories with shape [BxTx...]
        # NB we don't use "num_parallel_calls" so that the result is deterministic,
        # and we don't use "prefetch" since the sampling probabilities can change
        # on every step.
        dataset = self._replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=self._n_step_update + 1)
        self._iterator = iter(dataset)

        if use_tf_functions:
            self._initial_collect_driver.run = common.function(self._initial_collect_driver.run, autograph=True)
            self._collect_driver.run = common.function(self._collect_driver.run, autograph=True)
            self._train_step = common.function(self._train_step, autograph=True)

        # Collect initial replay data.
        logging.info(
            'Initializing replay buffer by collecting experience for %d steps '
            'with a random policy.', self._initial_collect_driver._num_steps)
        self._initial_collect_driver.run()

        results = metric_utils.eager_compute(
            self._eval_metrics,
            self._eval_tf_env,
            self._eval_policy,
            num_episodes=self._num_eval_episodes,
            train_step=self._train_step_counter,
            summary_writer=self._eval_summary_writer,
            summary_prefix='Metrics',
        )
        metric_utils.log_metrics(self._eval_metrics)

    def train(self, num_iterations: int):

        with tf.summary.record_if(
                lambda: tf.math.equal(self._train_step_counter % self._summary_interval, 0)):
            time_step = None
            policy_state = self._collect_policy.get_initial_state(self._tf_env.batch_size)

            timed_at_step = self._train_step_counter.numpy()
            time_acc = 0

            # The training loop collects data from the environment and uses that data
            # to train the agent's neural network(s). We also periodically evaluate
            # the policy and print the current score.

            for iteration in range(num_iterations):
                start_time = time.time()
                time_step, policy_state = self._collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state,
                )
                if self._profile and iteration == 1000:
                    tf.profiler.experimental.start(self._train_dir)
                for _ in range(self._train_steps_per_iteration):
                    train_loss = self._train_step()
                if self._profile and iteration == 2000:
                    tf.profiler.experimental.stop()
                time_acc += time.time() - start_time

                if self._train_step_counter.numpy() % self._log_interval == 0:
                    logging.info('step = %d, loss = %f', self._train_step_counter.numpy(),
                                 train_loss.loss)
                    steps_per_sec = (self._train_step_counter.numpy() - timed_at_step) / time_acc
                    logging.info('%.3f steps/sec', steps_per_sec)
                    tf.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=self._train_step_counter)
                    timed_at_step = self._train_step_counter.numpy()
                    time_acc = 0

                    self._simple_metrics.add_value('loss', train_loss.loss, self._train_step_counter)
                    if hasattr(train_loss.extra, 'actor_loss'):
                        self._simple_metrics.add_value('actor_loss', train_loss.extra.actor_loss, self._train_step_counter)
                    if hasattr(train_loss.extra, 'critic_loss'):
                        self._simple_metrics.add_value('critic_loss', train_loss.extra.critic_loss, self._train_step_counter)
                    self._simple_metrics.write_csv()

                for train_metric in self._train_metrics:
                    train_metric.tf_summaries(
                        train_step=self._train_step_counter, step_metrics=self._train_metrics[:2])

                if self._checkpoint:
                    if self._train_step_counter.numpy() % self._checkpoint_interval == 0:
                        self._train_checkpointer.save(global_step=self._train_step_counter.numpy())
                    if self._train_step_counter.numpy() % self._checkpoint_interval == 0:
                        self._policy_checkpointer.save(global_step=self._train_step_counter.numpy())
                    if self._train_step_counter.numpy() % self._checkpoint_interval == 0:
                        self._rb_checkpointer.save(global_step=self._train_step_counter.numpy())

                if self._train_step_counter.numpy() % self._eval_interval == 0:
                    results = metric_utils.eager_compute(
                        self._eval_metrics,
                        self._eval_tf_env,
                        self._eval_policy,
                        num_episodes=self._num_eval_episodes,
                        train_step=self._train_step_counter,
                        summary_writer=self._eval_summary_writer,
                        summary_prefix='Metrics',
                    )
                    metric_utils.log_metrics(self._eval_metrics)

                    if self._print_actions:
                        env_state = self._eval_tf_env.reset()
                        policy_state = self._agent.policy.get_initial_state(self._eval_tf_env.batch_size)
                        i = 0
                        while not env_state.is_last():
                            i += 1
                            action_step = self._agent.policy.action(env_state, policy_state)
                            print('Action {}: {}'.format(i, action_step.action))
                            #q_values, _ = q_net(time_step.observation, time_step.step_type)
                            #print('Q-values {}: {}'.format(i, q_values.numpy()))
                            env_state = self._eval_tf_env.step(action_step.action)
                            policy_state = action_step.state

                    self._simple_metrics.add_results(results, self._train_step_counter)
                    #for i, var in enumerate(actor_net.trainable_variables):
                    #    all_metrics.add_value(f'{var.name}_{i}_norm', tf.linalg.global_norm([var]), global_step)
                    self._simple_metrics.write_csv()

            return train_loss

    def _sample_batch(self):
        if self._use_tf_functions:
            return next(self._iterator)
        else:
            return self._replay_buffer.get_next(
                sample_batch_size=self._batch_size, num_steps=self._n_step_update + 1, time_stacked=True)

    def _train_step(self):
        experience, buffer_info = self._sample_batch()

        experience_shape = tf.nest.map_structure(lambda x: tf.TensorShape([None] + x.shape[1:]), experience)
        buffer_info_shape = tf.nest.map_structure(lambda x: tf.TensorShape([None] + x.shape[1:]), buffer_info)

        if self._remove_boundaries:
            has_boundary = tf.reshape(experience.is_boundary()[:, 0], [-1])
            good_transitions = tf.reshape(tf.where(~has_boundary), [-1])
            while tf.size(good_transitions) < self._batch_size:
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(experience, experience_shape),
                                      (buffer_info, buffer_info_shape),
                                      (good_transitions, tf.TensorShape([None]))])

                # Remove transitions where the initial state is a boundary, and cut
                # the remaining ones down to batch_size. To have a sufficient number
                # of experiences, we fetch a second set from the iterator.
                extra_experience, extra_buffer_info = self._sample_batch()

                # Shuffle the second set to make it work correctly with stratified sampling.
                # Otherwise the front elements of the second set will always be preferred
                # to the back elements.
                order = tf.random.shuffle(tf.range(0, self._batch_size))
                extra_experience = tf.nest.map_structure(lambda x: tf.gather(x, order), extra_experience)
                extra_buffer_info = tf.nest.map_structure(lambda x: tf.gather(x, order), extra_buffer_info)

                experience = tf.nest.map_structure(lambda x, y: tf.concat([x, y], 0), experience, extra_experience)
                buffer_info = tf.nest.map_structure(lambda x, y: tf.concat([x, y], 0), buffer_info, extra_buffer_info)

                has_boundary = tf.reshape(experience.is_boundary()[:, 0], [-1])
                if self._prioritized_replay:
                    # Set the priorities of bad transitions to 0 so that they are not fetched again.
                    # Otherwise, as td errors go down, we will be fetching more and more invalid
                    # transitions because their original high td errors will never be updated.
                    # In practice, transitions with zero priorities might still be fetched due
                    # to numerical noise in the sum-tree structure, but that doesn't matter.
                    bad_transitions = tf.reshape(tf.where(has_boundary), [-1])
                    bad_ids = tf.gather(buffer_info.ids, bad_transitions)
                    if tf.size(bad_ids) > 0:
                        zeros = tf.reshape(tf.zeros_like(bad_ids, dtype=buffer_info.probabilities.dtype)[:, 0], [-1])
                        self._replay_buffer.set_priority(bad_ids, zeros)

                good_transitions = tf.where(~has_boundary)
                # tf.where returns a Nx1 tensor where N is the number of true elements.
                good_transitions = tf.reshape(good_transitions, [-1])[:self._batch_size]

                experience = tf.nest.map_structure(lambda x: tf.gather(x, good_transitions), experience)
                buffer_info = tf.nest.map_structure(lambda x: tf.gather(x, good_transitions), buffer_info)

        weights = None
        if self._prioritized_replay:
            # Weighed importance sampling used together with the prioritized replay buffer
            # Beta is annealed from 0.5 to 1 as step goes from 0 to 50000
            tf.debugging.assert_non_negative(buffer_info.probabilities, "probabilities are negative")
            beta = tf.math.minimum(tf.cast(1.0, buffer_info.probabilities.dtype),
                                   (tf.cast(self._train_step_counter, buffer_info.probabilities.dtype) / 50000.) * 0.5 + 0.5)
            weights = 1. / tf.math.pow(buffer_info.probabilities + 1e-8, beta)
            weights /= tf.math.reduce_max(weights)

        train_loss = self._agent.train(experience, weights)

        if self._prioritized_replay:
            td_error = train_loss.extra.td_error
            self._replay_buffer.set_priority(buffer_info.ids, tf.math.abs(td_error) + 1e-8)

        return train_loss

import os
import time

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from doom.DoomEnvironment import DoomEnvironment
from utils.visualization_helper import create_video


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'Root directory for writing summaries and checkpoints')
FLAGS = flags.FLAGS


def create_networks(tf_env):
	actor_net = ActorDistributionRnnNetwork(tf_env.observation_spec(), tf_env.action_spec(), conv_layer_params=[(16, 8, 4), (32, 4, 2)], input_fc_layer_params=(256,), lstm_size=(256,),
											output_fc_layer_params=(128,))
	value_net = ValueRnnNetwork(tf_env.observation_spec(), conv_layer_params=[(16, 8, 4), (32, 4, 2)], input_fc_layer_params=(256,), lstm_size=(256,), output_fc_layer_params=(128,),
								activation_fn=tf.nn.elu)

	return actor_net, value_net


def train_eval_doom(
		root_dir,
		# Params for collect
		num_environment_steps=30000000,
		collect_episodes_per_iteration=6,
		num_parallel_environments=36,
		replay_buffer_capacity=301,  # Per-environment
		# Params for train
		num_epochs=25,
		learning_rate=4e-4,
		# Params for eval
		num_eval_episodes=40,
		eval_interval=500,
		num_video_episodes=20,
		# Params for summaries and logging
		checkpoint_interval=500,
		log_interval=50,
		summary_interval=50,
		summaries_flush_secs=1,
		use_tf_functions=True):
	"""A simple train and eval for PPO."""

	root_dir = os.path.expanduser(root_dir)
	train_dir = os.path.join(root_dir, 'train')
	eval_dir = os.path.join(root_dir, 'eval')
	saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

	train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
	train_summary_writer.set_as_default()

	eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir, flush_millis=summaries_flush_secs * 1000)
	eval_metrics = [
		tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
		tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
	]

	global_step = tf.compat.v1.train.get_or_create_global_step()

	with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
		eval_py_env = DoomEnvironment()

		eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
		tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment([DoomEnvironment] * num_parallel_environments))
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

		actor_net, value_net = create_networks(tf_env)

		tf_agent = ppo_agent.PPOAgent(
			tf_env.time_step_spec(),
			tf_env.action_spec(),
			optimizer,
			actor_net=actor_net,
			value_net=value_net,
			num_epochs=num_epochs,
			gradient_clipping=0.5,
			entropy_regularization=1e-2,
			importance_ratio_clipping=0.2,
			use_gae=True,
			use_td_lambda_return=True
		)
		tf_agent.initialize()

		environment_steps_metric = tf_metrics.EnvironmentSteps()
		step_metrics = [
			tf_metrics.NumberOfEpisodes(),
			environment_steps_metric,
		]

		train_metrics = step_metrics + [
			tf_metrics.AverageReturnMetric(batch_size=num_parallel_environments),
			tf_metrics.AverageEpisodeLengthMetric(batch_size=num_parallel_environments),
		]

		eval_policy = tf_agent.policy
		collect_policy = tf_agent.collect_policy

		replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
			tf_agent.collect_data_spec,
			batch_size=num_parallel_environments,
			max_length=replay_buffer_capacity)

		train_checkpointer = common.Checkpointer(ckpt_dir=train_dir, agent=tf_agent, global_step=global_step, metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
		policy_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(train_dir, 'policy'), policy=eval_policy, global_step=global_step)
		saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
		train_checkpointer.initialize_or_restore()

		collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(tf_env, collect_policy, observers=[replay_buffer.add_batch] + train_metrics, num_episodes=collect_episodes_per_iteration)


		def train_step():
			trajectories = replay_buffer.gather_all()
			return tf_agent.train(experience=trajectories)


		def evaluate():
			metric_utils.eager_compute(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, global_step, eval_summary_writer, 'Metrics')
			create_video(eval_py_env, eval_tf_env, eval_policy, num_episodes=num_video_episodes, video_filename=os.path.join(eval_dir, "video_%d.mp4" % global_step_val))


		if use_tf_functions:
			# TODO(b/123828980): Enable once the cause for slowdown was identified.
			collect_driver.run = common.function(collect_driver.run, autograph=True)
			tf_agent.train = common.function(tf_agent.train, autograph=True)
			train_step = common.function(train_step)

		collect_time = 0
		train_time = 0
		timed_at_step = global_step.numpy()

		while environment_steps_metric.result() < num_environment_steps:
			start_time = time.time()
			collect_driver.run()
			collect_time += time.time() - start_time

			start_time = time.time()
			total_loss, _ = train_step()
			replay_buffer.clear()
			train_time += time.time() - start_time

			for train_metric in train_metrics:
				train_metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)

			global_step_val = global_step.numpy()

			if global_step_val % log_interval == 0:
				logging.info('step = %d, loss = %f', global_step_val, total_loss)
				steps_per_sec = ((global_step_val - timed_at_step) / (collect_time + train_time))
				logging.info('%.3f steps/sec', steps_per_sec)
				logging.info('collect_time = {}, train_time = {}'.format(collect_time, train_time))

				with tf.compat.v2.summary.record_if(True):
					tf.compat.v2.summary.scalar(name='global_steps_per_sec', data=steps_per_sec, step=global_step)

				timed_at_step = global_step_val
				collect_time = 0
				train_time = 0

			if global_step_val % eval_interval == 0 and global_step_val > 0:
				evaluate()

			if global_step_val % checkpoint_interval == 0:
				train_checkpointer.save(global_step=global_step_val)
				policy_checkpointer.save(global_step=global_step_val)
				saved_model_path = os.path.join(saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
				saved_model.save(saved_model_path)

		# One final eval before exiting.
		evaluate()


def main(_):
	logging.set_verbosity(logging.INFO)

	if FLAGS.root_dir is None:
		raise AttributeError('train_eval requires a root_dir.')

	train_eval_doom(root_dir=FLAGS.root_dir)


if __name__ == '__main__':
	flags.mark_flag_as_required('root_dir')
	app.run(main)

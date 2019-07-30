import imageio
from absl import logging
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies import tf_policy
from tf_agents.trajectories.policy_step import PolicyStep


def create_video(py_environment: PyEnvironment, tf_environment: TFEnvironment, policy: tf_policy.Base, num_episodes=10, video_filename='imageio.mp4'):
	logging.info("Generating video %s" % video_filename)
	with imageio.get_writer(video_filename, fps=60) as video:
		for episode in range(num_episodes):
			logging.info("Generating episode %d of %d" % (episode, num_episodes))

			time_step = tf_environment.reset()
			state = policy.get_initial_state(tf_environment.batch_size)

			video.append_data(py_environment.render())
			while not time_step.is_last():
				policy_step: PolicyStep = policy.action(time_step, state)
				state = policy_step.state
				time_step = tf_environment.step(policy_step.action)
				video.append_data(py_environment.render())

	logging.info("Finished video %s" % video_filename)

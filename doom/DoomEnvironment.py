import os

import cv2
import numpy as np
from tf_agents.environments import py_environment, utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from vizdoom import DoomGame


class DoomEnvironment(py_environment.PyEnvironment):

	def __init__(self):
		super().__init__()

		self._game = self.configure_doom()
		self._num_actions = self._game.get_available_buttons_size()

		self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self._num_actions - 1, name='action')
		self._observation_spec = array_spec.BoundedArraySpec(shape=(84, 84, 3), dtype=np.float32, minimum=0, maximum=1, name='observation')


	@staticmethod
	def configure_doom(config_name="basic.cfg"):
		game = DoomGame()
		directory = os.path.dirname(os.path.realpath(__file__))
		game.load_config(os.path.join(directory, config_name))
		game.init()
		return game


	def action_spec(self):
		return self._action_spec


	def observation_spec(self):
		return self._observation_spec


	def _reset(self):
		self._game.new_episode()
		return time_step.restart(self.get_screen_buffer_preprocessed())


	def _step(self, action):
		if self._game.is_episode_finished():
			# The last action ended the episode. Ignore the current action and start a new episode.
			return self.reset()

		# construct one hot encoded action as required by ViZDoom
		one_hot = [0] * self._num_actions
		one_hot[action] = 1

		# execute action and receive reward
		reward = self._game.make_action(one_hot)

		# return transition depending on game state
		if self._game.is_episode_finished():
			return time_step.termination(self.get_screen_buffer_preprocessed(), reward)
		else:
			return time_step.transition(self.get_screen_buffer_preprocessed(), reward)


	def render(self, mode='rgb_array'):
		""" Return image for rendering. """
		return self.get_screen_buffer_frame()


	def get_screen_buffer_preprocessed(self):
		"""
		Preprocess frame for agent by:
		- cutout interesting square part of screen
		- downsample cutout to 84x84 (same as used for atari games)
		- normalize images to interval [0,1]
		"""
		frame = self.get_screen_buffer_frame()
		cutout = frame[10:-10, 30:-30]
		resized = cv2.resize(cutout, (84, 84))
		return np.divide(resized, 255, dtype=np.float32)


	def get_screen_buffer_frame(self):
		""" Get current screen buffer or an empty screen buffer if episode is finished"""
		if self._game.is_episode_finished():
			return np.zeros((120, 160, 3), dtype=np.float32)
		else:
			return self._game.get_state().screen_buffer


if __name__ == "__main__":
	environment = DoomEnvironment()
	utils.validate_py_environment(environment, episodes=5)

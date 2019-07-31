## Examples for Applied Reinforcement Learning: Playing Doom with TF-Agents and PPO

In this repository, we provide the code for our [tutorial on applied reinforcement learning](https://www.arconsis.com/unternehmen/blog/reinforcement-learning-doom-with-tf-agents-and-ppo). 
We utilize [TensorFlow's TF-Agents library](https://github.com/tensorflow/agents) to build a neural network agent capable of playing the video game [Doom](https://de.wikipedia.org/wiki/Doom) from pixels. 
To train the agent, Proximal Policy Optimization (PPO) is used.

For more details, [have a look at our article](https://www.arconsis.com/unternehmen/blog/reinforcement-learning-doom-with-tf-agents-and-ppo).

### Repository Contents
This repository contains the following main components:

- **ppo_train_eval_doom_simple.py**: 
A minimal example on how to train Doom with TF-Agents and PPO.
- **ppo_train_eval_doom_extended.py**: 
A full example on how to train Doom with TF-Agents and PPO. 
This also includes logging metrics of training performance with TensorBoard and saving checkpoints. 
- **doom/DoomEnvironment.py**: 
An implementation of TF-Agents' `PyEnvironment` mapping running a [vizdoom](https://github.com/mwydmuch/ViZDoom) instance and mapping actions and the observations for our agent.
- **basic.cfg**: 
Configuration for the [basic Doom scenario](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios) with adaptions to work with our setup.

### Installation
Please refer to our [instructions in the article](https://www.arconsis.com/unternehmen/blog/reinforcement-learning-doom-with-tf-agents-and-ppo). 
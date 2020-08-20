from dol.env.gymwrapper import GymWrapper
import gym
import numpy as np
import tensorflow as tf


def test_gym_wrapper_tuple_discrete():
    gym_env = gym.make("Copy-v0")
    # action space is Tuple(Discrete(2), Discrete(2), Discrete(5))
    # observation space is Discrete(6)
    env = GymWrapper(gym_env)
    assert env.action_spec.shape == ((2,), (2,), (5,))
    assert env.observation_spec.shape == (6,)
    fields = vars(env.reset())
    assert fields["observation"].shape == (6,)
    assert fields["reward"] == [0]
    assert fields["done"] == [False]

    action = (tf.convert_to_tensor(np.eye(2)[np.random.randint(2)]), tf.convert_to_tensor(np.eye(
        2)[np.random.randint(2)]), tf.convert_to_tensor(np.eye(6)[np.random.randint(6)]))
    step = env.step(action)
    fields = vars(step)
    assert fields["observation"].shape == (6,)
    assert fields["reward"] == np.array([0])
    assert fields["done"] == [False]


def test_gym_wrapper_box():
    gym_env = gym.make("CarRacing-v0")
    # observation_space is Box(96, 96, 3)
    # action_space is Box(3,)
    env = GymWrapper(gym_env)
    assert env.observation_spec.shape == (96, 96, 3)
    assert env.action_spec.shape == (3,)
    action = env.action_spec.sample()
    action = tf.convert_to_tensor(action)
    env.reset()
    step = env.step(action)
    fields = vars(step)
    assert fields["observation"].shape == (96, 96, 3)
    assert fields["reward"].shape == (1,)
    assert fields["done"].shape == (1,)

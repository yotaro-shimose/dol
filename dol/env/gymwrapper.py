import gym
import numpy as np
from dol.env.base import ContinuousSpec, DiscreteSpec, Env, TupleSpec
from dol.core import EnvStep
import tensorflow as tf


class GymWrapper(Env):
    def __init__(self, env):
        self.env = env
        self.observation_spec = self.space_to_spec(self.env.observation_space)
        self.action_spec = self.space_to_spec(self.env.action_space)
        self.observation_parser = self.create_observation_parser(
            self.env.observation_space)
        self.action_parser = self.input_numpyize(
            self.create_action_parser(self.env.action_space))

    def step(self, action):
        action = self.action_parser(action)
        obs, reward, done, _ = self.env.step(action)
        obs = self.observation_parser(obs)
        return EnvStep(obs, np.array([reward]), np.array([done]))

    def reset(self):
        observation = self.observation_parser(self.env.reset())
        return EnvStep(observation, np.array([0]), np.array([False]))

    def space_to_spec(self, space):
        if isinstance(space, gym.spaces.Box):
            low = space.low
            high = space.high
            return ContinuousSpec(low, high)
        elif isinstance(space, gym.spaces.Discrete):
            return DiscreteSpec(space.n)
        elif isinstance(space, gym.spaces.Tuple):
            return TupleSpec(tuple(self.space_to_spec(s) for s in space.spaces))
        else:
            raise TypeError("Unsupported gym space!")

    def create_observation_parser(self, observation_space):
        if isinstance(observation_space, gym.spaces.Box):
            def parse_observation(observation):
                if isinstance(observation, float):
                    return np.array([observation])
                elif isinstance(observation, np.ndarray) and observation.shape == ():
                    return np.array([observation])
                else:
                    return observation
            return parse_observation
        elif isinstance(observation_space, gym.spaces.Discrete):
            def parse_observation(observation):
                return np.eye(observation_space.n)[observation]
            return parse_observation
        elif isinstance(observation_space, gym.spaces.Tuple):
            def parse_observation(observation):
                return tuple(self.parse_observation(obs) for obs in observation)
            return parse_observation

    def create_action_parser(self, action_space):
        if isinstance(action_space, gym.spaces.Box):
            def parse_action(action):
                return action
            return parse_action
        elif isinstance(action_space, gym.spaces.Discrete):
            def parse_action(action):
                action = np.argmax(action)
                return action
            return parse_action
        elif isinstance(action_space, gym.spaces.Tuple):
            parsers = tuple(self.create_action_parser(sp)
                            for sp in action_space)

            def parse_action(action):
                return tuple(parser(action) for parser, action in zip(parsers, action))
            return parse_action

    def input_numpyize(self, method):
        def parse_action(tensor):
            if isinstance(tensor, tf.Tensor):
                tensor = tensor.numpy()
            elif isinstance(tensor, tuple):
                tensor = tuple(t.numpy() if isinstance(
                    t, tf.Tensor) else t for t in tensor)
            action = method(tensor)

            return action
        return parse_action

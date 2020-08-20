from abc import ABC, abstractmethod
import numpy as np


class Env(ABC):
    def reset(self):
        pass

    def step(self, action):
        pass

    @property
    def observation_space(self):
        pass

    @property
    def action_space(self):
        pass


class Spec(ABC):
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

    @abstractmethod
    def sample(self):
        raise NotImplementedError


class ContinuousSpec(Spec):
    def __init__(self, low, high):
        assert (low <= high).all()
        assert low.shape == high.shape
        super().__init__(low, high, low.shape)

    def sample(self):
        return np.random.random(self.shape) * (self.high - self.low) + self.low


class DiscreteSpec(Spec):
    def __init__(self, num_action):
        self.num_action = num_action
        shape = (num_action,)
        low = np.zeros(shape)
        high = np.ones(shape)
        super().__init__(low, high, shape)

    def sample(self):
        n = np.random.randint(self.num_action)
        return np.eye(self.num_action)[n]


class TupleSpec(Spec):
    def __init__(self, specs):
        self.specs = specs
        shape = tuple(spec.shape for spec in specs)
        low = tuple(spec.low for spec in specs)
        high = tuple(spec.high for spec in specs)
        super().__init__(low, high, shape)

    def sample(self):
        return tuple(spec.sample() for spec in self.specs)

"""Basic data structures and common concepts are described in this module.
"""
from abc import ABC, abstractmethod
import ray
import copy


class Actor(ABC):

    def memorize(self, envstep, action, extra={}):
        """memorize step data.
        when you want to memorize extra step information, oveeride this method which
        create extra dict and pass it to adder

        Args:
            envstep (EnvStep): step information from environment
            action (np.ndarray): action taken by the agent
        """
        self.adder.add_step(envstep, action, extra)

    @abstractmethod
    def get_action(self, envstep):
        raise NotImplementedError

    def on_step_end(self, step, extra={}):
        pass

    def on_episode_end(self, envstep):
        pass

    @abstractmethod
    def update(self):
        raise NotImplementedError

    def run(self, num_episode=100):

        for episode in range(num_episode):
            reward = 0
            envstep = self.env.reset()
            while not envstep.done:
                action = self.get_action(envstep)
                self.memorize(envstep, action)
                envstep = self.env.step(action)
                reward += envstep.reward[0]
                # some step-wise processing like sending buffer
                self.on_step_end()
            # some episode-wise processing like logging
            self.on_episode_end(envstep)
            if episode % 10 == 0:
                print(f"Episode_reward: {reward}")


class Learner(ABC):
    def run(self):
        while True:
            self.learn()

    @abstractmethod
    def learn(self):
        raise NotImplementedError


class ParameterHolder:
    def __init__(self):
        self.weight = None

    def upload_weight(self, weight):
        self.weight = weight

    def download_weight(self):
        if self.weight:
            return self.weight
        else:
            return


class ParameterServer(ParameterHolder):
    def __init__(self):
        self.holder = ray.remote(ParameterHolder).remote()

    def upload_weight(self, weight):
        self.holder.upload_weight.remote(weight)

    def download_weight(self):
        return ray.get(self.holder.download_weight.remote())


class Builder(ABC):
    @abstractmethod
    def start(self):
        raise NotImplementedError


###################
# Data Structures #
###################


class Batch(ABC):
    """Batch is return of the replay buffer's sample() method.
    """
    pass


class EnvStep:
    def __init__(self, obs, reward, done):
        self.observation = obs
        self.reward = reward
        self.done = done


class StepData(object):
    """Data object to store SINGLE step and action information.
    Sequences are defined by holding references.
    """

    def __init__(
        self,
        observation,
        action,
        reward,
        done,
        extra={},
    ):
        """
        Args:
            observation (Union[np.ndarray, Sequence[np.ndarray]]): observation ndarray
            which DOES NOT include batch dimension.
            action (Union[np.ndarray, Sequence[np.ndarray]]): action ndarray.
            reward (float): [description]
            done (bool): [description]
        """
        self.observation = observation
        self.action = action
        self.reward = reward
        self.done = done
        self.extra = extra

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class Item(object):
    """Item object like transition or sequences.
    Each Item object is consist of ids of datas and priority for each.
    """

    def __init__(self, ids, priority):
        self.ids = ids
        self.priority = priority

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

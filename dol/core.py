"""Basic data structures and common concepts are described in this module.
"""
from abc import ABC, abstractmethod


class Actor(ABC):

    def memorize(self, envstep, action):
        """memorize step data.
        when you want to memorize extra step information, oveeride this method which
        create extra dict and pass it to adder

        Args:
            envstep (EnvStep): step information from environment
            action (np.ndarray): action taken by the agent
        """
        self.adder.add(envstep, action)

    def get_action(self, envstep):
        raise NotImplementedError

    def on_step_end(self):
        pass

    def on_episode_end(self):
        pass

    @abstractmethod
    def update(self):
        raise NotImplementedError

    def run(self, num_episode=100):
        # reset env
        envstep = self.env.reset()

        for episode in range(num_episode):
            envstep = self.env.reset()
            while not envstep.done:
                action = self.get_action(envstep)
                self.memorize(envstep, action)
                envstep = self.env.step(action)
                # some step-wise processing like sending buffer
                self.on_step_end()
            # some episode-wise processing like logging
            self.on_episode_end()


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


class Item(object):
    """Item object like transition or sequences.
    Each Item object is consist of ids of datas and priority for each.
    """

    def __init__(self, ids, priority):
        self.ids = ids
        self.priority = priority

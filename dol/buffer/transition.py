from dol.buffer.base import ReplayBuffer
from dol.core import Batch
from dol.utils import stack_to_batch
import numpy as np


class TransitionBatch(Batch):
    NUM_ELEMENTS = 7

    def __init__(
        self,
        _id,
        observation,
        action,
        reward,
        next_observation,
        done,
        weight
    ):
        self.id = _id
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.done = done
        self.weight = weight


class TransitionBuffer(ReplayBuffer):

    def sample(self, size):
        priorities = np.array([item.priority for item in self.items.values()])
        target = priorities ** self._alpha
        probabilities = target / np.sum(target)
        keys = np.random.choice(
            list(self.items.keys()),
            size=size,
            replace=False,
            p=probabilities
        )
        divisor = (np.sum(target) * len(self.items)) ** self._beta

        id_hash = {key: self.items[key] for key in keys}
        [_id, obs, act, reward, obs_, done, weight] = \
            [[] for _ in range(TransitionBatch.NUM_ELEMENTS)]
        for item_id, item in id_hash.items():
            r = 0
            for i in range(1, len(item.ids)):
                data = self.datas[item.ids[i]]
                r += data.reward * (self._gamma ** (i - 1))
            _id.append(item_id)
            first = self.datas[item.ids[0]]
            last = self.datas[item.ids[-1]]
            obs.append(first.observation)
            act.append(first.action)
            reward.append(r)
            obs_.append(last.observation)
            done.append(last.done)
            weight.append(item.priority / divisor)

        batch = TransitionBatch(
            _id,
            stack_to_batch(obs),
            np.array(act).reshape(-1, 1),
            np.array(reward).reshape(-1, 1),
            stack_to_batch(obs_),
            np.array(done),
            np.array(weight)
        )
        return batch

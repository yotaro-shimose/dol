from dol.buffer.base import ReplayBuffer
from dol.utils import stack_to_batch
from dol.core import Batch


class TransitionBatch(Batch):
    NUM_ELEMENT = 7

    def __init__(
        self,
        _id,
        observation,
        action,
        reward,
        next_observation,
        done,
        is_weight
    ):
        self.id = _id
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.done = done
        self.is_weight = is_weight


class TransitionBuffer(ReplayBuffer):

    def __init__(
        self,
        capacity: int = 20000,
        alpha: float = 0.6,
        beta: float = 0.4,
        gamma: float = 0.99,
        minimum_sample_size: int = 100
    ):
        self.gamma = gamma
        super().__init__(
            capacity=capacity,
            alpha=alpha,
            beta=beta,
            minimum_sample_size=minimum_sample_size
        )

    def _create_sample(self, item_ids):

        # define empty arrays to create TransitionBatch
        _id, observation, action, reward, next_observation, done, is_weight = (
            [[] for _ in range(TransitionBatch.NUM_ELEMENT)])
        # loop for items (an item corresponds to a transition)
        for item_id in item_ids:
            item = self.items[item_id]
            n_step_reward = 0
            # calculate n_step reward
            for i in range(1, len(item.ids)):
                n_step_reward += self.data[item.ids[i]
                                           ].reward * self.gamma ** (i - 1)

            first_data = self.data[item.ids[0]]
            last_data = self.data[item.ids[-1]]
            _id.append(item_id)
            observation.append(first_data.observation)
            action.append(first_data.action)
            reward.append(n_step_reward)
            next_observation.append(last_data.observation)
            done.append(last_data.done)

        # calculate is_weight
        is_weight = self.get_is_weight(_id)
        return TransitionBatch(
            _id,
            stack_to_batch(observation),
            stack_to_batch(action),
            stack_to_batch(reward),
            stack_to_batch(next_observation),
            stack_to_batch(done),
            stack_to_batch(is_weight),
        )

import numpy as np
from dol.core import Item
from abc import ABC, abstractmethod
import copy


class ReplayBuffer(ABC):
    def __init__(
        self,
        capacity: int = 20000,
        alpha: float = 0.6,
        beta: float = 0.4,
        minimum_sample_size: int = 100
    ):

        self.capacity = capacity
        self._minimum_sample_size = minimum_sample_size

        # capacity is always 2 ** integer
        self._num_leafnodes = int(2 ** np.ceil(np.log2(capacity)))

        # data storage
        self.data = {}
        self.items = []

        # hash which maps data_id to list of item_id which includes data_id
        self._item_inverse = {}

        # tree index
        self.tree_length = 2 * self._num_leafnodes - 1

        # data pointers
        self._data_pointer = 0
        self._item_pointer = 0
        self.tree = np.zeros(shape=(self.tree_length,))

        # learning parameters
        self.alpha = alpha
        self.beta = beta

    def _sample_item_ids(self, size):
        """execute importance sampling and returns list of item_ids with length of size

        Args:
            size (int): length of ids

        Returns:
            [type]: list of item_ids
        """
        if size > len(self):
            raise ValueError(
                "Batch size must be equal or smaller than size of items collected")
        explored = []
        while len(explored) < size:
            p_index = np.random.random() * self.tree[0]
            tree_index = 0
            leaf = self._explore(p_index, tree_index)
            item_id = leaf - (self._num_leafnodes - 1)
            explored.append(item_id)
        return explored

    def _all_item_ids(self):
        return list(range(len(self.items)))

    def _explore(self, p_index: float, tree_index: int):
        if tree_index >= self._num_leafnodes - 1:
            return tree_index
        left = tree_index * 2 + 1
        right = tree_index * 2 + 2
        # go left
        if self.tree[left] > p_index:
            return self._explore(p_index, left)
        # go right
        else:
            return self._explore(p_index - self.tree[left], right)

    def add_item(self, item: Item):
        # update tree(does not insert item yet)
        self._update_tree(self._item_pointer, item.priority)

        # insert item
        if len(self.items) - 1 < self._item_pointer:
            self.items.append(item)
        else:
            self.items[self._item_pointer] = item

        # update _item_inverse
        for _id in item.ids:
            self._item_inverse.setdefault(_id, []).append(self._item_pointer)
        self._item_pointer = int((self._item_pointer + 1) % self.capacity)

    def _delete_reference(self, item_id: int):
        for _id in self.items[item_id].ids:
            self._item_inverse[_id].remove(item_id)
            if len(self._item_inverse[_id]) == 0:
                self.data.pop(_id)
                del self._item_inverse[_id]

    def add_data(self, data):
        _id = self._data_pointer
        self.data[_id] = data
        self._data_pointer += 1
        return _id

    def upload(self, outer_buffer):
        """Recommended interface to add experiences from local(outer) buffer to this buffer.
        All of the data and items in outer buffer will be copied and inserted into this buffer.

        Args:
            outer_buffer (AbstractReplayBuffer): basically local buffer on worker process.
        """
        outer_buffer = copy.deepcopy(outer_buffer)
        outer_data = outer_buffer.data
        outer_items = outer_buffer.items
        # hash table from outer data id to inner data id
        data_hash = {}

        # copy data
        for outer_key, data in outer_data.items():
            inner_key = self.add_data(data)
            data_hash[outer_key] = inner_key

        for item in outer_items:
            new_ids = [data_hash[_id] for _id in item.ids]
            new_item = Item(new_ids, item.priority)
            self.add_item(new_item)
        # print(f"inner_items: {len(self.items)}")
        # print(f"len(self): {len(self)}")

    def get_max_p(self):
        return self.tree[0]

    def clear(self):
        self.data = {}
        self.items = []
        self._item_inverse = {}
        self.tree = np.zeros(shape=(self.tree_length,))
        # data pointers
        self._data_pointer = 0
        self._item_pointer = 0

    def update_priority(self, ids, priorities):
        # update item
        for _id, priority in zip(ids, priorities):
            self.items[_id].priority = priority
            # update sum tree
            self._update_tree(_id, priority)

    def _update_tree(self, item_id, new_priority):
        pointer = self._num_leafnodes - 1 + item_id
        pointer = pointer
        difference = new_priority ** self.alpha - self.tree[pointer]
        self.tree[pointer] += difference
        while pointer > 0:
            pointer = (pointer - 1) // 2
            self.tree[pointer] += difference

    def get_is_weight(self, item_ids: np.ndarray) -> np.ndarray:
        """calculate importance weight from array of item_ids

        Args:
            item_ids (np.ndarray): array of item_ids with shape (n,)

        Returns:
            is_weights: importance sampling weight as np.ndarray with the same shape as the item_ids
        """
        is_weights = []
        for item_id in item_ids:
            is_weights.append((self.items[item_id].priority ** self.alpha) /
                              (self.tree[0] * len(self)) ** self.beta)
        is_weights = np.array(is_weights)
        # normalize
        is_weights /= np.sum(is_weights)
        return is_weights.reshape(-1, 1)

    def __len__(self):
        return len(self.items)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def sample(self, size):
        """sample ids and create sample object by calling _create_sample() method.

        Args:
            List (int): list of item_ids
        """
        if size > len(self) or size < self._minimum_sample_size:
            return
        ids = self._sample_item_ids(size)
        sample = self._create_sample(ids)
        return sample

    def sample_all(self):
        ids = self._all_item_ids()
        sample = self._create_sample(ids)
        return sample

    @ abstractmethod
    def _create_sample(self, ids):
        """Create algorithm-specific data construction strategy

        Args:
            size (List[int]): list of item_ids

        Returns:
            sample: Sample object
        """
        raise NotImplementedError

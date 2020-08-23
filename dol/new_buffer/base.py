import numpy as np
from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    # TODO
    def __init__(self, capacity, alpha=0.6, beta=0.4, gamma=0.99):

        # capacity is always 2 ** integer
        self.capacity = int(2 ** np.ceil(np.log2(capacity)))

        # data storage
        self.datas = {}
        self.items = [None] * self.capacity

        # hash which maps data_id to list of item_id which includes data_id
        self._item_inverse = {}

        # tree index
        self.tree_length = 2 * self.capacity - 1

        # data pointers
        self._data_pointer = 0
        self._item_pointer = 0
        self.tree = np.zeros(shape=(self.tree_length,))

        # learning parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _sample_item(self, size):
        explored = []
        while len(explored) < size:
            p_index = np.random.random() * self.tree[0]
            tree_index = 0
            leaf = self.explore(p_index, tree_index)
            if leaf not in explored:
                explored.append(leaf - (self.capacity - 1))
        return [self.items[val] for val in explored]

    def _explore(self, p_index, tree_index):
        if tree_index >= self.capacity - 1:
            return tree_index
        left = tree_index * 2 + 1
        right = tree_index * 2 + 2
        # go left
        if self.tree[left] > p_index:
            return self._explore(p_index, left)
        # go right
        else:
            return self._explore(p_index - self.tree[left], right)

    def add_item(self, item):
        # update tree(does not insert item yet)
        self._update_tree(self._item_pointer, item.priority)

        # insert item
        self.items[self._item_pointer] = item

        # update _item_inverse
        for _id in item.ids:
            self._item_inverse.setdefault(_id, []).append(self._item_pointer)
        self._item_pointer = (self._item_pointer + 1) % self.capacity

    def _delete_reference(self, item_id):
        for _id in self.items[item_id].ids:
            self._item_inverse[_id].remove(item_id)
            if len(self._item_inverse[_id]) == 0:
                self.datas.pop(_id)
                del self._item_inverse[_id]

    def add_data(self, data):
        _id = self._data_pointer
        self.datas[_id] = data
        self._data_pointer += 1
        return _id

    def upload(self, outer_buffer):
        """Recommended interface to add experiences from local(outer) buffer to this buffer.
        All of the datas and items in outer buffer will be copied and inserted into this buffer.

        Args:
            outer_buffer (AbstractReplayBuffer): basically local buffer on worker process.
        """
        raise NotImplementedError

    def get_max_p(self):
        return self.tree[0]

    def clear(self):
        self.datas.clear()
        self.items = [None] * self.capacity
        self._item_inverse()
        self.tree = np.zeros(shape=(self.tree_length,))

    def update_priority(self, ids, priorities):
        # update item
        for _id, priority in zip(ids, priorities):
            self.items[_id].priority = priority

            # update sum tree
            self._update_tree(_id, priority)

    def _update_tree(self, item_id, new_priority):
        pointer = self.capacity - 1 + item_id
        difference = new_priority ** self.alpha - self.tree[pointer]
        while pointer > 0:
            pointer = (pointer - 1) // 2
            self.tree[pointer] += difference

    # @abstractmethod

    def sample(self, size):
        """define algorithm-specific sampling strategy

        Args:
            size (int): size of items.

        Returns:
            Batch: sample objects
        """
        raise NotImplementedError

from abc import ABC, abstractmethod
from dol.core import Item
import ray


class ReplayBuffer(ABC):
    """Abstract Replay Buffer to hold data table and reference table to some datas in data tables.
    Inherit and implement sampling strategy.
    """
    def __init__(self, n_sequence, size=20000, alpha=0.6, beta=0.4, gamma=0.99):
        self._n_sequence = n_sequence
        self._size = size
        self.items = {}
        self._item_id = 0
        self.datas = {}
        self._data_id = 0
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def add_data(self, data) -> int:
        _id = self._next_data_id()
        self.datas[_id] = data
        return _id

    def add_item(self, item: Item):
        _id = self._next_item_id()
        self.items[_id] = item

    def upload(self, outer_buffer):
        """Recommended interface to add experiences from local(outer) buffer to this buffer.
        All of the datas and items in outer buffer will be copied and inserted into this buffer.

        Args:
            outer_buffer (AbstractReplayBuffer): basically local buffer on worker process.
        """
        id_hash = {}
        for outer_id, data in outer_buffer.datas.items():
            inner_id = self.add_data(data)
            id_hash[outer_id] = inner_id

        # copy items and datas into this buffer
        for item in outer_buffer.items.values():
            ids = []
            for _id in item.ids:
                ids.append(id_hash[_id])
            new_item = Item(ids, item.priority)
            self.add_item(new_item)

    def clear(self):
        self.items = {}
        self.datas = {}

    def _next_data_id(self) -> int:
        _id = self._data_id
        self._data_id += 1
        return _id

    def _next_item_id(self) -> int:
        _id = self._item_id
        self._item_id += 1
        return _id

    def _delete_item(self, item_id):
        if not item_id:
            item = next(iter(self.items))
        else:
            item = self.items[item_id]
        for _id in item.ids:
            exist = False
            for item in self.items[1:]:
                if _id in item.ids:
                    exist = True
            if not exist:
                del self.datas[_id]

    @abstractmethod
    def sample(self, size: int):
        """define algorithm-specific sampling strategy

        Args:
            size (int): size of items.

        Returns:
            Dataset: dataset object
        """
        raise NotImplementedError

    def get_max_p(self) -> float:
        max_p = max([item.priority for item in self.items.values()])
        return max_p

    def update_priority(self, ids, priorities):
        for _id, priority in zip(ids, priorities):
            self.items[_id].priority = priority


class RemoteBuffer(object):
    def __init__(self, buffer_cls):
        self.buffer = ray.remote(buffer_cls).remote()

    def upload(self, outer_buffer):
        self.buffer.upload.remote(outer_buffer)

    def sample(self, size):
        future = self.buffer.sample.remote(size)
        return ray.get(future)

    def update_priority(self, ids, priorities):
        self.buffer.update_priority.remote(ids, priorities)

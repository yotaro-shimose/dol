from dol.new_buffer.adder import SequenceAdder
from dol.new_buffer.base import ReplayBuffer
from dol.core import EnvStep
import numpy as np
import time


class StupidBuffer(ReplayBuffer):
    def sample(self, size):
        raise NotImplementedError


def dummy_step_action():
    step = EnvStep(
        (np.random.random((3, 3)), np.random.random((4,))),
        np.random.random(),
        False
    )
    action = np.random.randint(3)
    return step, action


def test_add_step():
    capacity = 16
    n_step = 3
    buffer = StupidBuffer(capacity)
    adder = SequenceAdder(n_step, buffer)
    n_item = 20
    for _ in range(n_step - 1):
        step, action = dummy_step_action()
        adder.add_step(step, action)
        assert len(buffer) == 0
    for i in range(n_item):
        step, action = dummy_step_action()
        adder.add_step(step, action)
        if i + 1 <= capacity:
            assert (i + 1) == len(buffer)
        else:
            assert capacity == len(buffer)


def test_sumtree_calculation():
    raise NotImplementedError
    indice = range(3, 8)
    sample_size = 100
    base = 10
    last_time = None
    for index in indice:
        capa = base ** index
        buffer = ReplayBuffer(capa)
        adder = SequenceAdder(2, buffer)
        adder.add_step()
        for _id in range(sample_size):
            start = time.time()
            buffer.sample(sample_size)
            end = time.time()
        nowtime = end - start
        if last_time:
            assert nowtime < last_time * base / 2
        last_time = nowtime
        # print(f"n: {capa}  time: {nowtime}")


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def test_sumtree_priority():
    eps = 0.1
    alpha = 0.6
    n_iter = 10000
    capacity = 16
    priorities = np.arange(1, capacity + 1).astype(np.float)
    ids = np.arange(capacity)
    sumtree = StupidoBuffer(capacity, alpha)
    sampled = np.zeros((capacity,), dtype=np.int)
    for _id, priority in zip(ids, priorities):
        sumtree.add(_id, priority)
    for _ in range(n_iter):
        sample = sumtree.sample(capacity // 2)
        for item_id in sample:
            sampled[item_id] += 1
    expected = priorities ** alpha

    assert cos_sim(sampled, expected) > 1 - eps

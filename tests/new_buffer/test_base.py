from dol.new_buffer.base import PriorityTree
import numpy as np
import time


def test_sumtree_calculation():
    indice = range(3, 8)
    sample_size = 100
    base = 10
    last_time = None
    for index in indice:
        capa = base ** index
        sumtree = PriorityTree(capa)
        for _id in range(sample_size):
            sumtree.add(_id, 1)
        start = time.time()
        sumtree.sample(sample_size)
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
    sumtree = PriorityTree(capacity, alpha)
    sampled = np.zeros((capacity,), dtype=np.int)
    for _id, priority in zip(ids, priorities):
        sumtree.add(_id, priority)
    for _ in range(n_iter):
        sample = sumtree.sample(capacity // 2)
        for item_id in sample:
            sampled[item_id] += 1
    expected = priorities ** alpha

    assert cos_sim(sampled, expected) > 1 - eps

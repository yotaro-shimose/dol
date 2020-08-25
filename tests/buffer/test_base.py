from dol.buffer.adder import SequenceAdder
from dol.buffer.transition import TransitionBuffer
from dol.core import EnvStep
import numpy as np


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
    buffer = TransitionBuffer(capacity)
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


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def test_buffer_priority():
    eps = 0.1
    alpha = 0.6
    n_iter = 10000
    capacity = 16
    n_step = 2
    priorities = np.arange(1, capacity + 1).astype(np.float)
    ids = np.arange(capacity)
    buffer = TransitionBuffer(capacity, alpha, minimum_sample_size=capacity//2)
    sampled = np.zeros((capacity,), dtype=np.int)
    adder = SequenceAdder(n_step=n_step, buffer=buffer)
    for _ in range(capacity * n_step - 1):
        step, action = dummy_step_action()
        adder.add_step(step, action)
    sample = buffer.sample_all()
    buffer.update_priority(ids, priorities)

    for _ in range(n_iter):
        sample = buffer.sample(capacity // 2)
        for item_id in sample.id:
            sampled[item_id] += 1
    expected = priorities ** alpha

    assert cos_sim(sampled, expected) > 1 - eps

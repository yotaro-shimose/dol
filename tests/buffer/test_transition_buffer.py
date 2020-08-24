from dol.buffer.transition import TransitionBuffer
from dol.buffer.adder import SequenceAdder
from dol.core import EnvStep
import numpy as np
import time


def dummy_step_action():
    step = EnvStep(
        (np.random.random((3, 3)), np.random.random((4,))),
        np.random.random(),
        False
    )
    action = np.random.randint(3)
    return step, action


def test_computational_complexity():
    n_list = [1000, 10000, 100000]
    n_step = 3
    sample_size = 1000
    sampling_time = None
    for n in n_list:
        capacity = n

        buffer = TransitionBuffer(capacity, )
        adder = SequenceAdder(n_step, buffer)
        n_item = capacity
        for _ in range(n_step - 1):
            step, action = dummy_step_action()
            adder.add_step(step, action)
            assert len(buffer) == 0
        for i in range(n_item):
            step = EnvStep(
                (np.random.random((3, 3)), np.random.random((4,))),
                np.random.random(),
                False
            )
            action = np.random.randint(3)
            adder.add_step(step, action)

        start_sample = time.time()
        buffer.sample(sample_size)
        end_sample = time.time()
        if sampling_time is not None:
            assert sampling_time * 3 > end_sample - start_sample
        sampling_time = end_sample - start_sample


def test_reward_calculation():
    n_items = 1000
    n_step = 4
    n_datas = n_items + n_step - 1
    gamma = 0.5
    buffer = TransitionBuffer(gamma=gamma)
    adder = SequenceAdder(n_step, buffer)

    for _ in range(n_datas):
        obs = (np.random.random((5, 5)), np.zeros(3))
        action = np.random.randint(5)
        reward = 1
        done = False
        step = EnvStep(obs, reward, done)
        adder.add_step(step, action)
    sample = buffer.sample(100)
    assert sample.observation[0].shape == (100, 5, 5)
    assert (sample.reward.numpy() == sum(
        [(gamma ** i) * 1 for i in range(n_step - 1)])).all()
